"""
================================
Calculate simulated dipole error
================================

This example calculates the RMSE between an experimental dipole waveform
and a simulated waveform using MNE-Neuron.
"""

# Authors: Blake Caldwell <blake_caldwell@brown.edu>
#          Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>


import os.path as op
from os import environ

###############################################################################
# Let us import mne_neuron

import mne_neuron
from mne_neuron import simulate_dipole, average_dipoles, Dipole, Params, Network
from mne_neuron import get_rank, shutdown

mne_neuron_root = op.join(op.dirname(mne_neuron.__file__), '..')

###############################################################################
# Try to read the parameters and exp data via MPI

from mpi4py import MPI
import numpy as np
import nlopt
from math import ceil, floor

def split_by_evinput(params, sd_range):
    import re
    import scipy.stats as stats

    chunks = {}
    # go through all params
    for k, v in params.items():
        input_mu = re.match('^t_ev(prox|dist)_([0-9]+)', k)
        if input_mu:
            id_str = 'ev' + input_mu.group(1) + '_' + input_mu.group(2)
            if not id_str in chunks:
                chunks[id_str] = {}
            chunks[id_str]['mean'] = float(v)
            continue
        input_sigma = re.match('^sigma_t_ev(prox|dist)_([0-9]+)', k)
        if input_sigma:
            id_str = 'ev' + input_sigma.group(1) + '_' + input_sigma.group(2)
            if not id_str in chunks:
                chunks[id_str] = {}
            chunks[id_str]['sigma'] = float(v)

    # bound by sd_range sigma
    for c in chunks.keys():
        chunks[c]['start'] = max(0, chunks[c]['mean'] - sd_range * chunks[c]['sigma'])
        chunks[c]['end'] = min(float(params['tstop']), chunks[c]['mean'] + sd_range * chunks[c]['sigma'])

    sorted_chunks = sorted(chunks.items(), key=lambda x: x[1]['start'])
    print("sorted chunks:", sorted_chunks)

    def grow_chunks(input_chunks, tstart, tstop):
        grown_chunks = []

        for index, chunk in enumerate(input_chunks):
            grown_chunks.append(input_chunks[index])
            if index >= len(input_chunks) - 1:
                grown_chunks[index]['end'] = tstop
                grown_chunks[0]['start'] = tstart
                break
            grown_chunks[index]['end'] = input_chunks[index+1]['start']

        return grown_chunks

    def consolidate_chunks(input_chunks):

        consolidated_chunks = []
        for index, chunk in enumerate(input_chunks):
            if (not index == 0) and (chunk['start'] <= consolidated_chunks[-1]['end']):
                # update previous chunk
                consolidated_chunks[-1]['inputs'].extend(chunk['inputs'])
                consolidated_chunks[-1]['end'] = chunk['end']
            else:
                consolidated_chunks.append(chunk)

        return consolidated_chunks

    def insert_chunks(chunk_list, new_chunk):
#        if chunk_list is None:
        if len(chunk_list) > 0:
            for index, old_chunk in enumerate(chunk_list):
                # within
                if new_chunk['start'] >= old_chunk['start'] and \
                     new_chunk['end'] <= old_chunk['end']:
                     #chunk_list[index]['inputs'].extend(new_chunk['inputs'])
                     #return chunk_list
                     break
                # overlap
                elif new_chunk['start'] >= old_chunk['start'] and \
                       new_chunk['start'] <= old_chunk['end']:
                     # end needs to be extended
                     #chunk_list[index]['end'] =  new_chunk['end']
                     #chunk_list[index]['inputs'].extend(new_chunk['inputs'])
                     print("Warning chunk [%3.3f-%3.3f] overlaps with old_chunk %d [%3.3f-%3.3f]" % (
                           new_chunk['start'], new_chunk['end'], index, old_chunk['start'],
                           old_chunk['end']))
                     #return consolidate_chunks(chunk_list)
                     break
                elif new_chunk['end'] <= old_chunk['end'] and \
                     new_chunk['end'] >= old_chunk['start']:
                     # start needs to be extended
                     #chunk_list[index]['start'] =  new_chunk['end']
                     #chunk_list[index]['inputs'].extend(new_chunk['inputs'])
                     print("Warning chunk [%3.3f-%3.3f] overlaps with old_chunk %d [%3.3f-%3.3f]" % (
                           new_chunk['start'], new_chunk['end'], index, old_chunk['start'],
                           old_chunk['end']))
                     #return consolidate_chunks(chunk_list)
                     break
                else:
                     # outside = new chunk
                     continue
        chunk_list.append(new_chunk)


        return chunk_list

    consolidated_chunks = []
    for chunk in sorted_chunks:
        single_chunk = { 'inputs': [chunk[0]], 'start': chunk[1]['start'], 'end': chunk[1]['end'] }
        single_chunk['cdf'] = stats.norm.cdf(times, chunk[1]['mean'], chunk[1]['sigma'])
        consolidated_chunks = insert_chunks(consolidated_chunks, single_chunk)

    return grow_chunks(consolidated_chunks, 0.0, float(params['tstop']))

def spawn_simulation():
    global exp_data
    global params_input
    global subcomm

    # Initializations and preliminaries
    comm = MPI.COMM_WORLD   # get MPI communicator object
    size = comm.size        # total number of processes
    rank = comm.rank        # rank of this process
    status = MPI.Status()   # get MPI status object
    name = MPI.Get_processor_name()

    # number of processes to run nrniv with
    if 'SLURM_CPUS_ON_NODE' in environ:
        n_procs = int(environ['SLURM_CPUS_ON_NODE']) - 2
    else:
        n_procs = 4

    # limit MPI to this host only
    mpiinfo = MPI.Info().Create()
    mpiinfo.Set('host', name.split('.')[0])
    mpiinfo.Set('ompi_param', 'rmaps_base_inherit=0')
    mpiinfo.Set('ompi_param', 'rmaps_base_mapping_policy=core')
    mpiinfo.Set('ompi_param', 'rmaps_base_oversubscribe=1')
    # spawn NEURON sim
    subcomm = MPI.COMM_SELF.Spawn('nrniv',
            args=['nrniv', '-python', '-mpi', '-nobanner', 'python',
                  'examples/calculate_dipole_err.py'],
            info = mpiinfo, maxprocs=n_procs)

    # send params and exp_data to spawned nrniv procs
    simdata = (exp_data, params_input)
    subcomm.bcast(simdata, root=MPI.ROOT)

def weighted_rmse(dpl, tstart, tstop):
    exp_dpl_data = np.c_[exp_data[:,1]]
    exp_dpl = Dipole(exp_data[:,0], exp_dpl_data, data_cols=1)
    from numpy import sqrt
    from scipy import signal
    # make sure start and end times are valid for both dipoles
    exp_start_index = (np.abs(exp_dpl.t - tstart)).argmin()
    exp_end_index = (np.abs(exp_dpl.t - tstop)).argmin()
    exp_length = exp_end_index - exp_start_index

    sim_start_index = (np.abs(dpl.t - tstart)).argmin()
    sim_end_index = (np.abs(dpl.t - tstop)).argmin()
    sim_length = sim_end_index - sim_start_index

    dpl1 = dpl.dpl['agg'][sim_start_index:sim_end_index]
    dpl2 = exp_dpl.dpl['agg'][exp_start_index:exp_end_index]

    weight = current_weight[sim_start_index:sim_end_index]

    if (sim_length > exp_length):
        # downsample simulation timeseries to match exp data
        dpl1 = signal.resample(dpl1, exp_length)
        weight = signal.resample(weight, exp_length)
    elif (sim_length < exp_length):
        # downsample exp timeseries to match simulation data
        dpl2 = signal.resample(dpl2, sim_length)

    return sqrt(weight * ((dpl1 - dpl2) ** 2)).mean()


def run_remote_sim(new_params, grad=0):
    global params_input
    global subcomm

    optparams = params_input
    for k,v in zip(params_input['opt_params'].keys(), new_params):
        if v >= params_input['opt_params'][k]['minval'] and v <= params_input['opt_params'][k]['maxval']:
            optparams[k] = v
        else:
            # invalid param value -> large error
            return 1e9

    optparams['task_index'] = opt.get_numevals()

    # send new_params to spawned nrniv procs
    subcomm.bcast(optparams, root=MPI.ROOT)

    # wait to recevie results from child rank 0
    #temp_results = np.array([np.zeros(int(params_input['tstop'] / params_input['dt'] + 1)),
    #                         np.zeros(2)])
    temp_results = subcomm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)

    dpl = Dipole(times, np.c_[temp_results[0]], data_cols=1)

    rmse = temp_results[1][0]
    #subcomm.Recv(temp_results, source=MPI.ANY_SOURCE)

    print("Avg. RMSE:", rmse)
    if len(params_input['opt_inputs']) == 1:
        rmse = weighted_rmse(dpl, params_input['opt_start'], params_input['opt_end'])
        print("Weighted RMSE:", rmse)

    return rmse

def get_dipole_error(new_params, grad=0):
    global exp_data
    global params_input

    verbose=True
    optparams = params_input
    time = params_input['opt_time']
    for k,v in zip(params_input['opt_params'].keys(), new_params):
        if v >= params_input['opt_params'][k]['minval'] and v <= params_input['opt_params'][k]['maxval']:
            optparams[k] = v
        else:
            # invalid param value -> large error
            return 1e9

    avg_sim_times = []

    ###############################################################################
    # Get number of trials

    try:
        ntrials = params_input['N_trials']
    except KeyError:
        ntrials = 1

    net = Network(optparams)

    if verbose and get_rank() == 0:
        print("Running %d trials" % ntrials)

    ###############################################################################
    # Split by input times
    sd_range = 1
    chunks = split_by_evinput(optparams, sd_range)

    ###############################################################################
    # Now let's simulate the dipole

    # Start clock
    start = MPI.Wtime()


    dpls = [None]*ntrials
    for trial in range(ntrials):
        dpls[trial] = simulate_dipole(net, trial=trial,
                                      verbose=verbose)

    rmse = None
    if get_rank() == 0:
        avg_dpl = average_dipoles(dpls)
        data = np.c_[exp_data[:,1],
                     exp_data[:,1],
                     exp_data[:,1]]
        exp_dpl = Dipole(exp_data[:,0], data)

        # get errors by chunks
        for chunk in chunks:
            if time >= chunk['start'] and time <= chunk['end']:
                rmse = avg_dpl.rmse(exp_dpl, chunk['start'], chunk['end'])
                if verbose:
                    print("Inputs: %s [%3.2f - %3.2f] RMSE is %.8f \n" % (chunk['inputs'], chunk['start'], chunk['end'], rmse))
                break

        if rmse is None:
            rmse = avg_dpl.rmse(exp_dpl, 0.0, float(optparams['tstop']))
            if verbose:
                print("Total RMSE is %.8f" % rmse)

    # reset the network
    net.gid_clear()
    del net

    finish = MPI.Wtime() - start
    avg_sim_times.append(finish)
    if verbose and get_rank() == 0:
        print('took %.2fs for simulation (avg=%.2fs)' % (finish, np.mean(avg_sim_times)))

    return rmse

def set_parameters(include_weights):
    global params_input

    timing_weight_bound = 5.0
    sigma_bound = 2.0
    timing_bound = float(params_input['tstop']) * 0.06
    parameters = {}

    for name in params_input['opt_inputs']:
        param_input_name = 't_%s' % name
        mu = float(params_input[param_input_name])
        input_name = param_input_name.split('t_', 1)[1]
        if 'timing_only' in include_weights or 'timing_and_weights' in include_weights:
            timing_min = max(0, mu - timing_bound)
            timing_max = min(float(params_input['tstop']), mu + timing_bound)
            #print("Varying %s in range[%.4f-%.4f]" % (param_input_name, timing_min, timing_max))
            parameters[param_input_name] = {'initial': mu, 'minval': timing_min, 'maxval': timing_max }

            param_input_sigma_name = 'sigma_%s' % param_input_name
            sigma_value = float(params_input[param_input_sigma_name])
            sigma_min = max(0, sigma_value - sigma_value * sigma_bound)
            sigma_max = sigma_value + sigma_value * sigma_bound
            parameters[param_input_sigma_name] = {'initial': sigma_value, 'minval': sigma_min, 'maxval': sigma_max }
            
        if 'weights_only' in include_weights or 'timing_and_weights' in include_weights:
            for weight in ['L2Pyr_ampa', 'L2Pyr_nmda',
                           'L2Basket_ampa', 'L2Basket_nmda',
                           'L5Pyr_ampa', 'L5Pyr_nmda',
                           'L5Basket_ampa', 'L5Basket_nmda']:

                timing_weight_name = "gbar_%s_%s"%(input_name, weight)
                if not timing_weight_name in params_input:
                    #print("could not find variable %s in params"%timing_weight_name)
                    continue
                if input_name == 'evprox_1' and weight in ['L2Pyr_ampa', 'L2Basket_ampa', 'L2Basket_nmda', 'L5Pyr_ampa']:
                    continue
                if input_name == 'evdist_1' and weight in ['L2Pyr_ampa', 'L5Basket_ampa', 'L5Basket_nmda', 'L5Pyr_ampa']:
                    continue
                if input_name == 'evprox_2' and weight in ['L2Pyr_ampa', 'L2Pyr_nmda', 'L2Basket_ampa', 'L2Basket_nmda', 'L5Basket_ampa', 'L5Basket_nmda']:
                    continue
                timing_weight_value = float(params_input[timing_weight_name])
                if timing_weight_value == 0.:
                    weight_min = 0.
                    weight_max = 1.
                else:
                    weight_min = max(0, timing_weight_value - timing_weight_value * timing_weight_bound)
                    weight_max = timing_weight_value + timing_weight_value * timing_weight_bound

                #print("Varying %s in range[%.4f-%.4f]" % (timing_weight_name, weight_min, weight_max))
                parameters[timing_weight_name] = {'initial': timing_weight_value, 'minval': weight_min, 'maxval': weight_max }

    return parameters

def optimize(params_input, evals):
    num_params = len(params_input['opt_params'])
    opt_params = np.zeros(num_params)
    lb = np.zeros(num_params)
    ub = np.zeros(num_params)
    
    for idx, param_name in enumerate(params_input['opt_params'].keys()):
        ub[idx] = params_input['opt_params'][param_name]['maxval']
        lb[idx] = params_input['opt_params'][param_name]['minval']
        opt_params[idx] = params_input['opt_params'][param_name]['initial']
        print("label=%s initial=%3.8f lower=%3.8f upper=%3.8f" % (param_name, opt_params[idx], lb[idx], ub[idx]))
    
    if algorithm == nlopt.G_MLSL_LDS or algorithm == nlopt.G_MLSL:
        local_opt = nlopt.opt(nlopt.LN_COBYLA, num_params)
        opt.set_local_optimizer(local_opt)
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    opt.set_min_objective(run_remote_sim)
    opt.set_xtol_rel(1e-4)
    opt.set_maxeval(evals)
    x = opt.optimize(opt_params)
    minf = opt.last_optimum_value()
    print("optimum at ", x)
    print("minimum value = ", minf)
    print("result code = ", opt.last_optimize_result())

    return x

try:
    comm = MPI.Comm.Get_parent()
    rank = comm.Get_rank()

    # receive exp_data and params
    (exp_data, base_params_input) = comm.bcast(rank, root=0)

    params_input = Params().from_obj(base_params_input)

    # if run by MPI, suppress output and wait for more input
    verbose = False

except MPI.Exception:
    params_fname = environ['PARAMS_FNAME']
    params_input = Params(params_fname)

    ###############################################################################
    # Read the dipole data file to compare against

    exp_data = np.loadtxt('S1_SupraT.txt')

    verbose = True

algorithm = None
if 'ALGORITHM' in environ:
    if environ['ALGORITHM'] == "NLOPT_GN_DIRECT_L":
        algorithm = nlopt.GN_DIRECT_L
    elif environ['ALGORITHM'] == "NLOPT_GN_DIRECT":
        algorithm = nlopt.GN_DIRECT
    elif environ['ALGORITHM'] == "NLOPT_GN_CRS2_LM":
        algorithm = nlopt.GN_CRS2_LM
    elif environ['ALGORITHM'] == "NLOPT_G_MLSL_LDS":
        algorithm = nlopt.G_MLSL_LDS
    elif environ['ALGORITHM'] == "NLOPT_G_MLSL":
        algorithm = nlopt.G_MLSL
    elif environ['ALGORITHM'] == "NLOPT_GD_STOGO_RAND":
        algorithm = nlopt.GD_STOGO_RAND
    elif environ['ALGORITHM'] == "NLOPT_GD_STOGO":
        algorithm = nlopt.GD_STOGO
    elif environ['ALGORITHM'] == "NLOPT_GN_AGS":
        algorithm = nlopt.GN_AGS
    elif environ['ALGORITHM'] == "NLOPT_GN_ISRES":
        algorithm = nlopt.GN_ISRES
    elif environ['ALGORITHM'] == "NLOPT_GN_ESCH":
        algorithm = nlopt.GN_ESCH
    elif environ['ALGORITHM'] == "NLOPT_LN_PRAXIS":
        algorithm = nlopt.LN_PRAXIS
    elif environ['ALGORITHM'] == "NLOPT_LN_COBYLA":
        algorithm = nlopt.LN_COBYLA

    if algorithm is None:
        print('must specify an algorithm with the ALGORITHM environment variable')
    else:
        print('using algorithm %s' %  environ['ALGORITHM'])


#user_input_names = []
#input_name = ''
#if 'INPUT_NAME_1' in environ:
#    user_input_names.append(environ['INPUT_NAME_1'])
#    input_name = input_name + '_' + environ['INPUT_NAME_1']
#if 'INPUT_NAME_2' in environ:
#    user_input_names.append(environ['INPUT_NAME_2'])
#    input_name = input_name + '_' + environ['INPUT_NAME_2']
#if 'INPUT_NAME_3' in environ:
#    user_input_names.append(environ['INPUT_NAME_3'])
#    input_name = input_name + '_' + environ['INPUT_NAME_3']

#verbose = False

num_step = int(params_input['tstop'] / params_input['dt'] + 1)
times = np.linspace(0, params_input['tstop'], num_step)
current_weight = np.zeros(num_step)

sd_range = 1  # used for chunk['start'] and chunk['end']
chunks = split_by_evinput(params_input, sd_range)

include_weights = environ['INCLUDE_WEIGHTS']
subcomm = None
spawn_simulation()

all_inputs = []
for chunk in chunks:
    all_inputs = list(set(chunk['inputs'] + all_inputs))

for index, chunk in enumerate(chunks):
    inputs_to_opt_str = '_'.join(chunk['inputs'])
    params_input['sim_prefix'] = \
        "%s_chunk_%d_%s_%s_%s" % (
                                 op.basename(params_fname).split('.json')[0],
                                 index, inputs_to_opt_str, include_weights,
                                 environ['ALGORITHM'])

    current_weight = chunk['cdf'].copy()

    for other_chunk in chunks:
        if other_chunk['inputs'][0] == chunk['inputs'][0]:
            # don't subtract our own cdf(s)
            continue
        if params_input["t_%s"%other_chunk['inputs'][0]] < params_input["t_%s"%chunk['inputs'][0]]:
            # check ordering to only use inputs after us
            continue
        else:
            current_weight -= other_chunk['cdf']

    current_weight = np.clip(current_weight, a_min=0, a_max=1)
    # get rid of very small numbers for sqrt calculation to succeed
    indices = np.where(current_weight < 1e-4)
    current_weight[indices] = 0

    # use the weight to define start and stop points for the optimization
    params_input['opt_start'] = times[np.where(current_weight > 0.1)][0]
    params_input['opt_end'] = times[np.where(current_weight > 0.1)][-1]

    # convert to multiples of dt
    params_input['opt_start'] = floor(params_input['opt_start']/params_input['dt'])*params_input['dt']
    params_input['opt_end'] = ceil(params_input['opt_end']/params_input['dt'])*params_input['dt']

    # stop the simulation early if possible
    params_input['tstart'] = max(0,params_input['opt_start'])
    params_input['tstop'] = params_input['opt_end']

    # fill out the rest of the param ranges (uses tstop)
    params_input['opt_inputs'] = chunk['inputs']
    params_input['opt_params'] = set_parameters(include_weights)

    print('optimizing from [%3.3f-%3.3f]' % (params_input['opt_start'], params_input['opt_end']))
    num_params = len(params_input['opt_params'])
    opt = nlopt.opt(algorithm, num_params)
    opt_results = optimize(params_input, 50)

    # update params
    for var_name, value in zip(params_input['opt_params'].keys(), opt_results):
        params_input[var_name] = value

    # update prefix for purposes of writing
    params_input['sim_prefix'] = \
        "%s_best_%s_%s_%s" % (
                                 op.basename(params_fname).split('.json')[0],
                                 inputs_to_opt_str, include_weights, environ['ALGORITHM'])
    params_input['task_index'] = index
    # write params to file with task_index representing the chunk
    params_input.write(unique=False)

# last optimization
params_input['sim_prefix'] = \
    "%s_all_%s_%s_%s" % (
                      op.basename(params_fname).split('.json')[0],
                      inputs_to_opt_str, include_weights,
                      environ['ALGORITHM'])
params_input['opt_start'] = 0.0
params_input['opt_end'] = params_input['tstop'] = chunks[-1]['end']
params_input['opt_inputs'] = all_inputs
params_input['opt_params'] = set_parameters(include_weights)

print('optimizing from [%3.3f-%3.3f]' % (params_input['opt_start'], params_input['opt_end']))
num_params = len(params_input['opt_params'])
opt = nlopt.opt(algorithm, num_params)
opt_results = optimize(params_input, 100)

# update params
for var_name, value in zip(params_input['opt_params'].keys(), opt_results):
    params_input[var_name] = value

params_input['sim_prefix'] = \
    "%s_best_%s_%s_%s" % (
                                op.basename(params_fname).split('.json')[0],
                                inputs_to_opt_str, include_weights,
                                environ['ALGORITHM'])
params_input['task_index'] = index + 1
# write params to file with task_index representing the chunk
params_input.write(unique=False)

#    for i_name in user_input_names:
#        input_time = float(params_input['t_%s' % i_name]) 
#        if input_time >= chunk['start'] and input_time <= chunk['end']:
#            # this chunk is in the optimization range
#            # we need to make sure that inputs with overlapping ranges are
#            # optimized together
#            inputs_to_opt = list(set(chunk['inputs'] + inputs_to_opt))
#
#            # ordered, so only set start time if it hasn't been on a previous loop
#            if not 'opt_start' in params_input or chunk['end'] < params_input['opt_start']:
#                params_input['opt_start'] = chunk['start']
#            params_input['opt_end'] = chunk['end']
#
#            # stop the simulation early if possible
#            params_input['tstop'] = ceil(chunk['end'])
#            break

# send empty new_params to stop nrniv procs
subcomm.bcast(None, root=MPI.ROOT)

shutdown()