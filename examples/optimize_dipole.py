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


from numpy import loadtxt, mean
import os.path as op
from os import environ

###############################################################################
# Let us import mne_neuron

import mne_neuron
from mne_neuron import simulate_dipole, average_dipoles, Dipole, Params, Network
from mne_neuron import get_rank, shutdown, get_parallel_context

mne_neuron_root = op.join(op.dirname(mne_neuron.__file__), '..')

###############################################################################
# Try to read the parameters and exp data via MPI

from mpi4py import MPI
import numpy as np

def split_by_evinput(params):
    import re

    chunks = {}
    # go through all params
    for k, v in params.items():
        input_mu = re.match('^t_ev(prox|dist)_([0-9]+)', k)
        if input_mu:
            id_str = input_mu.group(1) + '_' + input_mu.group(2)
            if not id_str in chunks:
                chunks[id_str] = {}
            chunks[id_str]['mean'] = float(v)
            continue
        input_sigma = re.match('^sigma_t_ev(prox|dist)_([0-9]+)', k)
        if input_sigma:
            id_str = input_sigma.group(1) + '_' + input_sigma.group(2)
            if not id_str in chunks:
                chunks[id_str] = {}
            chunks[id_str]['sigma'] = float(v)

    # bound by 3 sigma
    for c in chunks.keys():
        chunks[c]['start'] = max(0, chunks[c]['mean'] - 3 * chunks[c]['sigma'])
        chunks[c]['end'] = min(float(params['tstop']), chunks[c]['mean'] + 3 * chunks[c]['sigma'])

    sorted_chunks = sorted(chunks.items(), key=lambda x: x[1]['start'])


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

        single_chunk = { 'inputs': new_chunk['inputs'], 'start': new_chunk['start'], 'end': new_chunk['end'] }

#        if chunk_list is None:
        if len(chunk_list) > 0:
            for index, old_chunk in enumerate(chunk_list):
                # within
                if new_chunk['start'] >= old_chunk['start'] and \
                     new_chunk['end'] <= old_chunk['end']:
                     chunk_list[index]['inputs'].extend(new_chunk['inputs'])
                     return chunk_list
                # overlap
                elif new_chunk['start'] >= old_chunk['start'] and \
                       new_chunk['start'] <= old_chunk['end']:
                     # end needs to be extended
                     chunk_list[index]['end'] =  new_chunk['end']
                     chunk_list[index]['inputs'].extend(new_chunk['inputs'])
                     return consolidate_chunks(chunk_list)
                elif new_chunk['end'] <= old_chunk['end'] and \
                     new_chunk['end'] >= old_chunk['start']:
                     # start needs to be extended
                     chunk_list[index]['start'] =  new_chunk['end']
                     chunk_list[index]['inputs'].extend(new_chunk['inputs'])
                     return consolidate_chunks(chunk_list)
                else:
                     # outside = new chunk
                     continue
        chunk_list.append(single_chunk)

        return chunk_list


    # consolidate
    consolidated_chunks = []
    for chunk in sorted_chunks:
        single_chunk = { 'inputs': [chunk[0]], 'start': chunk[1]['start'], 'end': chunk[1]['end'] }
        consolidated_chunks = insert_chunks(consolidated_chunks, single_chunk)

    return consolidated_chunks

def get_dipole_error(new_params_vector):
    global extdata
    global params_input

    optparams = params_input
    time = params_input['opt_time']
    for k,v in zip(params_input['opt_params'].keys(), new_params_vector):
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

    if verbose and get_rank() == 0:
        print("Running %d trials" % ntrials)

    net = Network(optparams, njobs=ntrials)
    ###############################################################################
    # Split by input times
    chunks = split_by_evinput(optparams)

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
        data = np.c_[extdata[:,1],
                     extdata[:,1],
                     extdata[:,1]]
        exp_dpl = Dipole(extdata[:,0], data)

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
        print('took %.2fs for simulation (avg=%.2fs)' % (finish, mean(avg_sim_times)))

    return rmse

def set_parameters(include_weights, input_names):
    global params_input

    timing_weight_bound = 5.00
    timing_bound = float(params_input['tstop']) * 0.06
    parameters = {}

    for name in input_names:
        param_input_name = 't_%s' % name
        input_times = { param_input_name: float(params_input[param_input_name]) }

        for var_name, mu in input_times.items():
            input_name = var_name.split('t_', 1)[1]

            if 'timing_only' in include_weights or 'timing_and_weights' in include_weights:
                timing_min = max(0, mu - timing_bound)
                timing_max = min(float(params_input['tstop']), mu + timing_bound)
                print("Varying %s in range[%.4f-%.4f]" % (var_name, timing_min, timing_max))
                parameters[var_name]['initial'] = mu
                parameters[var_name]['minval'] = weight_min
                parameters[var_name]['maxval'] = weight_max
            if 'weights_only' in include_weights or 'timing_and_weights' in include_weights:
                for weight in ['L2Pyr_ampa', 'L2Pyr_nmda',
                               'L2Basket_ampa', 'L2Basket_nmda',
                               'L5Pyr_ampa', 'L5Pyr_nmda',
                               'L5Basket_ampa', 'L5Basket_nmda']:

                    timing_weight_name = "gbar_%s_%s"%(input_name, weight)
                    try:
                        timing_weight_value = float(params_input[timing_weight_name])
                        if timing_weight_value == 0.:
                            weight_min = 0.
                            weight_max = 1.
                        else:
                            weight_min = max(0, timing_weight_value - timing_weight_value * timing_weight_bound)
                            weight_max = min(float(params_input['tstop']), timing_weight_value + timing_weight_value * timing_weight_bound)

                        print("Varying %s in range[%.4f-%.4f]" % (timing_weight_name, weight_min, weight_max))
                        parameters[timing_weight_name]['initial'] = mu
                        parameters[timing_weight_name]['minval'] = weight_min
                        parameters[timing_weight_name]['maxval'] = weight_max
                    except KeyError:
                        pass

    return parameters

global extdata, params_input, verbose

try:
    comm = MPI.Comm.Get_parent()
    rank = comm.Get_rank()

    # receive extdata and params
    (extdata, base_params_input) = comm.bcast(rank, root=0)

    params_input = Params().from_obj(base_params_input)

    # if run by MPI, suppress output and wait for more input
    verbose = False

except MPI.Exception:
    params_fname = op.join(mne_neuron_root, 'param', 'ERPYesSupra-1trial_sync_opt.json')
    params_input = Params(params_fname)

    ###############################################################################
    # Read the dipole data file to compare against

    extdata = loadtxt('S1_SupraT.txt')

#    verbose = True
    verbose = True

from neuron import h

input_names = []
input_name = ''
if 'INPUT_NAME_1' in environ:
    input_names.append(environ['INPUT_NAME_1'])
    input_name = input_name + '_' + environ['INPUT_NAME_1']
if 'INPUT_NAME_2' in environ:
    input_names.append(environ['INPUT_NAME_2'])
    input_name = input_name + '_' + environ['INPUT_NAME_2']
if 'INPUT_NAME_3' in environ:
    input_names.append(environ['INPUT_NAME_3'])
    input_name = input_name + '_' + environ['INPUT_NAME_3']

include_weights = environ['INCLUDE_WEIGHTS']
parameters =  set_parameters(include_weights, input_names)

#verbose = False
params_input['opt_time'] = 0.
params_input['opt_params'] = parameters
#rmse = get_dipole_error([])
#pc = get_parallel_context()

#rmse_vector = h.Vector(1)
#if rmse is None:
#    rmse = 0
#pc.broadcast(rmse, 0)

h.attr_praxis(1e-5, 0.5, 3)
nstep = 1
h.stop_praxis(nstep)

opt_params = h.Vector()
for param_name in params_input['opt_params'].keys():
    initial_value = params_input['opt_params'][param_name]['initial']
    opt_params.append(initial_value)

#while True:
#    get_dipole_error([])

pc = h.ParallelContext()
h.fit_praxis(get_dipole_error, opt_params)
shutdown()

