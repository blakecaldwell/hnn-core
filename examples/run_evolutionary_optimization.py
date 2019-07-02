"""
=========================================
Evolutionary optimization on dipole simulation
=========================================

This example evolutionary optimization on a simulated dipole
using the inspyred package

MPI master/worker task scheme used from:
https://github.com/jbornschein/mpi4py-examples/blob/master/09-task-pull.py

Run with:

"""

# Authors: Blake Caldwell <blake_caldwell@brown.edu>
#          Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>

from mpi4py import MPI
from os import environ
import os.path as op

# Initializations and preliminaries
comm = MPI.COMM_WORLD   # get MPI communicator object
size = comm.size        # total number of processes
rank = comm.rank        # rank of this process
status = MPI.Status()   # get MPI status object
name = MPI.Get_processor_name()

def enum(*sequential, **named):
    """Handy way to fake an enumerated type in Python
    http://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python
    """
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)
# Define MPI message tags
tags = enum('READY', 'DONE', 'HOLD', 'EXIT', 'START')

if rank != 0:
    from statistics import mean
    import numpy as np

    print("Worker started with rank %d on %s." % (rank, name))

    # receive experimental data
    (exp_data, params_input) = comm.bcast(rank, root=0)

    # number of processes to run nrniv with
    if 'SLURM_CPUS_ON_NODE' in environ:
        n_procs = int(environ['SLURM_CPUS_ON_NODE']) - 2
    else:
        n_procs = size - 2

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

    avg_sim_times = []

    #subcomm.Barrier()
    print("Worker %d waiting on master to signal start" % rank)

    while True:
        # Start clock
        #start = MPI.Wtime()

        # tell rank 0 we are ready
        comm.isend(None, dest=0, tag=tags.READY)

        # Receive updated params (blocking)
        new_params = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

        tag = status.Get_tag()
        if tag == tags.EXIT:
            print('worker %d on %s has received exit signal'%(rank, name))
            break
        elif tag == tags.HOLD:
            print('worker %d on %s has received hold signal'%(rank, name))
            # wait for this round to complete
            comm.bcast(rank, root=0)
            continue

        #assert(tag == tags.START)

        #finish = MPI.Wtime() - start
        #print('worker %s waited %.2fs for param set' % (name, finish))

        # Start clock
        start = MPI.Wtime()

        # send new_params to spawned nrniv procs
        subcomm.bcast(new_params, root=MPI.ROOT)

        # wait to recevie results from child rank 0
        #temp_results = np.array([np.zeros(int(params_input['tstop'] / params_input['dt'] + 1)),
        #                         np.zeros(3)])
        temp_results = subcomm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
        #subcomm.Recv(temp_results, source=MPI.ANY_SOURCE)

        finish = MPI.Wtime() - start
        avg_sim_times.append(finish)
        print('worker %s took %.2fs for simulation (avg=%.2fs)' % (name, finish, mean(avg_sim_times)))
   
        # send results back
        comm.isend(temp_results, dest=0, tag=tags.DONE)

    # tell rank 0 we are closing
    comm.send(None, dest=0, tag=tags.EXIT)

    # send empty new_params to stop nrniv procs
    subcomm.bcast(None, root=MPI.ROOT)
    #subcomm.Barrier()

if rank == 0:
    print("Master starting on %s" % name)

    from inspyred import ec   # evolutionary algorithm
    from random import Random # pseudorandom number generation
    from time import sleep
    from numpy import loadtxt
    from json import load
    from batch import Batch

    print("Master started")

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
                    #print("Varying %s in range[%.4f-%.4f]" % (var_name, timing_min, timing_max))
                    parameters[var_name] = {'initial': mu, 'minval': timing_min, 'maxval': timing_max }
                if 'weights_only' in include_weights or 'timing_and_weights' in include_weights:
                    for weight in ['L2Pyr_ampa', 'L2Pyr_nmda',
                                'L2Basket_ampa', 'L2Basket_nmda',
                                'L5Pyr_ampa', 'L5Pyr_nmda',
                                'L5Basket_ampa', 'L5Basket_nmda']:

                        timing_weight_name = "gbar_%s_%s"%(input_name, weight)
                        if not timing_weight_name in params_input:
                            #print("could not find variable %s in params"%timing_weight_name)
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


    # Parse command-line arguments
    if environ['PARAMS_FNAME'] and op.exists(environ['PARAMS_FNAME']):
        params_fname = environ['PARAMS_FNAME']
    else:
        import mne_neuron
        mne_neuron_root = op.join(op.dirname(mne_neuron.__file__), '..')
        params_fname = op.join(mne_neuron_root, 'param', 'default.json')
        print("using default param file:", params_fname)

    if environ['EXP_FNAME'] and op.exists(environ['EXP_FNAME']):
        exp_fname = environ['EXP_FNAME']
    else:
        import mne_neuron
        mne_neuron_root = op.join(op.dirname(mne_neuron.__file__), '..')
        exp_fname = op.join(mne_neuron_root, 'yes_trial_S1_ERP_all_avg.txt')
        print("using default experimental data:", exp_fname)

    print("Master loading file data")
    # Read the dipole data and params files once
    exp_data = loadtxt(exp_fname)
    with open(params_fname) as json_data:
        params_input = load(json_data)

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
    opt_parameters =  set_parameters(include_weights, input_names)
    params_input['sim_prefix'] = "%s_evol%s_%s" % (op.basename(params_fname).split('.json')[0], input_name, include_weights)

    simdata = (exp_data, params_input)
    print("Master has finished loading file data. Sending to the workers.")

    # broadcast simdata to all of the workers
    comm.bcast(simdata, root=0)

    if 'SLURM_NNODES' in environ:
        n_nodes = max(1, size - 1)
    else:
        n_nodes = 1

    # create Batch object with paramaters to modify, and specifying files to use
    b = Batch(params=opt_parameters)

    # Set output folder, grid method (all param combinations), and run configuration
    b.saveFolder = './'+b.batchLabel
    b.method = 'evol'
    b.runCfg = {
        'script': 'examples/calculate_dipole_err.py',
        # options required only for hpc
        'mpiCommand': 'mpirun',  
        'nodes': n_nodes,
        'coresPerNode': 2,
        'folder': '$HOME/mne-neuron/data/evol'
        #'custom': 'export LD_LIBRARY_PATH="$HOME/.openmpi/lib"' # only for conda users
    }
    b.evolCfg = {
        'evolAlgorithm': 'custom',
        'pop_size': n_nodes * 40,
        'num_elites': 10, # keep this number of parents for next generation if they are fitter than children
        'mutation_rate': 0.3,
        'crossover': 0.5,
        'tournament_size': 20,
        'maximize': False, # maximize fitness function?
        'max_generations': 60,
        'time_sleep': 5, # wait this time before checking again if sim is completed (for each generation)
        'maxiter_wait': 40, # max number of times to check if sim is completed (for each generation)
        'defaultFitness': 1000 # set fitness value in case simulation time is over
    }

    print("Master starting optimization on %d nodes" % n_nodes)

    # Run batch simulations
    b.run()

    print("Master finished")

