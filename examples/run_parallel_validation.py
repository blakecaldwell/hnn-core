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
tags = enum('READY', 'DONE', 'EXIT', 'START')

if rank != 0:
    from statistics import mean
    import numpy as np

    # start the workers

    print("Worker started with rank %d on %s." % (rank, name))

    # receive experimental data
    (exp_data, params_input) = comm.bcast(comm.Get_rank(), root=0)

    # number of processes to run nrniv with
    if 'SLURM_CPUS_ON_NODE' in environ:
        n_procs = int(environ['SLURM_CPUS_ON_NODE']) - 2
    else:
        n_procs = 1

    # limit MPI to this host only
    mpiinfo = MPI.Info().Create()
    mpiinfo.Set('host', name.split('.')[0])
    mpiinfo.Set('ompi_param', 'rmaps_base_inherit=0')
    mpiinfo.Set('ompi_param', 'rmaps_base_mapping_policy=core')
    mpiinfo.Set('ompi_param', 'rmaps_base_oversubscribe=1')
    # spawn NEURON sim
    subcomm = MPI.COMM_SELF.Spawn('nrniv',
            args=['nrniv', '-python', '-mpi', '-nobanner', 'python',
                  'examples/validate_dipole.py'],
            info = mpiinfo, maxprocs=n_procs)

    # send params and exp_data to spawned nrniv procs
    simdata = (exp_data, params_input)
    subcomm.bcast(simdata, root=MPI.ROOT)

    #subcomm.Barrier()
    print("Worker %d waiting on master to signal start" % rank)
    # tell rank 0 we are ready
    comm.send(None, dest=0, tag=tags.READY)

    # Receive updated params (blocking)
    new_params = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

    # relay params to spawned nrniv procs
    subcomm.bcast(new_params, root=MPI.ROOT)

    # wait for processes to finish
    subcomm.Barrier()

    # tell rank 0 we are closing
    comm.send(None, dest=0, tag=tags.EXIT)

    # send empty new_params to stop nrniv procs

if rank == 0:
    print("Master starting on %s" % name)

    import uncertainpy as un
    import chaospy as cp                       # To create distributions
    from numpy import loadtxt, linspace
    from json import load
    print("Master started")

    if environ['DATA_DIR'] and op.exists(environ['DATA_DIR']):
        mne_neuron_root = environ['DATA_DIR']

    params_fname = op.join(mne_neuron_root, 'param', 'validate_hnn.json')

    ###############################################################################
    # Read the dipole data file to compare against

    print("Master loading file data")
    extdata = []
    for i in range (5):
        extdata.append(loadtxt('default_hnn_trial%d.txt' % i))

    with open(params_fname) as json_data:
        params_input = load(json_data)

    params_input['max_loops'] = 100
    simdata = (extdata, params_input)
    print("Master has finished loading file data. Sending to the workers.")

    # broadcast simdata to all of the workers
    comm.bcast(simdata, root=0)
    
    if 'SLURM_NNODES' in environ:
        n_nodes = max(1, size - 1)
    else:
        n_nodes = 1
    
    print("Master starting validation on %d nodes" % n_nodes)

    num_workers = n_nodes
    closed_workers = 0

    while closed_workers < num_workers:
        comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()

        if tag == tags.READY:
            # Worker is ready, so send it a task
            comm.send(params_input, dest=source, tag=tags.START)
        elif tag == tags.EXIT:
            print("Worker %d exited (%d running)" % (source, closed_workers))
            closed_workers += 1

    print("Master finished")

