"""
=========================================
Uncertainty analysis on dipole simulation
=========================================

This example performs uncertainty analysis on a simulated dipole
using the UncertainPy package

MPI master/worker task scheme used from:
https://github.com/jbornschein/mpi4py-examples/blob/master/09-task-pull.py

Run with:
mpiexec -np 4 python examples/run_uncertainty_analysis.py
"""

# Authors: Blake Caldwell <blake_caldwell@brown.edu>
#          Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>

from mpi4py import MPI
import dill  
MPI.pickle.__init__(dill.dumps, dill.loads)
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

if rank != 0:
    from statistics import mean
    import numpy as np

    # start the workers

    # Define MPI message tags
    tags = enum('READY', 'DONE', 'EXIT', 'START')

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
                  'examples/calculate_dipole_err.py'],
            info = mpiinfo, maxprocs=n_procs)

    # send params and exp_data to spawned nrniv procs
    simdata = (exp_data, params_input)
    subcomm.bcast(simdata, root=MPI.ROOT)

    avg_sim_times = []

    #subcomm.Barrier()
    print("Worker %d waiting on master to signal start" % rank)
    # tell rank 0 we are ready
    comm.isend(None, dest=0, tag=tags.READY)

    while True:
        # Start clock
        #start = MPI.Wtime()

        # Receive updated params (blocking)
        new_params = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

        tag = status.Get_tag()
        if tag == tags.EXIT:
            print('worker %d on %s has received exit signal'%(rank, name))
            break

        #assert(tag == tags.START)

        #finish = MPI.Wtime() - start
        #print('worker %s waited %.2fs for param set' % (name, finish))

        # Start clock
        start = MPI.Wtime()

        # send new_params to spawned nrniv procs
        subcomm.bcast(new_params, root=MPI.ROOT)

        # wait to recevie results from child rank 0
        #temp_results = np.array([np.zeros(int(params_input['tstop'] / params_input['dt'] + 1)),
        #                         np.zeros(2)])
        temp_results = subcomm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
        #subcomm.Recv(temp_results, source=MPI.ANY_SOURCE)

        finish = MPI.Wtime() - start
        avg_sim_times.append(finish)
        print('worker %s took %.2fs for simulation (avg=%.2fs)' % (name, finish, mean(avg_sim_times)))
   
        # send results back
        comm.isend(temp_results, dest=0, tag=tags.DONE)

        # tell rank 0 we are ready (again)
        comm.isend(None, dest=0, tag=tags.READY)

    # tell rank 0 we are closing
    comm.send(None, dest=0, tag=tags.EXIT)

if rank == 0:
    print("Master starting on %s" % name)

    import uncertainpy as un
    import chaospy as cp                       # To create distributions
    from numpy import loadtxt
    from json import load
    print("Master started")

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
    simdata = (exp_data, params_input)
    print("Master has finished loading file data. Sending to the workers.")

    # broadcast simdata to all of the workers
    comm.bcast(simdata, root=0)

    def optimize_erp_2():  
        pass
    # define the model to analyze
    model = un.Model(run=optimize_erp_2, labels=["time (ms)", "dipole (nAm)"])
    
    def rmse_output(time, dipole_output, info):
        return None, info["rmse_output"]
    
    # define some parameter distributions to sample from
    t_evprox_1_dist = cp.Uniform(9,12)
    t_evdist_1_dist = cp.Uniform(10,120)
    t_evprox_2_dist = cp.Uniform(60,90)
    sigma_t_evprox_1_dist = cp.Uniform(1,8)
    #sigma_t_evprox_1_dist = cp.Uniform(1,15)
    gbar_evprox_1_L2Pyr_ampa_dist = cp.Uniform(0,1)
    gbar_evprox_1_L2Pyr_nmda_dist = cp.Uniform(0,1)
    gbar_evprox_1_L5Pyr_ampa_dist = cp.Uniform(0,1)
    gbar_evprox_1_L5Pyr_nmda_dist = cp.Uniform(0,1)
    gbar_evprox_1_L2Basket_nmda_dist = cp.Uniform(0,1)
    gbar_evprox_1_L2Basket_ampa_dist = cp.Uniform(0,1)
    gbar_evprox_1_L5Basket_nmda_dist = cp.Uniform(0,1)
    gbar_evprox_1_L5Basket_ampa_dist = cp.Uniform(0,1)
    sigma_t_evdist_1_dist = cp.Uniform(1,15)
    #sigma_t_evdist_1_dist = cp.Uniform(3.5,5)
    gbar_evdist_1_L2Pyr_ampa_dist = cp.Uniform(0,5)
    gbar_evdist_1_L2Pyr_nmda_dist = cp.Uniform(0.0,2)
    gbar_evdist_1_L2Basket_ampa_dist = cp.Uniform(0,1)
    gbar_evdist_1_L2Basket_nmda_dist = cp.Uniform(0.0,0.4)
    gbar_evdist_1_L5Pyr_ampa_dist = cp.Uniform(0.,4)
    gbar_evdist_1_L5Pyr_nmda_dist = cp.Uniform(0,5)
    sigma_t_evprox_2_dist = cp.Uniform(1,15)
    #sigma_t_evprox_2_dist = cp.Uniform(9,13)
    gbar_evprox_2_L2Pyr_ampa_dist = cp.Uniform(28.23,68)
    gbar_evprox_2_L2Basket_ampa_dist = cp.Uniform(0.0,0.0001)
    gbar_evprox_2_L5Pyr_ampa_dist = cp.Uniform(29.87,70)
    gbar_evprox_2_L5Basket_ampa_dist = cp.Uniform(0.0079,0.0279)
    
    parameters = {
    #    "sigma_t_evprox_1": sigma_t_evprox_1_dist,
    #    "t_evprox_1": t_evprox_1_dist,
    #    "gbar_evprox_1_L2Pyr_ampa": gbar_evprox_1_L2Pyr_ampa_dist,
    #    "gbar_evprox_1_L2Pyr_nmda": gbar_evprox_1_L2Pyr_nmda_dist,
    #    "gbar_evprox_1_L5Pyr_ampa": gbar_evprox_1_L5Pyr_ampa_dist,
    #    "gbar_evprox_1_L5Pyr_nmda": gbar_evprox_1_L5Pyr_nmda_dist,
    #    "gbar_evprox_1_L2Basket_ampa": gbar_evprox_1_L2Basket_ampa_dist,
    #    "gbar_evprox_1_L2Basket_nmda": gbar_evprox_1_L2Basket_nmda_dist,
    #    "gbar_evprox_1_L5Basket_ampa": gbar_evprox_1_L5Basket_ampa_dist,
    #    "gbar_evprox_1_L5Basket_nmda": gbar_evprox_1_L5Basket_nmda_dist,
        "sigma_t_evdist_1": sigma_t_evdist_1_dist,
        "t_evdist_1": t_evdist_1_dist,
        "gbar_evdist_1_L2Pyr_ampa": gbar_evdist_1_L2Pyr_ampa_dist,
        "gbar_evdist_1_L2Pyr_nmda": gbar_evdist_1_L2Pyr_nmda_dist,
        "gbar_evdist_1_L2Basket_ampa": gbar_evdist_1_L2Basket_ampa_dist,
        "gbar_evdist_1_L2Basket_nmda": gbar_evdist_1_L2Basket_nmda_dist,
        "gbar_evdist_1_L5Pyr_ampa": gbar_evdist_1_L5Pyr_ampa_dist,
        "gbar_evdist_1_L5Pyr_nmda": gbar_evdist_1_L5Pyr_nmda_dist,
    #    "sigma_t_evprox_2": sigma_t_evprox_2_dist,
    #    "t_evprox_2": t_evprox_2_dist,
    #    "gbar_evprox_2_L2Pyr_ampa": gbar_evprox_2_L2Pyr_ampa_dist,
    #    "gbar_evprox_2_L2Basket_ampa": gbar_evprox_2_L2Basket_ampa_dist,
    #    "gbar_evprox_2_L5Pyr_ampa": gbar_evprox_2_L5Pyr_ampa_dist,
    #    "gbar_evprox_2_L5Basket_ampa": gbar_evprox_2_L5Basket_ampa_dist,
    }
    
    feature_list = [rmse_output]
    
    if 'SLURM_NNODES' in environ:
        n_nodes = max(1, size - 1)
    else:
        n_nodes = 1
    
    UQ = un.UncertaintyQuantification(
        model=model,
        parameters=parameters,
        CPUs=n_nodes,
        features=feature_list
    )

    # Run uncertainty quantification using polynomial chaos expansion
    # - specify a seed for repeatability
    #UQ.quantify(method="pc", seed=10, plot="all")
    #UQ.quantify(method="pc", rosenblatt=True, polynomial_order=7, seed=10, plot="all")
    print("Master starting sensitivity analysis on %d cores" % n_nodes)
    UQ.quantify(method="mc", nr_samples=10000, rosenblatt=True, seed=10, plot="all")
    print("Master finished")

