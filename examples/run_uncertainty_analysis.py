"""
=========================================
Uncertainty analysis on dipole simulation
=========================================

This example performs uncertainty analysis on a simulated dipole
using the UncertainPy package

Run with:
mpiexec --oversubscribe -np 1 python examples/run_uncertainty_analysis.py
"""

# Authors: Blake Caldwell <blake_caldwell@brown.edu>
#          Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>


import uncertainpy as un
import chaospy as cp                       # To create distributions
from numpy import loadtxt, linspace
from json import dump

from os import environ
import os.path as op

###############################################################################
# Let us import mne_neuron

import mne_neuron
from mne_neuron import Params

mne_neuron_root = op.join(op.dirname(mne_neuron.__file__), '..')

###############################################################################
# Then we read the parameters file
params_fname = op.join(mne_neuron_root, 'param', 'ERPYesSupra-3trial.json')


###############################################################################
# Read the dipole data file to compare against

extdata = loadtxt('S1_SupraT.txt')

###############################################################################
# Prepare class to get monotonically increasing unique ID


from threading import Lock, Thread

# counter from https://gist.github.com/benhoyt/8c8a8d62debe8e5aa5340373f9c509c7
class AtomicCounter:
    """An atomic, thread-safe incrementing counter."""
    def __init__(self, initial=0):
        """Initialize a new atomic counter to given initial value (default 0)."""
        self.value = initial
        self._lock = Lock()

    def increment(self, num=1):
        """Atomically increment the counter by num (default 1) and return the
        new value.
        """
        with self._lock:
            self.value += num
            return self.value

counter = AtomicCounter()

###############################################################################
#
def simulate_dipole(**kwargs):
    from numpy import loadtxt, mean
    import os.path as op

    from mpi4py import MPI

    # Start clock
    start = MPI.Wtime()
    ###############################################################################
    # Let us import mne_neuron

    from mne_neuron import simulate_dipole, average_dipoles, Params, Network
    from mne_neuron import get_rank, shutdown

    params = Params(params_fname)
    for key, value in kwargs.items():
        params[key] = value

    extdata = loadtxt('yes_trial_S1_ERP_all_avg.txt')

    ###############################################################################
    # Build our Network and set up parallel simulation

    net = Network(params)

    ###############################################################################
    # Get number of trials

    try:
        ntrials = net.params['N_trials']
    except KeyError:
        ntrials = 1

    ###############################################################################
    # Now let's simulate the dipole

    print("Running %d trials" % ntrials)

    dpls = []
    errs = []
    for trial in range(ntrials):
        dpl, err = simulate_dipole(net, trial=trial,
                                inc_evinput=net.params['inc_evinput'],
                                verbose=False, extdata=extdata)
        dpls.append(dpl)
        errs.append(err)

    avg_rmse = mean(errs)
    print("Avg. RMSE:", avg_rmse)
    params['avg_RMSE'] = avg_rmse

    avg_dpl = average_dipoles(dpls)
 
    times = linspace(0, params['tstop'], len(avg_dpl.dpl['agg']))
    info = {"rmse_output" : err}

    finish = MPI.Wtime() - start
    print('Simulation in %.2f secs' % finish)

    # write params to file with RMSE
    params.write(unique=True)

    return times, avg_dpl.dpl['agg'], info


def simulate_dipole_mpi(**kwargs):
    from time import sleep
    from mpi4py import MPI
    from numpy import loadtxt, mean

    # Start clock
    start = MPI.Wtime()

    params = Params(params_fname)
    for key, value in kwargs.items():
        params[key] = value

    extdata = loadtxt('yes_trial_S1_ERP_all_avg.txt')

    # number of processes to run nrniv with
    if 'SLURM_CPUS_ON_NODE' in environ:
#        n_procs = int(environ['SLURM_CPUS_ON_NODE'])
        n_procs = 1
    else:
        n_procs = 1

    # limit MPI to this host only
    mpiinfo = MPI.Info().Create()
#    if 'SLURMD_NODENAME' in environ:
#        mpiinfo.Set('host', environ['SLURMD_NODENAME'])
#    mpiinfo.Set('npernode', str(n_procs))
#    mpiinfo.Set('rank_by', 'slot')
#    mpiinfo.Set('ompi_param', 'rmaps_base_maps_inherit=0')
#    mpiinfo.Set('ompi_param', 'rmaps_base_schedule_local=0')

    # spawn NEURON sim
    while True:
        try:
            comm = MPI.COMM_SELF.Spawn('nrniv',
                    args=['nrniv', '-python', '-mpi', '-nobanner', 'python',
                          'examples/calculate_dipole_err.py'],
                    info = mpiinfo, maxprocs=n_procs)
            break
        except MPI.Exception:
            # try again to see if a slot has opened up
            sleep(1)
            continue

    # send params and extdata to spawned nrniv procs
    simdata = (params, extdata)
    comm.bcast(simdata, root=MPI.ROOT)

    # wait to recevie results from child rank 0
    avg_dpl, avg_rmse = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
    comm.Barrier()
    comm.Disconnect()

    params['avg_RMSE'] = avg_rmse
    # write params to file with RMSE
    params.write(unique=True)

    info = {"rmse_output" : avg_rmse} 
    times = linspace(0, params['tstop'], len(avg_dpl.dpl['agg']))

    finish = MPI.Wtime() - start
    print('Simulation in %.2f secs' % finish)

    return times, avg_dpl.dpl['agg'], info


# define the model to analyze
model = un.Model(run=simulate_dipole_mpi, labels=["time (ms)", "dipole (nAm)"])

def rmse_output(time, dipole_output, info):
    return None, info["rmse_output"]


# define some parameter distributions to sample from
t_evprox_1_dist = cp.Uniform(10,25)
t_evprox_2_dist = cp.Uniform(90,96)
t_evdist_1_dist = cp.Uniform(58,65)
sigma_t_evprox_1_dist = cp.Uniform(1,6)
gbar_evprox_1_L2Pyr_ampa_dist = cp.Uniform(0,1)
gbar_evprox_1_L2Pyr_nmda_dist = cp.Uniform(0,1)
gbar_evprox_1_L5Pyr_ampa_dist = cp.Uniform(0,1)
gbar_evprox_1_L5Pyr_nmda_dist = cp.Uniform(0,1)
gbar_evprox_1_L2Basket_nmda_dist = cp.Uniform(0,1)
gbar_evprox_1_L2Basket_ampa_dist = cp.Uniform(0,1)
gbar_evprox_1_L5Basket_nmda_dist = cp.Uniform(0,1)
gbar_evprox_1_L5Basket_ampa_dist = cp.Uniform(0,1)
sigma_t_evdist_1_dist = cp.Uniform(3.5,5)
gbar_evdist_1_L2Pyr_ampa_dist = cp.Uniform(0.155,2)
gbar_evdist_1_L2Pyr_nmda_dist = cp.Uniform(0.099,0.5)
gbar_evdist_1_L2Basket_ampa_dist = cp.Uniform(0.031,0.090)
gbar_evdist_1_L2Basket_nmda_dist = cp.Uniform(0.001,0.04)
gbar_evdist_1_L5Pyr_ampa_dist = cp.Uniform(0.5,1.504)
gbar_evdist_1_L5Pyr_nmda_dist = cp.Uniform(0.279,0.8)
sigma_t_evprox_2_dist = cp.Uniform(9,13)
gbar_evprox_2_L2Pyr_ampa_dist = cp.Uniform(28.23,68)
gbar_evprox_2_L2Basket_ampa_dist = cp.Uniform(0.0,0.0001)
gbar_evprox_2_L5Pyr_ampa_dist = cp.Uniform(29.87,70)
gbar_evprox_2_L5Basket_ampa_dist = cp.Uniform(0.0079,0.0279)

parameters = {
    "sigma_t_evprox_1": sigma_t_evprox_1_dist,
    "t_evprox_1": t_evprox_1_dist,
    "gbar_evprox_1_L2Pyr_ampa": gbar_evprox_1_L2Pyr_ampa_dist,
    "gbar_evprox_1_L2Pyr_nmda": gbar_evprox_1_L2Pyr_nmda_dist,
    "gbar_evprox_1_L5Pyr_ampa": gbar_evprox_1_L5Pyr_ampa_dist,
    "gbar_evprox_1_L5Pyr_nmda": gbar_evprox_1_L5Pyr_nmda_dist,
#    "gbar_evprox_1_L2Basket_ampa": gbar_evprox_1_L2Basket_ampa_dist,
#    "gbar_evprox_1_L2Basket_nmda": gbar_evprox_1_L2Basket_nmda_dist,
#    "gbar_evprox_1_L5Basket_ampa": gbar_evprox_1_L5Basket_ampa_dist,
#    "gbar_evprox_1_L5Basket_nmda": gbar_evprox_1_L5Basket_nmda_dist,
#    "sigma_t_evdist_1": sigma_t_evdist_1_dist,
#    "t_evdist_1": t_evdist_1_dist,
#    "gbar_evdist_1_L2Pyr_ampa": gbar_evdist_1_L2Pyr_ampa_dist,
#    "gbar_evdist_1_L2Pyr_nmda": gbar_evdist_1_L2Pyr_nmda_dist,
#    "gbar_evdist_1_L2Basket_ampa": gbar_evdist_1_L2Basket_ampa_dist,
#    "gbar_evdist_1_L2Basket_nmda": gbar_evdist_1_L2Basket_nmda_dist,
#    "gbar_evdist_1_L5Pyr_ampa": gbar_evdist_1_L5Pyr_ampa_dist,
#    "gbar_evdist_1_L5Pyr_nmda": gbar_evdist_1_L5Pyr_nmda_dist,
#    "sigma_t_evprox_2": sigma_t_evprox_2_dist,
#    "t_evprox_2": t_evprox_2_dist,
#    "gbar_evprox_2_L2Pyr_ampa": gbar_evprox_2_L2Pyr_ampa_dist,
#    "gbar_evprox_2_L2Basket_ampa": gbar_evprox_2_L2Basket_ampa_dist,
#    "gbar_evprox_2_L5Pyr_ampa": gbar_evprox_2_L5Pyr_ampa_dist,
#    "gbar_evprox_2_L5Basket_ampa": gbar_evprox_2_L5Basket_ampa_dist,
}

feature_list = [rmse_output]

if 'SLURM_NNODES' in environ:
    n_nodes = max(1,int(environ['SLURM_NNODES']) *
                    int(environ['SLURM_CPUS_ON_NODE']) - 1)
else:
    n_nodes = 1

# Run concurrent simulations
UQ = un.UncertaintyQuantification(
    model=model,
    parameters=parameters,
    CPUs=n_nodes,
    features=feature_list
)

def run_uncertainpy():
    # Run uncertainty quantification using polynomial chaos expansion
    # - specify a seed for repeatability
    UQ.quantify(method="pc", seed=10, plot="all")


if __name__ == '__main__':
#    import multiprocessing as mp
#    mp.set_start_method('spawn')
    run_uncertainpy()
