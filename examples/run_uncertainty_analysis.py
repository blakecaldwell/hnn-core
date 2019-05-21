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

import os.path as op

###############################################################################
# Let us import mne_neuron

import mne_neuron
from mne_neuron import Params

mne_neuron_root = op.join(op.dirname(mne_neuron.__file__), '..')

###############################################################################
# Then we read the parameters file
params_fname = op.join(mne_neuron_root, 'param', 'default.json')


###############################################################################
# Read the dipole data file to compare against

extdata = loadtxt('yes_trial_S1_ERP_all_avg.txt')

###############################################################################
# Now let's run the analysis

from mpi4py import MPI

def simulate_dipole(**kwargs):
    # Start clock
    start = MPI.Wtime()

    params = Params(params_fname)
    for key, value in kwargs.items():
        params[key] = value

    # how many trials to average
    n_trials = 1

    # number of processes to run nrniv with
    n_procs = 1

    # asynchronously call examples/calculate_dipole_err.py usinf nrniv
    # to allow NEURON parallelization (inner)
    comm = MPI.COMM_SELF.Spawn('nrniv',
                       args=['nrniv', '-python', '-mpi', '-nobanner', 'python',
                             'examples/calculate_dipole_err.py', str(n_trials)],
                       maxprocs=n_procs)

    # send params to spawned nrniv procs
    comm.bcast(params, root=MPI.ROOT)

    comm.Barrier()

    # send extdata to spawned nrniv procs
    comm.bcast(extdata, root=MPI.ROOT)

    # Merge the communicators before receiving
    common_comm=comm.Merge(False)

    # recevie results
    dpl, err = common_comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)

    # this seems to break the analysis
    #comm.Disconnect()
    finish = MPI.Wtime() - start
    print('\nProcessed in %.2f secs' % finish)

    times = linspace(0, params['tstop'], len(dpl.dpl['agg']))
    info = {"rmse_output" : err}

    return times, dpl.dpl['agg'], info

model = un.Model(run=simulate_dipole, labels=["time (ms)", "dipole (nAm)"])

def rmse_output(time, dipole_output, info):
    return None, info["rmse_output"]

# define some parameter distributions to sample from
t_evprox_1_dist = cp.Uniform(10,30)
t_evprox_2_dist = cp.Uniform(130,140)
t_evdist_1_dist = cp.Uniform(60,70)
parameters = {"t_evprox_1": t_evprox_1_dist, "t_evprox_2": t_evprox_2_dist, "t_evdist_1": t_evdist_1_dist}
feature_list = [rmse_output]

# Run 8 concurrent simulations
UQ = un.UncertaintyQuantification(
    model=model,
    parameters=parameters,
    CPUs=8,
    features=feature_list
)

# Run uncertainty quantification using polynomial chaos expansion
# - specify a seed for repeatability
data = UQ.quantify(method="pc", seed=10, plot="all")
