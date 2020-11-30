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
from os import environ
import os.path as op

###############################################################################
# Let us import mne_neuron

import mne_neuron
from mne_neuron import simulate_dipole, average_dipoles, Params, Network, get_rank, shutdown

mne_neuron_root = op.join(op.dirname(mne_neuron.__file__), '..')

###############################################################################
# Then we read the parameters via MPI

try:
    from mpi4py import MPI

    comm = MPI.Comm.Get_parent()

    # receive params and extdata
    (params, extdata) = comm.bcast(comm.Get_rank(), root=0)

    # guard against a missing unique ID (should be sent by parent)
    if not 'unique_id' in params:
        params['unique_id'] = 0

    # if run by MPI, suppress output
    print_progress=False

except MPI.Exception:
    comm = None
    # Have to read the parameters from a file
    params_fname = op.join(mne_neuron_root, 'param', 'default.json')
    print("Reading parameters from file:", params_fname)
    params = Params(params_fname)

    extdata = loadtxt('yes_trial_S1_ERP_all_avg.txt')

    print_progress=True

net = Network(params)

###############################################################################
# get number of trials

try:
    ntrials = net.params['N_trials']
except KeyError:
    ntrials = 1

###############################################################################
# Now let's simulate the dipole

if get_rank() == 0 and print_progress:
    print("Running %d trials" % ntrials)

dpls = []
errs = []
for trial in range(ntrials):
    dpl, err = simulate_dipole(net, trial=trial, inc_evinput=net.params['inc_evinput'],
                               print_progress=print_progress, extdata=extdata)
    dpls.append(dpl)
    errs.append(err)

if get_rank() == 0:
    avg_rmse = mean(errs)
    if print_progress:
        print("Avg. RMSE:", avg_rmse)

    if not comm == None:
        # send results back to parent
        comm.send((average_dipoles(dpls), avg_rmse), dest=0)

if not comm == None:
    comm.Barrier()
    comm.Disconnect()

shutdown()
