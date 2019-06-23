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
from mne_neuron import get_rank, shutdown

mne_neuron_root = op.join(op.dirname(mne_neuron.__file__), '..')

###############################################################################
# Try to read the parameters and exp data via MPI

from mpi4py import MPI
import numpy as np

# default is to loop once unless set in params
max_loops = 1

try:
    comm = MPI.Comm.Get_parent()
    rank = comm.Get_rank()

    # receive extdata and params
    (extdata, base_params_input) = comm.bcast(rank, root=0)
    params = Params().from_obj(base_params_input)

    # if run by MPI, suppress output and wait for more input
    verbose = False

except MPI.Exception:
    params_fname = op.join(mne_neuron_root, 'param', 'validate_hnn.json')
    params = Params(params_fname)

    ###############################################################################
    # Read the dipole data file to compare against

    extdata1 = loadtxt('default_hnn_trial1.txt')
    extdata2 = loadtxt('default_hnn_trial2.txt')
    extdata3 = loadtxt('default_hnn_trial3.txt')
    extdata4 = loadtxt('default_hnn_trial4.txt')
    extdata5 = loadtxt('default_hnn_trial5.txt')
    extdata = [extdata1, extdata2, extdata3, extdata4, extdata5]

    verbose = True

if 'max_loops' in params:
    max_loops = params['max_loops']

###############################################################################
# Build our Network and set up parallel simulation

net = Network(params)

loop = 0
while loop < max_loops:

    ###############################################################################
    # Get number of trials

    try:
        ntrials = net.params['N_trials']
    except KeyError:
        ntrials = 1

    if verbose and get_rank() == 0:
        print("Running %d trials" % ntrials)

    ###############################################################################
    # Now let's simulate the dipole

    for trial in range(ntrials):
        dpl = simulate_dipole(net, trial=trial,
                                      verbose=verbose)

        if get_rank() == 0:
            # calculate RMSE
            exp_dpl = Dipole(extdata[trial][:,0], np.c_[extdata[trial][:,1]], data_cols=1)
            rmse = dpl.rmse(exp_dpl, 0, params['tstop'])

            if not rmse == 0.0:
                print('Error: RMSE is %d' % rmse)
                loop = max_loops

            if not exp_dpl.dpl['agg'] == dpl.dpl['agg']:
                print('Error: dipoles are not equal')
                loop = max_loops

    loop = loop + 1

#comm.Barrier()
shutdown()

