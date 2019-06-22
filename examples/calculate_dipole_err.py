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
from mne_neuron import simulate_dipole, average_dipoles, Params, Network
from mne_neuron import get_rank, shutdown

mne_neuron_root = op.join(op.dirname(mne_neuron.__file__), '..')

###############################################################################
# Try to read the parameters and exp data via MPI

from mpi4py import MPI
import numpy as np

comm = MPI.Comm.Get_parent()
rank = comm.Get_rank()

# receive extdata and params
(extdata, base_params_input) = comm.bcast(rank, root=0)
base_params = Params().from_obj(base_params_input)

# if run by MPI, suppress output
verbose = False

sim_params = {}
while True:
    # receive params
    new_params = comm.bcast(rank, root=0)

    # exit if we received empty params
    if new_params is None:
        break

    # to make sure we don't have stale params, use the line below
    #params = base_params.copy()
    # note: this not possible with a sweep over the same params
    params = base_params

    # set new params
    for key, value in new_params.items():
        params[key] = value

    ###############################################################################
    # Build our Network and set up parallel simulation

    net = Network(params)

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

    dpls = [None]*ntrials
    errs = np.zeros(ntrials)
    for trial in range(ntrials):
        dpls[trial], errs[trial] = simulate_dipole(net, trial=trial,
                                                   verbose=False, extdata=extdata)

    if get_rank() == 0:
        params['avg_RMSE'] =  mean(errs)
        # send results back to parent
        data = np.array([np.array(average_dipoles(dpls).dpl['agg']).T,
                        [params['avg_RMSE'], params['tstop']]])
        comm.send(data, dest=0)

        # write params to file with RMSE
        params.write(unique=False)

        if verbose:
           print("Avg. RMSE:", params['avg_RMSE'])

    # reset the network
    net.gid_clear()
    del net

#comm.Barrier()
shutdown()

