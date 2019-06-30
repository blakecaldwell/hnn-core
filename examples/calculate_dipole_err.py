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

try:
    comm = MPI.Comm.Get_parent()
    rank = comm.Get_rank()

    # receive extdata and params
    (extdata, base_params_input) = comm.bcast(rank, root=0)
    base_params = Params().from_obj(base_params_input)

    # if run by MPI, suppress output and wait for more input
    verbose = False
    loop = True

except MPI.Exception:
    params_fname = op.join(mne_neuron_root, 'param', 'default.json')
    base_params = Params(params_fname)

    ###############################################################################
    # Read the dipole data file to compare against

    extdata = loadtxt('yes_trial_S1_ERP_all_avg.txt')

    verbose = True
    loop = False

avg_sim_times = []
while True:
    # to make sure we don't have stale params, use the line below
    #params = base_params.copy()
    # note: this not possible with a sweep over the same params
    params = base_params

    if loop:
        # receive params
        new_params = comm.bcast(rank, root=0)

        # exit if we received empty params
        if new_params is None:
            break

        # set new params
        for key, value in new_params.items():
            params[key] = value

    ###############################################################################
    # Build our Network and set up parallel simulation

    net = Network(params)

    if not 'tstart' in params:
        params['tstart'] = 0

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

    # Start clock
    start = MPI.Wtime()

    dpls = [None]*ntrials
    errs = np.zeros(ntrials)
    for trial in range(ntrials):
        dpls[trial] = simulate_dipole(net, trial=trial,
                                      verbose=verbose)

    if get_rank() == 0:
        # calculate RMSE
        avg_dpl = average_dipoles(dpls)

        # Round dipole to 2 decimal points for sensitivity analysis
        avg_dpl.dpl['agg'] = np.around(avg_dpl.dpl['agg'], decimals=2)

        exp_dpl = Dipole(extdata[:,0], np.c_[extdata[:,1]], data_cols=1)
        avg_rmse = avg_dpl.rmse(exp_dpl, params['tstart'], params['tstop'])
        params['avg_RMSE'] =  avg_rmse
        if 'opt_start' in params and 'opt_end' in params:
            chunk_rmse = avg_dpl.rmse(exp_dpl, params['opt_start'], params['opt_end'])
            if verbose:
                print("[%3.2f - %3.2f] RMSE is %.8f" % (params['opt_start'], params['opt_end'], chunk_rmse))
            params['avg_RMSE'] =  chunk_rmse

        if loop:
            # send results back to parent
            if 'task_index' in params:
                task_index = params['task_index']
            else:
                task_index = 0
            data = np.array([np.array(avg_dpl.dpl['agg']).T,
                            [params['avg_RMSE'], params['tstop'], task_index]])
            comm.send(data, dest=0)

            # write params to file with RMSE
            params.write(unique=False)

        if verbose:
            print("Avg. RMSE:", avg_rmse)

        finish = MPI.Wtime() - start
        avg_sim_times.append(finish)
        print('took %.2fs for simulation (avg=%.2fs)' % (finish, mean(avg_sim_times)))

    # reset the network
    net.gid_clear()
    del net

    if not loop:
        break

#comm.Barrier()
shutdown()

