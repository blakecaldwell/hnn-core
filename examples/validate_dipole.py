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

# simulate truncating digits
# modified from https://realpython.com/python-rounding/#truncation
# to correct decimal places index 
def truncate(n, decimals=0):
    multiplier = 10 ** (decimals)
    return int(n * multiplier) / multiplier

try:
    comm = MPI.Comm.Get_parent()
    rank = comm.Get_rank()

    # receive extdata and params
    (extdata, base_params_input) = comm.bcast(rank, root=0)
    if 'max_loops' in base_params_input:
        max_loops = int(base_params_input['max_loops'])

    params = Params().from_obj(base_params_input)

    # if run by MPI, suppress output and wait for more input
    verbose = False

except MPI.Exception:
    params_fname = op.join(mne_neuron_root, 'param', 'validate_hnn.json')
    params = Params(params_fname)

    ###############################################################################
    # Read the dipole data file to compare against

    extdata = []
    for i in range (5):
        extdata.append(loadtxt('default_hnn_trial%d.txt' % i))

#    verbose = True
    verbose = False


###############################################################################
# Build our Network and set up parallel simulation

avg_sim_times = []
loop = 0
max_threshold = 0.0
done = False
while loop < max_loops:

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

    decimals = 5


    # Start clock
    start = MPI.Wtime()

    for trial in range(ntrials):
        dpl = simulate_dipole(net, trial=trial,
                                      verbose=verbose)

        if get_rank() == 0:
            data = np.c_[extdata[trial][:,1],
                         extdata[trial][:,2],
                         extdata[trial][:,3]]
            exp_dpl = Dipole(extdata[trial][:,0], data)


            for cell_dipole in ['L2', 'L5']:
                rounded_dpl = np.around(dpl.dpl[cell_dipole], decimals=8)
                rounded_exp_dpl = np.around(exp_dpl.dpl[cell_dipole], decimals=8)
                diffs = np.logical_not(rounded_dpl == rounded_exp_dpl)
                indices = np.where(diffs == True)[0]
    
                # check the differences (if any) for truncation instead
                # of rounding
                true_diffs = []
                for index in indices:
#                    if not (truncate(dpl.dpl[cell_dipole][index], decimals) ==
#                            truncate(exp_dpl.dpl[cell_dipole][index], decimals)):
                    difference = abs(dpl.dpl[cell_dipole][index] - exp_dpl.dpl[cell_dipole][index])
                    if difference > max_threshold:
#                         print('new threshold = %.12f' % max_threshold)
                         max_threshold = difference
                         true_diffs.append(index)

                    if len(true_diffs) > 0:
                        print('ERROR: %s dipoles are not equal: %d Values differ' %
                              (cell_dipole, len(true_diffs)) +
                              ' even after truncation to %d decimal places' %
                              decimals)
                        print('Starting at time=%03.3fs: sim=%5.15f exper=%5.15f' %
                              (dpl.t[index],
#                               truncate(dpl.dpl[cell_dipole][index], decimals),
#                               truncate(exp_dpl.dpl[cell_dipole][index], decimals)))
                               dpl.dpl[cell_dipole][index],
                               exp_dpl.dpl[cell_dipole][index]))
                        print('Different times: %s' %
                              true_diffs)
                        # calculate RMSE
                        rmse = dpl.rmse(exp_dpl, 0.0, float(params['tstop']))
                        print("RMSE is %.8f \n" % rmse)
                        done = True
                        break
 
                if done:
                    break
#                elif len(indices) > 0:
#                    print('All %d %s differences were resolved by truncating to' %
#                          ((len(indices) -len(true_diffs)), cell_dipole) +
#                          ' %d ' % decimals + 'digits')

    if done:
        loop = max_loops
        break
    else:
        loop = loop + 1

    # reset the network
    net.gid_clear()
    del net

    finish = MPI.Wtime() - start
    avg_sim_times.append(finish)
    if get_rank() == 0:
        print('took %.2fs for simulation (avg=%.2fs)' % (finish, mean(avg_sim_times)))


print('ending validation')
shutdown()

