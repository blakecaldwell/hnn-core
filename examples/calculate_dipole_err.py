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
from sys import argv
import os.path as op

###############################################################################
# Let us import mne_neuron

import mne_neuron
from mne_neuron import simulate_dipole, average_dipoles, Params, Network, get_rank, shutdown

mne_neuron_root = op.join(op.dirname(mne_neuron.__file__), '..')

###############################################################################
# Parse command-line arguments

ntrials = 1
if len(argv) > 7:
    try:
        ntrials = int(argv[7])
    except TypeError:
        pass

###############################################################################
# Then we read the parameters via MPI

from mpi4py import MPI

try:
    comm = MPI.Comm.Get_parent()

    # receive params
    params = comm.bcast(comm.Get_rank(), root=0)

    # wait for master to send data to compare against
    comm.Barrier()

    # receive extdata
    extdata = comm.bcast(comm.Get_rank(), root=0)

    # merge communicators to prepare sending results
    common_comm=comm.Merge(True)

except MPI.Exception:
    # Have to read the parameters from a file
    params_fname = op.join(mne_neuron_root, 'param', 'default.json')
    print("Reading parameters from file:", params_fname)
    params = Params(params_fname)

    extdata = loadtxt('yes_trial_S1_ERP_all_avg.txt')

net = Network(params)

###############################################################################
# Read the dipole data file to compare against


###############################################################################
# Now let's simulate the dipole

if get_rank() == 0:
    print("Running %d trials" % ntrials)

dpls = []
errs = []
for trial in range(ntrials):
    dpl, err = simulate_dipole(net, trial=trial, inc_evinput=net.params['inc_evinput'],
                               print_progress=False, extdata=extdata)
    dpls.append(dpl)
    errs.append(err)

if get_rank() == 0:
    avg_rmse = mean(errs)
    print("Avg. RMSE:", avg_rmse)

    try:
        # send results back to parent
        common_comm.send((average_dipoles(dpls), avg_rmse), dest=0)
        #comm.Disconnect()
    except MPI.Exception:
        raise

shutdown()
