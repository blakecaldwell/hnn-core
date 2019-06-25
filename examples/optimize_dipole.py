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

def split_by_evinput(params):
    import re

    chunks = {}
    for k, v in params.items():
        input_mu = re.match('^t_ev(prox|dist)_([0-9]+)', k)
        if input_mu:
            id_str = input_mu.group(1) + '_' + input_mu.group(2)
            if not id_str in chunks:
                chunks[id_str] = {}
            chunks[id_str]['mean'] = float(v)
            continue
        input_sigma = re.match('^sigma_t_ev(prox|dist)_([0-9]+)', k)
        if input_sigma:
            id_str = input_sigma.group(1) + '_' + input_sigma.group(2)
            if not id_str in chunks:
                chunks[id_str] = {}
            chunks[id_str]['sigma'] = float(v)

    # bound by 3 sigma
    for c in chunks.keys():
        chunks[c]['start'] = max(0, chunks[c]['mean'] - 3 * chunks[c]['sigma'])
        chunks[c]['end'] = min(float(params['tstop']), chunks[c]['mean'] + 3 * chunks[c]['sigma'])

    sorted_chunks = sorted(chunks.items(), key=lambda x: x[1]['start'])

    # consolidate
    consolidated_chunks = {}
    for c1 in chunks.keys():
        if c1 in consolidated_chunks:
            continue
        for c2 in chunks.keys():
            if chunks[c1]['end'] == chunks[c2]['end']:
                # either the same chunk or alrady consolidated
                continue
            if chunks[c1]['end'] > chunks[c2]['start']:
                chunks[c1]['end'] = chunks[c2]['end']
        consolidated_chunks[c1] = chunks[c1]

    return consolidated_chunks

def get_dipole_error(params, extdata, verbose=True):

    avg_sim_times = []
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
    # Split by input times
    chunks = split_by_evinput(params)

    ###############################################################################
    # Now let's simulate the dipole

    # Start clock
    start = MPI.Wtime()


    dpls = [None]*ntrials
    for trial in range(ntrials):
        dpls[trial] = simulate_dipole(net, trial=trial,
                                      verbose=verbose)

    if get_rank() == 0:
        avg_dpl = average_dipoles(dpls)
        data = np.c_[extdata[:,1],
                     extdata[:,1],
                     extdata[:,1]]
        exp_dpl = Dipole(extdata[:,0], data)

        # get errors by chunks
        for k, v in chunks.items():
            rmse = avg_dpl.rmse(exp_dpl, v['start'], v['end'])
            if verbose:
                print("[%3.2f - %3.2f] RMSE is %.8f \n" % (v['start'], v['end'], rmse))

        rmse = avg_dpl.rmse(exp_dpl, 0.0, float(params['tstop']))
        if verbose:
            print("Total RMSE is %.8f \n" % rmse)

    # reset the network
    net.gid_clear()
    del net

    finish = MPI.Wtime() - start
    avg_sim_times.append(finish)
    if get_rank() == 0:
        print('took %.2fs for simulation (avg=%.2fs)' % (finish, mean(avg_sim_times)))

try:
    comm = MPI.Comm.Get_parent()
    rank = comm.Get_rank()

    # receive extdata and params
    (extdata, base_params_input) = comm.bcast(rank, root=0)

    params = Params().from_obj(base_params_input)

    # if run by MPI, suppress output and wait for more input
    verbose = False

except MPI.Exception:
    params_fname = op.join(mne_neuron_root, 'param', 'ERPYesSupra-3trial.json')
    params = Params(params_fname)

    ###############################################################################
    # Read the dipole data file to compare against

    extdata = loadtxt('S1_SupraT.txt')

#    verbose = True
    verbose = True


done = False
while not done:

    verbose = False
    get_dipole_error(params, extdata, verbose)
    break
    if done:
        break


shutdown()

