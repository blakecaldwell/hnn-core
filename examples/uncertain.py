"""
=========================================
Uncertainty analysis on dipole simulation
=========================================

This example performs uncertainty analysis on a simulated dipole
using the UncertainPy package

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
from mne_neuron import calculate_dipole_err, Params, Network

mne_neuron_root = op.join(op.dirname(mne_neuron.__file__), '..')

###############################################################################
# Then we read the parameters file
params_fname = op.join(mne_neuron_root, 'param', 'default.json')

def load_data_file (fn):
    # load a dipole data file
    try:
      datafile = loadtxt(fn)
      print('Loaded data in ', fn)
      return datafile
    except:
      print('Could not load data in ', fn)
      return None

###############################################################################
# Now let's simulate the dipole

def simulate_dipole(**kwargs):
    extdata = load_data_file('yes_trial_S1_ERP_all_avg.txt')

    params = Params(params_fname)
    for key, value in kwargs.items():
        params[key] = value

    net = Network(params)

    dpl, err = calculate_dipole_err(net, extdata)

    times = linspace(0, net.params['tstop'], len(dpl.dpl['agg']))
    info = {"rmse_output" : err}

    return times, dpl.dpl['agg'], info

model = un.Model(run=simulate_dipole, labels=["time (ms)", "dipole (nAm)"])

def rmse_output(time, dipole_output, info):
    return None, info["rmse_output"]

# define some parameter distributions to sample from
t_evprox_1_dist = cp.Uniform(10,30)
t_evprox_2_dist = cp.Uniform(130,140)
t_evdist_1_dist = cp.Uniform(60,70)
#parameters = {"t_evprox_1": t_evprox_1_dist}
parameters = {"t_evprox_1": t_evprox_1_dist, "t_evprox_2": t_evprox_2_dist, "t_evdist_1": t_evdist_1_dist}
feature_list = [rmse_output]

# Run 8 concurrent simulations using multiprocessing
UQ = un.UncertaintyQuantification(
    model=model,
    parameters=parameters,
    CPUs=8,
    features=feature_list
)

# Run uncertainty quantification using polynomial chaos expansion
# - specify a seed for repeatability
data = UQ.quantify(method="pc", seed=10, plot="all")
