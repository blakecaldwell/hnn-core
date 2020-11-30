"""Class to handle the dipoles."""

# Authors: Mainak Jas <mainak.jas@telecom-paristech.fr>
#          Sam Neymotin <samnemo@gmail.com>

import numpy as np
from numpy import convolve, hamming


def _hammfilt(x, winsz):
    """Convolve with a hamming window."""
    win = hamming(winsz)
    win /= sum(win)
    return convolve(x, win, 'same')


def initialize_sim(net):
    """
    Initialize NEURON simulation variables

    Parameters
    ----------
    net : Network object
        The Network object with parameter values
    Returns
    -------
    t_vec : Vector
          Vector that has been connected to time ref in NEURON
    dp_rec_L2 : Vector
          Vector that has been connected to L2 dipole ref in NEURON
    dp_rec_L5 : Vector
          Vector that has been connected to L5 dipole ref in NEURON
    """

    from neuron import h
    h.load_file("stdrun.hoc")

    # Set tstop before instantiating any classes
    h.tstop = net.params['tstop']
    h.dt = net.params['dt']  # simulation duration and time-step
    h.celsius = net.params['celsius']  # 37.0 - set temperature

    # create or reinitialize scalars in NEURON (hoc) context
    h("dp_total_L2 = 0.")
    h("dp_total_L5 = 0.")

    # Connect NEURON scalar references to python vectors
    t_vec = h.Vector(int(h.tstop / h.dt + 1)).record(h._ref_t)  # time recording
    dp_rec_L2 = h.Vector(int(h.tstop / h.dt + 1)).record(h._ref_dp_total_L2)  # L2 dipole recording
    dp_rec_L5 = h.Vector(int(h.tstop / h.dt + 1)).record(h._ref_dp_total_L5)  # L5 dipole recording

    return t_vec, dp_rec_L2, dp_rec_L5


def simulate_dipole(net, trial=0, verbose=True, extdata=None):
    """Simulate a dipole given the experiment parameters.

    Parameters
    ----------
    net : Network object
        The Network object specifying how cells are
        connected.
    trial : int
        Current trial number
    verbose: bool
        False will turn off "Simulation time" messages
    extdata : np.Array | None
        Array with preloaded data to compare simulation
        results against

    Returns
    -------
    dpl: instance of Dipole
        The dipole object
    """
    from .parallel import rank, nhosts, pc, cvode
    from os import path

    from neuron import h
    h.load_file("stdrun.hoc")
    t_vec, dp_rec_L2, dp_rec_L5 = initialize_sim(net)

    # make sure network state is consistent
    net.state_init()

    if trial != 0:
        # for reproducibility of original HNN results
        net.reset_src_event_times()

    # Now let's simulate the dipole

    if verbose:
        pc.barrier()  # sync for output to screen
        if rank == 0:
            print("Running trial %d (on %d cores)" % (trial + 1, nhosts))

    # initialize cells to -65 mV, after all the NetCon
    # delays have been specified
    h.finitialize()

    def prsimtime():
        print('Simulation time: {0} ms...'.format(round(h.t, 2)))

    printdt = 10
    if verbose and rank == 0:
        for tt in range(0, int(h.tstop), printdt):
            cvode.event(tt, prsimtime)  # print time callbacks

    h.fcurrent()

    pc.barrier()  # get all nodes to this place before continuing

    # actual simulation - run the solver
    pc.psolve(h.tstop)

    #pc.barrier()

    # these calls aggregate data across procs/nodes
    pc.allreduce(dp_rec_L2, 1)
    # combine dp_rec on every node, 1=add contributions together
    pc.allreduce(dp_rec_L5, 1)
    # aggregate the currents independently on each proc
    #net.aggregate_currents()
    #pc.barrier()
    # combine net.current{} variables on each proc
    pc.allreduce(net.current['L5Pyr_soma'], 1)
    pc.allreduce(net.current['L2Pyr_soma'], 1)

    #pc.barrier()  # get all nodes to this place before continuing

    dpl = None
    if rank == 0:
        dpl_data = np.c_[np.array(dp_rec_L2.to_python()) +
                        np.array(dp_rec_L5.to_python()),
                        np.array(dp_rec_L2.to_python()),
                        np.array(dp_rec_L5.to_python())]

        dpl = Dipole(np.array(t_vec.to_python()), dpl_data)

        if net.params['save_dpl']:
            if 'task_id' in net.params:
                idstr = '%d_%d' % (net.params['task_id'], trial)
            else:
                idstr = '%d' % (trial)
            fname = path.join('data', net.params['sim_prefix'], 'dipoles',
                              'rawdpl_%s.json' % idstr)
            dpl.write(fname)

        dpl.baseline_renormalize(net.params)
        dpl.convert_fAm_to_nAm()
        dpl.scale(net.params['dipole_scalefctr'])
        dpl.smooth(net.params['dipole_smooth_win'] / h.dt)

    return dpl


def average_dipoles(dpls):
    """Compute average over a list of Dipole objects.

    Parameters
    ----------
    dpls: list of Dipole objects
        Contains list of dipole results to be averaged

    Returns
    -------
    dpl: instance of Dipole
        A dipole object with averages of the dipole data
    """

    # need at least on Dipole to get times
    assert (len(dpls) > 0)

    agg_avg = np.mean(np.array([dpl.dpl['agg'] for dpl in dpls]), axis=0)
    L5_avg = np.mean(np.array([dpl.dpl['L5'] for dpl in dpls]), axis=0)
    L2_avg = np.mean(np.array([dpl.dpl['L2'] for dpl in dpls]), axis=0)

    avg_dpl_data = np.c_[agg_avg,
                         L2_avg,
                         L5_avg]

    avg_dpl = Dipole(dpls[0].t, avg_dpl_data)

    return avg_dpl


class Dipole(object):
    """Dipole class.

    Parameters
    ----------
    times : array (n_times,)
        The time vector
    data : array (n_times x 3)
        The data. The first column represents 'agg',
        the second 'L2' and the last one 'L5'
    data_cols: int | None
        The number of columns present in data. Must
        be in order 'agg', 'L2', 'L5'. Default is 3
        for HNN compatibility.

    Attributes
    ----------
    t : array
        The time vector
    dpl : dict of array
        The dipole with key 'agg' and optionally, 'L2' and 'L5'
    """

    def __init__(self, times, data, data_cols=3):  # noqa: D102
        self.units = 'fAm'
        self.N = data.shape[0]
        self.t = times
        self.dpl = {}
        if data_cols > 0:
            self.dpl['agg'] = data[:, 0]
        if data_cols > 1:
            self.dpl['L2'] = data[:, 1]
        if data_cols > 2:
            self.dpl['L5'] = data[:, 2]

    # conversion from fAm to nAm
    def convert_fAm_to_nAm(self):
        """ must be run after baseline_renormalization()
        """
        for key in self.dpl.keys():
            self.dpl[key] *= 1e-6
        self.units = 'nAm'

    def scale(self, fctr):
        for key in self.dpl.keys():
            self.dpl[key] *= fctr
        return fctr

    def smooth(self, winsz):
        # XXX: add check to make sure self.t is
        # not smaller than winsz
        if winsz <= 1:
            return
        for key in self.dpl.keys():
            self.dpl[key] = _hammfilt(self.dpl[key], winsz)

    def plot(self, ax=None, layer='agg'):
        """Simple layer-specific plot function.

        Parameters
        ----------
        ax : instance of matplotlib figure | None
            The matplotlib axis
        layer : str
            The layer to plot
        show : bool
            If True, show the figure

        Returns
        -------
        fig : instance of plt.fig
            The matplotlib figure handle.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        if layer in self.dpl.keys():
            ax.plot(self.t, self.dpl[layer])
            ax.set_xlabel('Time (ms)')
        if True:
            plt.show()
        return ax.get_figure()

    def baseline_renormalize(self, params):
        """Only baseline renormalize if the units are fAm.

        Parameters
        ----------
        params : dict
            The parameters
        """
        if self.units != 'fAm':
            print("Warning, no dipole renormalization done because units"
                  " were in %s" % (self.units))
            return
        elif (not 'L2' in self.dpl) and (not 'L5' in self.dpl):
            print("Warning, no dipole renormalization done because"
                  " L2 and L5 components are not available")
            return

        N_pyr_x = params['N_pyr_x']
        N_pyr_y = params['N_pyr_y']
        # N_pyr cells in grid. This is PER LAYER
        N_pyr = N_pyr_x * N_pyr_y
        # dipole offset calculation: increasing number of pyr
        # cells (L2 and L5, simultaneously)
        # with no inputs resulted in an aggregate dipole over the
        # interval [50., 1000.] ms that
        # eventually plateaus at -48 fAm. The range over this interval
        # is something like 3 fAm
        # so the resultant correction is here, per dipole
        # dpl_offset = N_pyr * 50.207
        dpl_offset = {
            # these values will be subtracted
            'L2': N_pyr * 0.0443,
            'L5': N_pyr * -49.0502
            # 'L5': N_pyr * -48.3642,
            # will be calculated next, this is a placeholder
            # 'agg': None,
        }
        # L2 dipole offset can be roughly baseline shifted over
        # the entire range of t
        self.dpl['L2'] -= dpl_offset['L2']
        # L5 dipole offset should be different for interval [50., 500.]
        # and then it can be offset
        # slope (m) and intercept (b) params for L5 dipole offset
        # uncorrected for N_cells
        # these values were fit over the range [37., 750.)
        m = 3.4770508e-3
        b = -51.231085
        # these values were fit over the range [750., 5000]
        t1 = 750.
        m1 = 1.01e-4
        b1 = -48.412078
        # piecewise normalization
        self.dpl['L5'][self.t <= 37.] -= dpl_offset['L5']
        self.dpl['L5'][(self.t > 37.) & (self.t < t1)] -= N_pyr * \
            (m * self.t[(self.t > 37.) & (self.t < t1)] + b)
        self.dpl['L5'][self.t >= t1] -= N_pyr * \
            (m1 * self.t[self.t >= t1] + b1)
        # recalculate the aggregate dipole based on the baseline
        # normalized ones
        self.dpl['agg'] = self.dpl['L2'] + self.dpl['L5']

    def write(self, fname):
        """Write dipole values to a file.

        Parameters
        ----------
        fname : str
            Full path to the output file (.txt)
        """
        from os import makedirs, path

        makedirs(path.dirname(fname), exist_ok=True)

        cols = [self.dpl.get(key) for key in ['agg', 'L2', 'L5'] if (key in self.dpl)]
        X = np.r_[[self.t] + cols].T
        np.savetxt(fname, X, fmt=['%3.3f', '%5.4f', '%5.4f', '%5.4f'],
                   delimiter='\t')

    def rmse(self, exp_dpl, tstart, tstop):
        """ Calculates RMSE compared to data in exp_dpl """
        from numpy import sqrt
        from scipy import signal

        # make sure start and end times are valid for both dipoles
        exp_start_index = (np.abs(exp_dpl.t - tstart)).argmin()
        exp_end_index = (np.abs(exp_dpl.t - tstop)).argmin()
        exp_length = exp_end_index - exp_start_index

        sim_start_index = (np.abs(self.t - tstart)).argmin()
        sim_end_index = (np.abs(self.t - tstop)).argmin()
        sim_length = sim_end_index - sim_start_index

        dpl1 = self.dpl['agg'][sim_start_index:sim_end_index]
        dpl2 = exp_dpl.dpl['agg'][exp_start_index:exp_end_index]
        if (sim_length > exp_length):
            # downsample simulation timeseries to match exp data
            dpl1 = signal.resample(dpl1, exp_length)
        elif (sim_length < exp_length):
            # downsample exp timeseries to match simulation data
            dpl2 = signal.resample(dpl2, sim_length)

        return np.sqrt(((dpl1 - dpl2) ** 2).mean())
