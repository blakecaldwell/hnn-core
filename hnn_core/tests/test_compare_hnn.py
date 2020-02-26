from mne.utils import _fetch_file
import hnn_core
from hnn_core import simulate_dipole, read_params, Network


def run_simulation(params, dpl_master):
    net = Network(params)
    dpls = simulate_dipole(net)

    return dpls


def test_hnn_core():
    import os.path as op

    from numpy import loadtxt
    from numpy.testing import assert_array_equal

    from mne.utils import _fetch_file
    import hnn_core
    from hnn_core import Params

    """Test to check if MNE neuron does not break."""
    # small snippet of data on data branch for now. To be deleted
    # later. Data branch should have only commit so it does not
    # pollute the history.
    data_url = ('https://raw.githubusercontent.com/jonescompneurolab/'
                'hnn-core/test_data/dpl.txt')
    if not op.exists('dpl.txt'):
        _fetch_file(data_url, 'dpl.txt')
    dpl_master = loadtxt('dpl.txt')

    hnn_core_root = op.join(op.dirname(hnn_core.__file__), '..')

    # default params
    params_fname = op.join(hnn_core_root, 'param', 'default.json')
    params = read_params(params_fname)

    # run the simulation
    dpl = run_simulation(params, dpl_master)[0]

    # write the dipole to a file and compare
    fname = './dpl2.txt'
    dpl.write(fname)

    dpl_pr = loadtxt(fname)
    assert_array_equal(dpl_pr[:, 2], dpl_master[:, 2])  # L2
    assert_array_equal(dpl_pr[:, 3], dpl_master[:, 3])  # L5


if __name__ == '__main__':
    # started as an MPI child from test_mpi_simulation.py

    from mpi4py import MPI
    comm = MPI.Comm.Get_parent()

    # receive params and dpl_master
    (params, dpl_master) = comm.bcast(comm.Get_rank(), root=0)

    # run the simulation
    dpl = run_simulation(params, dpl_master)[0]

    if comm.Get_rank() == 0:
        # send results back to parent
        comm.send(dpl, dest=0)

    comm.Barrier()
    comm.Disconnect()
    MPI.Finalize()
    exit(0)  # stop the child
