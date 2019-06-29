"""import NEURON module"""

# Authors: Blake Caldwell <blake_caldwell@brown.edu>

from neuron import h


def shutdown():
    pc.done()
    h.quit()


def get_rank():
    return rank


def create_parallel_context(n_jobs=None):
    """Create parallel context.

    Parameters
    ----------
    n_jobs: int | None
        Number of processors to use for a simulation.
        A value of None will allow NEURON to use all
        available processors.
    """

    global rank, nhosts, cvode, pc
    nhosts = n_jobs
    rank = 0

    from mpi4py import MPI
    from math import floor

    comm = MPI.COMM_WORLD
    size = comm.Get_size()

    pc = h.ParallelContext()
    if isinstance(n_jobs, int):
        if (size%n_jobs) == 0:
            pc.subworlds(size)
        else:
            pc.subworlds(size)
    else:
        # MPI: Initialize the ParallelContext class
        pc = h.ParallelContext()

    pc.done()
    nhosts = int(pc.nhost())  # Find number of hosts
    rank = int(pc.id())     # rank or node number (0 will be the master)
    if rank == 0:
        print("There are %d subworlds, each with %d procs" % (pc.nhost_bbs(), pc.nhost()))
    cvode = h.CVode()

    # be explicit about using fixed step integration
    cvode.active(0)

    # use cache_efficient mode for allocating elements in contiguous order
    #cvode.cache_efficient(1)

    # sets the default max solver step in ms (purposefully large)
    pc.set_maxstep(10)

    pc.runworker()
    return pc

