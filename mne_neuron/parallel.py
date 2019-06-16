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

    if n_jobs == None:
        # MPI: Initialize the ParallelContext class
        pc = h.ParallelContext()
    else:
        pc = h.ParallelContext(n_jobs)

    pc.done()
    nhosts = int(pc.nhost())  # Find number of hosts
    rank = int(pc.id())     # rank or node number (0 will be the master)
    cvode = h.CVode()

    # be explicit about using fixed step integration
    cvode.active(0)

    # use cache_efficient mode for allocating elements in contiguous order
    cvode.cache_efficient(1)

    if rank == 0:
        pc.gid_clear()

    # sets the default max solver step in ms (purposefully large)
    pc.set_maxstep(10)
