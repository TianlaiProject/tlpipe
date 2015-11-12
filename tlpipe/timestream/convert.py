"""Module to do data conversion."""

from tlpipe.kiyopy import parse_ini
from tlpipe.utils import mpiutil


# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
params_init = {
               'nprocs': mpiutil.size, # number of processes to run this module
               'aprocs': range(mpiutil.size), # list of active process rank no.
              }
prefix = 'cv_'

class Conversion(object):
    """Class to do data converion."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        mpiutil.barrier()

        # Read in the parameters.
        self.params = parse_ini.parse(parameter_file_or_dict, params_init,
                                 prefix=prefix, feedback=feedback)
        self.feedback = feedback
        nprocs = min(self.params['nprocs'], mpiutil.size)
        procs = set(range(mpiutil.size))
        aprocs = set(self.params['aprocs']) & procs
        self.aprocs = (list(aprocs) + list(set(range(nprocs)) - aprocs))[:nprocs]
        assert 0 in self.aprocs, 'Process 0 must be active'
        self.comm = mpiutil.active(self.aprocs) # communicator consists of active processes

    def __del__(self):
        # close the waiting of inactive processes and synchronize
        mpiutil.close(self.aprocs)

    def execute(self):

        print 'rank %d executing...' % mpiutil.rank

        if mpiutil.rank0:
            print 'Data conversion comes soon...'
