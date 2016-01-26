"""Module to estimate the power spectrum."""

from tlpipe.utils import mpiutil
from tlpipe.core.base_exe import Base


# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
params_init = {
               'nprocs': mpiutil.size, # number of processes to run this module
               'aprocs': range(mpiutil.size), # list of active process rank no.
              }
prefix = 'ps_'

class Ps(Base):
    """Module to estimate the power spectrum."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(Ps, self).__init__(parameter_file_or_dict, params_init, prefix, feedback)

    def execute(self):

        print 'rank %d executing...' % mpiutil.rank

        if mpiutil.rank0:
            print self.history
