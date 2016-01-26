"""Module to do the map-making."""

from tlpipe.utils import mpiutil
from tlpipe.core.base_exe import Base


# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
params_init = {
               'nprocs': mpiutil.size, # number of processes to run this module
               'aprocs': range(mpiutil.size), # list of active process rank no.
              }
prefix = 'mp_'

class MapMaking(Base):
    """Class to do the map-making."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(MapMaking, self).__init__(parameter_file_or_dict, params_init, prefix, feedback)

    def execute(self):

        print 'rank %d executing...' % mpiutil.rank

        if mpiutil.rank0:
            print self.history
