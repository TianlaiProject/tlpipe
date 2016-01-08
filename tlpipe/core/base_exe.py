"""A abstract base class template to be inherited by sub-classes."""

import abc

from tlpipe.kiyopy import parse_ini
from tlpipe.utils import mpiutil


# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
params_init = {
               'nprocs': mpiutil.size, # number of processes to run this module
               'aprocs': range(mpiutil.size), # list of active process rank no.
               'extra_history': '',
              }
prefix = 'bs_'



class Base(object):
    """A abstract base class template to be inherited by sub-classes."""

    __metaclass__ = abc.ABCMeta  # Enforce Abstract class


    def __init__(self, parameter_file_or_dict=None, params_init=params_init, prefix=prefix, feedback=2):

        # Read in the parameters.
        self.params = parse_ini.parse(parameter_file_or_dict, params_init,
                                 prefix=prefix, feedback=feedback)
        self.feedback = feedback
        nprocs = min(self.params['nprocs'], mpiutil.size)
        procs = set(range(mpiutil.size))
        aprocs = set(self.params['aprocs']) & procs
        self.aprocs = (list(aprocs) + list(set(range(nprocs)) - aprocs))[:nprocs]
        assert 0 in self.aprocs, 'Process 0 must be active'
        self.comm = mpiutil.active_comm(self.aprocs) # communicator consists of active processes

    @property
    def history(self):
        """History that will be added to the output file."""

        hist = 'Execute %s.%s with %s.\n' % (__name__, self.__class__.__name__, self.params)
        if self.params['extra_history'] != '':
            hist = self.params['extra_history'] + ' ' + hist

        return hist

    @abc.abstractmethod
    def execute(self):
        """Abstract method that needs to be implemented by sub-classes."""

        return
