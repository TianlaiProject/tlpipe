"""Plot eigen-values."""

import itertools
import numpy as np
import h5py
import matplotlib.pyplot as plt

from tlpipe.utils import mpiutil
from tlpipe.core.base_exe import Base
from tlpipe.utils.path_util import input_path, output_path


# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
params_init = {
               'nprocs': mpiutil.size, # number of processes to run this module
               'aprocs': range(mpiutil.size), # list of active process rank no.
               'input_file': 'gain.hdf5',
               'output_file': None, # None, str or a list of str
               'time_index': [0],
               'freq_index': [255],
              }
prefix = 'plte_'



class Plot(Base):
    """Plot eigen-values."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(Plot, self).__init__(parameter_file_or_dict, params_init, prefix, feedback)

    def execute(self):

        input_file = input_path(self.params['input_file'])
        output_file = self.params['output_file']
        time_index = self.params['time_index']
        freq_index = self.params['freq_index']

        if output_file is None:
            output_file = output_path('eigval.png')
        else:
            output_file = output_path(output_file)
        suffix = output_file.split('.')[-1]

        with h5py.File(input_file, 'r') as f:
            eigval = f['eigval'][...]

        lst = list(itertools.product(time_index, freq_index))
        for (ti, fi) in mpiutil.mpilist(lst):
            outfile = output_file.replace('.'+suffix, '_%d_%d.'+suffix)
            plt.figure()
            plt.plot(eigval[ti, :, 0, fi], 'o', label='xx')
            plt.plot(eigval[ti, :, 1, fi], 'o', label='yy')
            plt.xlabel('x')
            plt.ylabel('Eigen-values')
            plt.legend()
            plt.savefig(outfile % (ti, fi))
