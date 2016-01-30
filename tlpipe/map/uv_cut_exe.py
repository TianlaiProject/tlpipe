"""Flag uv in the uv-plane."""

try:
    import cPickle as pickle
except ImportError:
    import pickle

import os
import numpy as np
import ephem
import aipy as a
import h5py

from tlpipe.utils import mpiutil
from tlpipe.core.base_exe import Base
from tlpipe.utils.path_util import input_path, output_path


# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
params_init = {
               'nprocs': mpiutil.size, # number of processes to run this module
               'aprocs': range(mpiutil.size), # list of active process rank no.
               'input_file': 'uv.hdf5',
               'output_file': 'uv_cut.hdf5',
               # 'pol': 'I',
               'bl_range': [None, None], # use baseline length in this range only, in unit lambda
               'cut_threshold': 0.1, # cut 10% largest values
               'extra_history': '',
              }
prefix = 'uc_'


class UVCut(Base):
    """Flag uv in the uv-plane."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(UVCut, self).__init__(parameter_file_or_dict, params_init, prefix, feedback)

    def execute(self):

        input_file = input_path(self.params['input_file'])
        output_file = output_path(self.params['output_file'])
        min_bl, max_bl = self.params['bl_range']
        min_bl = min_bl if min_bl is not None else -np.Inf
        max_bl = max_bl if max_bl is not None else np.Inf
        cut_threshold = self.params['cut_threshold']

        if mpiutil.rank0:
            with h5py.File(input_file, 'r') as f:
                uv = f['uv'][...]
                uv_cov = f['uv_cov'][...]
                res = f.attrs['res']
                max_lm = 0.5 * 1.0 / res
                size = uv.shape[0]
                center = size / 2

                # select data in bl_range
                for vi in range(size):
                    for ui in range(size):
                        bl_len = np.sqrt(res**2 * ((ui - center)**2 + (vi - center)**2))
                        if not (bl_len >= min_bl and bl_len <= max_bl):
                            uv[vi, ui] = 0
                            uv_cov[vi, ui] = 0

                # cut data larger than threshold
                uv_imag = uv.imag
                uv_imag_sort = np.sort(np.abs(uv_imag).reshape(-1))
                uv_imag_sort = uv_imag_sort[uv_imag_sort > 0]
                val = uv_imag_sort[int((1 - cut_threshold) * len(uv_imag_sort))]
                uv = np.where(np.abs(uv_imag)>val, 0, uv)
                uv_cov = np.where(np.abs(uv_imag)>val, 0, uv_cov)

                # save data
                with h5py.File(output_file, 'w') as fout:
                    fout.create_dataset('uv_cov', data=uv_cov)
                    fout.create_dataset('uv', data=uv)
                    # copy meta data from input file
                    for attrs_name, attrs_value in f.attrs.iteritems():
                        fout.attrs[attrs_name] = attrs_value
                    # update some attrs
                    fout.attrs['history'] = fout.attrs['history'] + self.history
