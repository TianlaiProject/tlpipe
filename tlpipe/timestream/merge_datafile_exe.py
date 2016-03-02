"""Merge a list of data files to one file."""

try:
    import cPickle as pickle
except ImportError:
    import pickle

import os
import numpy as np
from scipy.linalg import eigh
import h5py
import ephem
import aipy as a

from tlpipe.utils import mpiutil
from tlpipe.core.base_exe import Base
from tlpipe.utils.path_util import input_path, output_path


# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
params_init = {
               'nprocs': mpiutil.size, # number of processes to run this module
               'aprocs': range(mpiutil.size), # list of active process rank no.
               'input_file': ['data1.hdf5', 'data2.hdf5'],
               'output_file': 'data_merge.hdf5',
               'extra_history': '',
              }
prefix = 'md_'


class Merge(Base):
    """Merge a list of data files to one file."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(Merge, self).__init__(parameter_file_or_dict, params_init, prefix, feedback)

    def execute(self):

        input_file = input_path(self.params['input_file'])
        output_file = output_path(self.params['output_file'])

        if mpiutil.rank0:
            # read in data shape info from data files
            data_shps = []
            data_types = []
            for infile in input_file:
                with h5py.File(infile, 'r') as f:
                    dataset = f['data']
                    data_shps.append(dataset.shape)
                    data_types.append(dataset.dtype)

            nts = [ shp[0] for shp in data_shps ]
            nt_cums = np.cumsum([0] + nts)
            nt = np.sum(nts)
            data_shp = (nt,) + data_shps[0][1:]
            data_type = data_types[0]

            with h5py.File(output_file, 'w') as fout:
                out_dset = fout.create_dataset('data', data_shp, dtype=data_type)
                out_time = fout.create_dataset('time', (nt,), dtype=np.float64)

                for fi, infile in enumerate(input_file):
                    with h5py.File(infile, 'r') as fin:
                        in_dset = fin['data']
                        out_dset[nt_cums[fi]:nt_cums[fi+1]] = in_dset[:]
                        out_time[nt_cums[fi]:nt_cums[fi+1]] = fin['time'][:]
                        if fi == 0:
                            # copy metadata from input file
                            for attrs_name, attrs_value in in_dset.attrs.iteritems():
                                out_dset.attrs[attrs_name] = attrs_value
                            # update some attrs
                            out_dset.attrs['history'] = out_dset.attrs['history'] + self.history
                        if fi == len(input_file):

                            out_dset.attrs['end_time'] = in_dset.attrs['end_time']
