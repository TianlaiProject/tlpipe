"""RFI ct."""

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
               'input_file': 'data_phs2src.hdf5',
               'output_file': 'data_rfi_cut.hdf5',
               'threshold': 0.1,
               'extra_history': '',
              }
prefix = 'rc_'


class RfiCut(Base):
    """RFI cut."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(RfiCut, self).__init__(parameter_file_or_dict, params_init, prefix, feedback)

    def execute(self):

        input_file = input_path(self.params['input_file'])
        output_file = output_path(self.params['output_file'])
        threshold = self.params['threshold']

        with h5py.File(input_file, 'r') as f:
            dset = f['data']
            data_shp = dset.shape
            data_type = dset.dtype
            ants = dset.attrs['ants']
            ts = f['time']
            freq = dset.attrs['freq']

            npol = dset.shape[2]
            nants = len(ants)
            bls = [(ants[i], ants[j]) for i in range(nants) for j in range(i, nants)] # start from 1
            nbls = len(bls)

            lbls, sbl, ebl = mpiutil.split_local(nbls)
            local_bls = range(sbl, ebl)
            # data of bl local to this process
            local_data = dset[:, sbl:ebl, :, :]

            local_data = np.where(np.isnan(local_data), 0, local_data)

        if mpiutil.rank0:
            data_rfi_cut = np.zeros(data_shp, dtype=data_type) # save data phased to src
        else:
            data_rfi_cut = None

        for bi in range(lbls):
            for pi in range(npol):
                data_slice = local_data[:, bi, pi, :].copy()
                data_slice_imag = np.sort(np.abs(data_slice.imag).reshape(-1))
                val = data_slice_imag[int((1 - threshold) * len(data_slice_imag))]
                local_data[:, bi, pi, :] = np.where(np.abs(local_data[:, bi, pi, :].imag)>val, 0, local_data[:, bi, pi, :])


        # Gather data in separate processes
        mpiutil.gather_local(data_rfi_cut, local_data, (0, sbl, 0, 0), root=0, comm=self.comm)

        # save data after phased to src
        if mpiutil.rank0:
            with h5py.File(output_file, 'w') as f:
                dset = f.create_dataset('data', data=data_rfi_cut)
                # copy metadata from input file
                with h5py.File(input_file, 'r') as fin:
                    f.create_dataset('time', data=fin['time'])
                    for attrs_name, attrs_value in fin['data'].attrs.iteritems():
                        dset.attrs[attrs_name] = attrs_value
                # update some attrs
                dset.attrs['history'] = dset.attrs['history'] + self.history
