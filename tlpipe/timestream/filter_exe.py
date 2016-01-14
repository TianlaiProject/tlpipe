"""Filtering out the strongest source."""

try:
    import cPickle as pickle
except ImportError:
    import pickle

import os
import numpy as np
import h5py

from tlpipe.utils import mpiutil
from tlpipe.core.base_exe import Base
from tlpipe.utils.path_util import input_path, output_path


# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
params_init = {
               'nprocs': mpiutil.size, # number of processes to run this module
               'aprocs': range(mpiutil.size), # list of active process rank no.
               'input_file': 'data_phs2zen.hdf5',
               'output_file': 'data_filtering.hdf5',
               'threshold': 0.01, # filtering threshold
               'low_pass': True, # True to filter out components above threshold, else below
               'extra_history': '',
              }
prefix = 'fl_'


pol_dict = {0: 'xx', 1: 'yy', 2: 'xy', 3: 'yx'}


class Filtering(Base):
    """Filtering out the strongest source."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(Filtering, self).__init__(parameter_file_or_dict, params_init, prefix, feedback)

    def execute(self):

        input_file = input_path(self.params['input_file'])
        output_file = output_path(self.params['output_file'])
        threshold = self.params['threshold']
        low_pass = self.params['low_pass']

        # read in ants, freq, time info from data files
        with h5py.File(input_file, 'r') as f:
            dataset = f['data']
            data_shp = dataset.shape
            data_type = dataset.dtype
            ants = dataset.attrs['ants']
            freq = dataset.attrs['freq']
            # ts = f['time'][...] # Julian date for data in this file only

            # nt = ts.shape[0]
            npol = data_shp[2]
            nfreq = len(freq)
            nants = len(ants)
            bls = [(ants[i], ants[j]) for i in range(nants) for j in range(i, nants)]
            nbls = len(bls)

            lbls, sbl, ebl = mpiutil.split_local(nbls)
            local_bls = range(sbl, ebl)
            # local data section corresponding to local bls
            local_data = dataset[:, sbl:ebl, :, :]

        if mpiutil.size == 1:
            data_filtering = local_data.view()
        else:
            if mpiutil.rank0:
                data_filtering = np.zeros(data_shp, dtype=data_type) # save data that phased to zenith
            else:
                data_filtering = None

        for pol_ind in range(npol):
            for bi, bl_ind in enumerate(local_bls): # mpi among bls
                # # ignore auto-correlation
                # bl = bls[bl_ind]
                # if bl[0] == bl[1]:
                #     continue

                data_slice = local_data[:, bi, pol_ind, :].copy() # will use local_data to save data_slice_dphs in-place, so here use copy

                # freq and time fft
                data_slice_fft2 = np.fft.fft2(data_slice)
                data_slice_fft2 = np.fft.fftshift(data_slice_fft2)

                # filtering out the strongest source
                max_fft2 = np.max(np.abs(data_slice_fft2))
                if low_pass:
                    data_slice_fft2 = np.where(np.abs(data_slice_fft2)>threshold*max_fft2, 0, data_slice_fft2)
                else:
                    data_slice_fft2 = np.where(np.abs(data_slice_fft2)<=threshold*max_fft2, 0, data_slice_fft2)

                # inverse fft2
                local_data[:, bi, pol_ind, :] = np.fft.ifft2(np.fft.ifftshift(data_slice_fft2))


        # Gather data in separate processes
        if self.comm is not None and self.comm.size > 1: # Reduce only when there are multiple processes
            mpiutil.gather_local(data_filtering, local_data, (0, sbl, 0, 0), root=0, comm=self.comm)


        # save filtered data
        if mpiutil.rank0:
            with h5py.File(output_file, 'w') as f:
                dset = f.create_dataset('data', data=data_filtering)
                # copy metadata from input file
                with h5py.File(input_file, 'r') as fin:
                    f.create_dataset('time', data=fin['time'])
                    for attrs_name, attrs_value in fin['data'].attrs.iteritems():
                        dset.attrs[attrs_name] = attrs_value
                # update some attrs
                dset.attrs['history'] = dset.attrs['history'] + self.history
