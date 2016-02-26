"""SVD RFI flagging by throwing out given number of eigenmodes."""

try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
from scipy import linalg
import h5py

from tlpipe.utils import mpiutil
from tlpipe.core.base_exe import Base
from tlpipe.utils.path_util import input_path, output_path


# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
params_init = {
               'nprocs': mpiutil.size, # number of processes to run this module
               'aprocs': range(mpiutil.size), # list of active process rank no.
               'input_file': 'data_cal.hdf5',
               'output_file': 'data_svd_rfi.hdf5',
               'nsvd': 100, # number of svd modes
               'save_svdmode': False,
               'extra_history': '',
              }
prefix = 'svr_'



class RfiFlag(Base):
    """SVD RFI flagging by throwing out given number of eigenmodes."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(RfiFlag, self).__init__(parameter_file_or_dict, params_init, prefix, feedback)

    def execute(self):

        input_file = input_path(self.params['input_file'])
        output_file = output_path(self.params['output_file'])
        nsvd = self.params['nsvd']
        save_svdmode = self.params['save_svdmode']

        with h5py.File(input_file, 'r') as f:
            dset = f['data']
            data_type = dset.dtype
            nt, nbls, npol, nfreq = dset.shape

            lpol, spol, epol = mpiutil.split_local(npol)
            local_pols = range(spol, epol)

            local_data = dset[:, :, spol:epol, :]

        if mpiutil.rank0:
            data_rfi_flag = np.zeros((nt, nbls, npol, nfreq), dtype=data_type) # save data that have rfi flagged
        else:
            data_rfi_flag= None

        for pi, pol_ind in enumerate(local_pols): # mpi among pols
            data_slice = local_data[:, :, pi, :].reshape(nt, -1)
            data_slice = np.where(np.isnan(data_slice), 0, data_slice)
            U, s, Vh = linalg.svd(data_slice, full_matrices=False, overwrite_a=True)

            local_data[:, :, pi, :] = np.dot(U[:, nsvd:] * s[nsvd:], Vh[nsvd:, :]).reshape((nt, nbls, nfreq))


        # Gather data in separate processes
        mpiutil.gather_local(data_rfi_flag, local_data, (0, 0, spol, 0), root=0, comm=self.comm)

        # save data rfi flagged
        if mpiutil.rank0:
            with h5py.File(output_file, 'w') as f:
                dset = f.create_dataset('data', data=data_rfi_flag)
                ### shold save all 4 pol s, U, Vh
                # if save_svdmode:
                #     f.create_dataset('s', data = s[:nsvd])
                #     f.create_dataset('U', data = U[:, :nsvd])
                #     f.create_dataset('Vh', data = Vh[:nsvd, :])

                # copy metadata from input file
                with h5py.File(input_file, 'r') as fin:
                    f.create_dataset('time', data=fin['time'])
                    for attrs_name, attrs_value in fin['data'].attrs.iteritems():
                        dset.attrs[attrs_name] = attrs_value
                # update some attrs
                dset.attrs['history'] = dset.attrs['history'] + self.history
