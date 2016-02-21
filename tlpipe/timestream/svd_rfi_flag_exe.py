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

        if mpiutil.rank0:
            with h5py.File(input_file, 'r') as fin, h5py.File(output_file, 'w') as fout:
                in_dset = fin['data']
                nt, nbls, npol, nfreq = in_dset.shape
                out_data = np.empty_like(in_dset)
                for pi in range(npol):
                    data = in_dset[:, :, pi, :].reshape(nt, -1)
                    data = np.where(np.isnan(data), 0, data)
                    U, s, Vh = linalg.svd(data, full_matrices=False, overwrite_a=True)

                    out_data[:, :, pi, :] = np.dot(U[:, nsvd:] * s[nsvd:], Vh[nsvd:, :]).reshape((nt, nbls, nfreq))

                out_dset = fout.create_dataset('data', data=out_data)
                if save_svdmode:
                    fout.create_dataset('s', data = s[:nsvd])
                    fout.create_dataset('U', data = U[:, :nsvd])
                    fout.create_dataset('Vh', data = Vh[:nsvd, :])

                fout.create_dataset('time', data=fin['time'])
                for attrs_name, attrs_value in in_dset.attrs.iteritems():
                    out_dset.attrs[attrs_name] = attrs_value
                # update some attrs
                out_dset.attrs['history'] = out_dset.attrs['history'] + self.history
