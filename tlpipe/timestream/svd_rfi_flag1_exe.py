"""SVD RFI flagging by throwing out given number of eigenmodes. This is the first stage, which just doing the SVD decomposition."""

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
               'output_file': 'svd_rfi.hdf5',
               'extra_history': '',
              }
prefix = 'svr1_'



class RfiFlag(Base):
    """SVD RFI flagging by throwing out given number of eigenmodes. This is the first stage, which just doing the SVD decomposition."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(RfiFlag, self).__init__(parameter_file_or_dict, params_init, prefix, feedback)

    def execute(self):

        input_file = input_path(self.params['input_file'])
        output_file = output_path(self.params['output_file'])

        with h5py.File(input_file, 'r') as f:
            dset = f['data']
            data_type = dset.dtype
            nt, nbls, npol, nfreq = dset.shape

            lpol, spol, epol = mpiutil.split_local(npol)
            local_pols = range(spol, epol)

            local_data = dset[:, :, spol:epol, :]

            if mpiutil.rank0:
                with h5py.File(output_file, 'w') as fout:
                    nK = min(nt, nbls*nfreq)
                    out_dset = fout.create_dataset('U', (npol, nt, nK), dtype=data_type)
                    fout.create_dataset('s', (npol, nK,), dtype=np.float64)
                    fout.create_dataset('Vh', (npol, nK, nbls*nfreq), dtype=data_type)
                    # copy metadata from input file
                    fout.create_dataset('time', data=f['time'])
                    for attrs_name, attrs_value in dset.attrs.iteritems():
                        out_dset.attrs[attrs_name] = attrs_value
                    # update some attrs
                    out_dset.attrs['history'] = out_dset.attrs['history'] + self.history

        if self.comm is not None:
            self.comm.barrier()


        with h5py.File(output_file, 'r+') as f:
            for pi, pol_ind in enumerate(local_pols): # mpi among pols
                data_slice = local_data[:, :, pi, :].reshape(nt, -1)
                data_slice = np.where(np.isnan(data_slice), 0, data_slice)
                U, s, Vh = linalg.svd(data_slice, full_matrices=False, overwrite_a=True)

                print 'rank = %d, U.shape = %s, s.shape = %s, Vh.shape = %s' % (mpiutil.rank, U.shape, s.shape, Vh.shape)
                f['U'][pol_ind, :, :] = U
                f['s'][pol_ind, :] = s
                f['Vh'][pol_ind, :, :] = Vh

