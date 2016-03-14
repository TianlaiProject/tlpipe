"""SVD RFI flagging by throwing out given number of eigenmodes. This is the second stage, which flags RFI by throwing out given number of eigenmodes."""

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
               'input_file': 'svd_rfi.hdf5',
               'output_file': 'data_svd_rfi.hdf5',
               'nsvd': 1, # number of largest svd modes
               'include': False, # if True get `nsvd` RFI modes, else flag out these modes
               'extra_history': '',
              }
prefix = 'svr2_'



class RfiFlag(Base):
    """SVD RFI flagging by throwing out given number of eigenmodes. This is the second stage, which flags RFI by throwing out given number of eigenmodes."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(RfiFlag, self).__init__(parameter_file_or_dict, params_init, prefix, feedback)

    def execute(self):

        input_file = input_path(self.params['input_file'])
        output_file = output_path(self.params['output_file'])
        nsvd = self.params['nsvd']
        include = self.params['include']

        with h5py.File(input_file, 'r') as f:
            dset = f['U']
            data_type = dset.dtype
            npol, nt = dset.shape[0], dset.shape[1]
            nfreq = len(dset.attrs['freq'])
            ants = dset.attrs['ants']
            nant = len(ants)
            bls = [(ants[i], ants[j]) for i in range(nant) for j in range(i, nant)]
            nbl  = len(bls)

            if mpiutil.rank0:
                with h5py.File(output_file, 'w') as fout:
                    out_dset = fout.create_dataset('data', (nt, nbl, npol, nfreq), dtype=data_type)
                    # if not filled with data first, subsequent parallel data writing gets wrong sometimes
                    out_dset[:] = np.array(0.0).astype(data_type)
                    # copy metadata from input file
                    fout.create_dataset('time', data=f['time'])
                    for attrs_name, attrs_value in dset.attrs.iteritems():
                        out_dset.attrs[attrs_name] = attrs_value
                    # update some attrs
                    out_dset.attrs['history'] = out_dset.attrs['history'] + self.history

            mpiutil.barrier(comm=self.comm)

            with h5py.File(output_file, 'r+') as fout:
                for pol_ind in mpiutil.mpirange(npol, comm=self.comm): # mpi among pols
                    if include:
                        fout['data'][:, :, pol_ind, :] = np.dot(f['U'][pol_ind, :, :nsvd] * f['s'][pol_ind, :nsvd], f['Vh'][pol_ind, :nsvd, :]).reshape((nt, nbl, nfreq))
                    else:
                        fout['data'][:, :, pol_ind, :] = np.dot(f['U'][pol_ind, :, nsvd:] * f['s'][pol_ind, nsvd:], f['Vh'][pol_ind, nsvd:, :]).reshape((nt, nbl, nfreq))

                mpiutil.barrier(comm=self.comm)
