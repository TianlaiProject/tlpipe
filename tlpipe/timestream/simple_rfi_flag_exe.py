"""Simple RFI flagging by throughing out values exceed the given threshold."""

try:
    import cPickle as pickle
except ImportError:
    import pickle

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
               'input_file': 'data_cal.hdf5',
               'output_file': 'data_simple_rfi.hdf5',
               'threshold': 3.0, # how much sigma
               'extra_history': '',
              }
prefix = 'sr_'



class RfiFlag(Base):
    """Simple RFI flagging by throughing out values exceed the given threshold."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(RfiFlag, self).__init__(parameter_file_or_dict, params_init, prefix, feedback)

    def execute(self):

        input_file = input_path(self.params['input_file'])
        output_file = output_path(self.params['output_file'])
        threshold = self.params['threshold']

        with h5py.File(input_file, 'r') as f:
            dset = f['data']
            data_shp = dset.shape
            data_type = dset.dtype
            ants = dset.attrs['ants']
            freq = dset.attrs['freq']
            # ts = f['time'] # Julian date for data in this file only

            nt = data_shp[0]
            npol = data_shp[2]
            nfreq = len(freq)

            nants = len(ants)
            bls = [(ants[i], ants[j]) for i in range(nants) for j in range(i, nants)]
            nbls = len(bls)

            lbls, sbl, ebl = mpiutil.split_local(nbls)
            local_bls = range(sbl, ebl)

            local_data = dset[:, sbl:ebl, :, :]


        if mpiutil.rank0:
            data_rfi_flag = np.zeros((nt, nbls, npol, nfreq), dtype=data_type) # save data that have rfi flagged
        else:
            data_rfi_flag= None

        for pol_ind in range(npol):
            for bi, bl_ind in enumerate(local_bls): # mpi among bls

                data_slice = local_data[:, bi, pol_ind, :].copy()
                data_slice = np.where(np.isnan(data_slice), 0, data_slice)
                mean = np.mean(data_slice)
                data_sub_mean = data_slice - mean
                sigma = np.std(np.abs(data_sub_mean))
                # rfi flagging
                local_data[:, bi, pol_ind, :] = np.where(np.abs(data_sub_mean) > threshold * sigma, complex(np.nan, np.nan), local_data[:, bi, pol_ind, :])


        # Gather data in separate processes
        mpiutil.gather_local(data_rfi_flag, local_data, (0, sbl, 0, 0), root=0, comm=self.comm)


        # save data rfi flagged
        if mpiutil.rank0:
            with h5py.File(output_file, 'w') as f:
                dset = f.create_dataset('data', data=data_rfi_flag)
                # copy metadata from input file
                with h5py.File(input_file, 'r') as fin:
                    f.create_dataset('time', data=fin['time'])
                    for attrs_name, attrs_value in fin['data'].attrs.iteritems():
                        dset.attrs[attrs_name] = attrs_value
                # update some attrs
                dset.attrs['history'] = dset.attrs['history'] + self.history
