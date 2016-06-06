"""Module to do calibration by multiply visibility with the conjugate data 12 hours later for North-Pole observation."""

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
               'input_file': 'data_phs2np.hdf5',
               'output_file': 'data_conj_cal.hdf5',
               'offset': 0, # where to start in time axes
               'extra_history': '',
              }
prefix = 'cc_'


sec_per_sidereal_day = 86164
sec_half_sidereal_day = sec_per_sidereal_day / 2


class ConjCal(Base):
    """Module to do calibration by multiply visibility with the conjugate data 12 hours later for North-Pole observation."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(ConjCal, self).__init__(parameter_file_or_dict, params_init, prefix, feedback)

    def execute(self):

        input_file = input_path(self.params['input_file'])
        output_file = output_path(self.params['output_file'])
        offset = self.params['offset']

        with h5py.File(input_file, 'r') as f:
            dset = f['data']
            data_type = dset.dtype
            nt, nbl, npol, nfreq = dset.shape
            assert nt >= offset + sec_per_sidereal_day, 'Invalid time offset'
            ts = f['time'][offset:offset+sec_half_sidereal_day]

            ants = dset.attrs['ants']
            nants = len(ants)
            bls = [(ants[i], ants[j]) for i in range(nants) for j in range(i, nants)]
            nbls = len(bls)

            lbls, sbl, ebl = mpiutil.split_local(nbls)
            local_bls = range(sbl, ebl)

            local_data = dset[offset:offset+sec_half_sidereal_day, sbl:ebl, :, :]
            local_data = np.sqrt(local_data * dset[offset+sec_half_sidereal_day:offset+sec_per_sidereal_day, sbl:ebl, :, :].conj())


        if mpiutil.rank0:
            data_conj_cal = np.zeros((sec_half_sidereal_day, nbls, npol, nfreq), dtype=data_type) # save data that have conj calibrated
        else:
            data_conj_cal= None

        # Gather data in separate processes
        mpiutil.gather_local(data_conj_cal, local_data, (0, sbl, 0, 0), root=0, comm=self.comm)


        # save data conj calibrated
        if mpiutil.rank0:
            with h5py.File(output_file, 'w') as f:
                dset = f.create_dataset('data', data=data_conj_cal)
                # copy metadata from input file
                with h5py.File(input_file, 'r') as fin:
                    f.create_dataset('time', data=ts)
                    for attrs_name, attrs_value in fin['data'].attrs.iteritems():
                        dset.attrs[attrs_name] = attrs_value
                # update some attrs
                dset.attrs['history'] = dset.attrs['history'] + self.history
                ### shold also change start and end time...
