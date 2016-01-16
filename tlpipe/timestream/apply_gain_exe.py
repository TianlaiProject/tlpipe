"""Applying gain to the visibility data."""

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
               'input_file': ['data1_conv.hdf5', 'data2_conv.hdf5'],
               'output_file': 'data_cal.hdf5',
               'gain_file': 'gain.hdf5',
               'extra_history': '',
              }
prefix = 'ag_'


pol_dict = {0: 'xx', 1: 'yy', 2: 'xy', 3: 'yx'}


class ApplyGain(Base):
    """Applying gain to the visibility data."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(ApplyGain, self).__init__(parameter_file_or_dict, params_init, prefix, feedback)

    def execute(self):

        input_file = input_path(self.params['input_file'])
        output_file = output_path(self.params['output_file'])
        gain_file = output_path(self.params['gain_file'])

        with h5py.File(gain_file, 'r') as f:
            gain = f['gain'][...]
            mean_gain = np.mean(gain, axis=0)

        # read in ants, freq, time info from data files
        with h5py.File(input_file[0], 'r') as f:
            dataset = f['data']
            data_shp = dataset.shape
            data_type = dataset.dtype
            ants = dataset.attrs['ants']
            freq = dataset.attrs['freq']

        npol = data_shp[2]
        nfreq = len(freq)

        nants = len(ants)
        bls = [(ants[i], ants[j]) for i in range(nants) for j in range(i, nants)]
        nbls = len(bls)

        lbls, sbl, ebl = mpiutil.split_local(nbls)
        local_bls = range(sbl, ebl)
        # local data section corresponding to local bls
        local_data = np.array([], dtype=data_type).reshape((0, lbls, npol, nfreq))
        ts = np.array([], dtype=np.float64)
        for ind, data_file in enumerate(input_file):
            with h5py.File(data_file, 'r') as f:
                local_data = np.concatenate((local_data, f['data'][:, sbl:ebl, :, :]), axis=0)
                ts = np.concatenate((ts, f['time'][...])) # Julian date
        nt = len(ts)

        if mpiutil.rank0:
            data_cal = np.zeros((nt,) + data_shp[1:], dtype=data_type) # save data that phased to zenith
        else:
            data_cal = None

        for pol_ind in [0, 1]: # only xx, yy
            for bi, bl_ind in enumerate(local_bls): # mpi among bls
                data_slice = local_data[:, bi, pol_ind, :].copy() # will use local_data to save data_slice_dphs in-place, so here use copy
                ai, aj = bls[bl_ind]
                i = ants.tolist().index(ai)
                j = ants.tolist().index(aj)
                bl_gain = mean_gain[i, pol_ind] * mean_gain[j, pol_ind].conj()
                local_data[:, bi, pol_ind, :] = data_slice / bl_gain

        # Gather data in separate processes
        mpiutil.gather_local(data_cal, local_data, (0, sbl, 0, 0), root=0, comm=self.comm)

        # save data after applying gainCas
        if mpiutil.rank0:
            with h5py.File(output_file, 'w') as f:
                f.create_dataset('time', data=ts)
                dset = f.create_dataset('data', data=data_cal)
                # copy metadata from input file
                with h5py.File(input_file[0], 'r') as fin:
                    for attrs_name, attrs_value in fin['data'].attrs.iteritems():
                        dset.attrs[attrs_name] = attrs_value
                # update some attrs
                dset.attrs['history'] = dset.attrs['history'] + self.history
