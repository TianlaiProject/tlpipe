"""Module to do un-polarized calibration using eigen-decomposition."""

try:
    import cPickle as pickle
except ImportError:
    import pickle

import os
import numpy as np
from scipy.linalg import eigh, inv, pinv2, LinAlgError
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
               'output_file': 'data_eigcal.hdf5',
               'eigval_file': 'eigval.hdf5',
               'gain_file': 'gain.hdf5',
               'extra_history': '',
              }
prefix = 'ec_'


pol_dict = {0: 'xx', 1: 'yy', 2: 'xy', 3: 'yx'}


class EigCal(Base):
    """Un-polarized calibration using eigen-decomposition."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(EigCal, self).__init__(parameter_file_or_dict, params_init, prefix, feedback)

    def execute(self):

        input_file = input_path(self.params['input_file'])
        output_file = output_path(self.params['output_file'])
        eigval_file = output_path(self.params['eigval_file'])
        gain_file = output_path(self.params['gain_file'])

        with h5py.File(input_file, 'r') as f:
            dset = f['data']
            data_shp = dset.shape
            data_type = dset.dtype
            data_int_time = f['data_int_time'][...]
            # data_phs2zen = dset[...]
            ants = dset.attrs['ants']
            ts = f['time'][...]
            freq = dset.attrs['freq']

            npol = dset.shape[2]
            nt = len(ts)
            nfreq = len(freq)
            nants = len(ants)
            bls = [(ants[i], ants[j]) for i in range(nants) for j in range(i, nants)]
            nbls = len(bls)

            lfreq, sfreq, efreq = mpiutil.split_local(nfreq)
            local_freq = range(sfreq, efreq)

            # save data after cal
            if mpiutil.rank0:
                data_cal = np.zeros_like(dset)
                eigval = np.zeros((nfreq, nants), dtype=np.float64)
                gain = np.zeros((nfreq, nants), dtype=np.complex128)
            else:
                data_cal = None
                eigval = None
                gain = None

            # local data_cal section corresponding to local freq
            local_data_cal = np.zeros_like(dset[:, :, :, sfreq:efreq])
            local_eigval = np.zeros((lfreq, nants), dtype=np.float64)
            local_gain = np.zeros((lfreq, nants), dtype=np.complex128)

            # construct visibility matrix for a single freq
            Vmat = np.zeros((nants, nants), dtype=data_type)
            for pol in [0, 1]: # cal for 0:xx and 1:yy
                for fi, freq_ind in enumerate(local_freq): # mpi among freq
                    for i, ai in enumerate(ants):
                        for j, aj in enumerate(ants):
                            try:
                                ind = bls.index((ai, aj))
                                Vmat[i, j] = data_int_time[ind, pol, freq_ind] # xx, yy
                            except ValueError:
                                ind = bls.index((aj, ai))
                                Vmat[i, j] = data_int_time[ind, pol, freq_ind].conj() # xx, yy
                    # Eigen decomposition
                    s, U = eigh(Vmat)
                    local_eigval[fi] = s[::-1] # descending order
                    # max eigen-val
                    lbd = s[-1] # lambda
                    # the gain matrix for this freq
                    gvec = U[:, -1] # only eigen-vector corresponding to the maximum eigen-val
                    local_gain[fi] = gvec
                    # calibrate for this freq
                    # construct (nt,) visibility for this pol and freq
                    Vij = np.zeros(nt, dtype=dset.dtype)
                    for i, ai in enumerate(ants):
                        for j, aj in enumerate(ants):
                            try:
                                ind = bls.index((ai, aj))
                                Vij[:] = dset[:, ind, pol, freq_ind] # xx, yy
                            except ValueError:
                                ind = bls.index((aj, ai))
                                Vij[:] = dset[:, ind, pol, freq_ind].conj() # xx, yy
                            # gain for this pol and freq
                            Gij = gvec[i] * np.conj(gvec[j]) # now only correct for the phase
                            Vij_cal = np.conj(Gij) / np.abs(Gij) * Vij

                            local_data_cal[:, ind, pol, fi] = Vij_cal # xx, yy


        # Gather data in separate processes
        if self.comm is not None and self.comm.size > 1: # Reduce only when there are multiple processes
            mpiutil.gather_local(data_cal, local_data_cal, (0, 0, 0, sfreq), root=0, comm=self.comm)
            mpiutil.gather_local(eigval, local_eigval, (sfreq, 0), root=0, comm=self.comm)
            mpiutil.gather_local(gain, local_gain, (sfreq, 0), root=0, comm=self.comm)

        # save data after cal
        if mpiutil.rank0:
            # save eigval
            with h5py.File(eigval_file, 'w') as f:
                dset = f.create_dataset('eigval', data=eigval)
                dset.attrs['ants'] = ants
            # save gain
            with h5py.File(gain_file, 'w') as f:
                dset = f.create_dataset('gain', data=gain)
                dset.attrs['ants'] = ants

            with h5py.File(output_file, 'w') as f:
                dset = f.create_dataset('data', data=data_cal)
                # copy metadata from input file
                with h5py.File(input_file, 'r') as fin:
                    f.create_dataset('time', data=fin['time'])
                    for attrs_name, attrs_value in fin['data'].attrs.iteritems():
                        dset.attrs[attrs_name] = attrs_value
                # update some attrs
                dset.attrs['history'] = dset.attrs['history'] + self.history
