"""Module to do the calibration."""

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
               'output_file': 'data_svdcal.hdf5',
               'eigval_file': 'singval.hdf5',
               'gain_file': 'gain.hdf5',
               'extra_history': '',
              }
prefix = 'sc_'


pol_dict = {0: 'xx', 1: 'yy', 2: 'xy', 3: 'yx'}


class SVDCal(Base):
    """Calibration."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(SVDCal, self).__init__(parameter_file_or_dict, params_init, prefix, feedback)

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
                singval = np.zeros((nfreq, 2*nants), dtype=np.float64)
                gain = np.zeros((nfreq, 2*nants, 2), dtype=np.complex128)
            else:
                data_cal = None
                singval = None
                gain = None

            # local data_cal section corresponding to local freq
            local_data_cal = np.zeros_like(dset[:, :, :, sfreq:efreq])
            local_singval = np.zeros((lfreq, 2*nants), dtype=np.float64)
            local_gain = np.zeros((lfreq, 2*nants, 2), dtype=np.complex128)

            # construct visibility matrix for a single freq
            Vmat = np.zeros((2*nants, 2*nants), dtype=data_type)
            for fi, freq_ind in enumerate(local_freq): # mpi among freq
                for i, ai in enumerate(ants):
                    for j, aj in enumerate(ants):
                        try:
                            ind = bls.index((ai, aj))
                            Vmat[2*i, 2*j] = data_int_time[ind, 0, freq_ind] # xx
                            Vmat[2*i+1, 2*j+1] = data_int_time[ind, 1, freq_ind] # yy
                            Vmat[2*i, 2*j+1] = data_int_time[ind, 2, freq_ind] # xy
                            Vmat[2*i+1, 2*j] = data_int_time[ind, 3, freq_ind] # yx
                        except ValueError:
                            ind = bls.index((aj, ai))
                            Vmat[2*i, 2*j] = data_int_time[ind, 0, freq_ind].conj() # xx
                            Vmat[2*i+1, 2*j+1] = data_int_time[ind, 1, freq_ind].conj() # yy
                            Vmat[2*i, 2*j+1] = data_int_time[ind, 2, freq_ind].conj() # xy
                            Vmat[2*i+1, 2*j] = data_int_time[ind, 3, freq_ind].conj() # yx
                # Eigen decomposition
                s, U = eigh(Vmat)
                local_singval[fi] = s[::-1] # descending order
                # the gain matrix for this freq
                Gmat = U[:, -2:] * np.sqrt(s[-2:]) # only the 2 maximum eigen-vals
                local_gain[fi] = Gmat
                # calibrate for this freq
                # construct nt x 2 x 2 visibility for this freq
                Vij = np.zeros((nt, 2, 2), dtype=dset.dtype)
                for i, ai in enumerate(ants):
                    for j, aj in enumerate(ants):
                        try:
                            ind = bls.index((ai, aj))
                            Vij[:, 0, 0] = dset[:, ind, 0, freq_ind] # xx
                            Vij[:, 1, 1] = dset[:, ind, 1, freq_ind] # yy
                            Vij[:, 0, 1] = dset[:, ind, 2, freq_ind] # xy
                            Vij[:, 1, 0] = dset[:, ind, 3, freq_ind] # yx
                        except ValueError:
                            ind = bls.index((aj, ai))
                            Vij[:, 0, 0] = dset[:, ind, 0, freq_ind].conj() # xx
                            Vij[:, 1, 1] = dset[:, ind, 1, freq_ind].conj() # yy
                            Vij[:, 0, 1] = dset[:, ind, 2, freq_ind].conj() # xy
                            Vij[:, 1, 0] = dset[:, ind, 3, freq_ind].conj() # yx
                        # 2x2 gain for this freq
                        Gi = Gmat[2*i:2*(i+1)]
                        Gj = Gmat[2*j:2*(j+1)]
                        try:
                            Giinv = inv(Gi)
                        except LinAlgError:
                            Giinv = pinv2(Gi)
                        try:
                            GjHinv = inv(Gj.T.conj())
                        except LinAlgError as e:
                            print e
                            GjHinv = pinv2(Gj.T.conj())
                        # nt x 2 x 2 visibility after calibrate
                        VijGj = np.dot(Vij, GjHinv)
                        Vij_cal = np.dot(Giinv[np.newaxis, :, :], VijGj)[0].swapaxes(0, 1)

                        local_data_cal[:, ind, 0, fi] = Vij_cal[:, 0, 0] # xx
                        local_data_cal[:, ind, 1, fi] = Vij_cal[:, 1, 1] # yy
                        local_data_cal[:, ind, 2, fi] = Vij_cal[:, 0, 1] # xy
                        local_data_cal[:, ind, 3, fi] = Vij_cal[:, 1, 0] # yx


        # Gather data in separate processes
        if self.comm is not None and self.comm.size > 1: # Reduce only when there are multiple processes
            mpiutil.gather_local(data_cal, local_data_cal, (0, 0, 0, sfreq), root=0, comm=self.comm)
            mpiutil.gather_local(singval, local_singval, (sfreq, 0), root=0, comm=self.comm)
            mpiutil.gather_local(gain, local_gain, (sfreq, 0, 0), root=0, comm=self.comm)

        # save data after cal
        if mpiutil.rank0:
            # save singval
            with h5py.File(eigval_file, 'w') as f:
                dset = f.create_dataset('singval', data=singval)
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
