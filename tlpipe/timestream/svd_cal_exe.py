"""Module to do the calibration."""

try:
    import cPickle as pickle
except ImportError:
    import pickle

import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.linalg import eigh, inv, pinv2, LinAlgError
import aipy as a
# import ephem
import h5py

from tlpipe.kiyopy import parse_ini
from tlpipe.utils import mpiutil


# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
params_init = {
               'nprocs': mpiutil.size, # number of processes to run this module
               'aprocs': range(mpiutil.size), # list of active process rank no.
               # 'data_dir': './',  # directory the data in
               'data_phs2zen_file': 'data_phs2zen.hdf5',
               'data_int_time_file': 'data_int_time.hdf5',
               'output_dir': './output/', # output directory
              }
prefix = 'sc_'


pol_dict = {0: 'xx', 1: 'yy', 2: 'xy', 3: 'yx'}


class SVDCal(object):
    """Calibration."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        # Read in the parameters.
        self.params = parse_ini.parse(parameter_file_or_dict, params_init,
                                 prefix=prefix, feedback=feedback)
        self.feedback = feedback
        nprocs = min(self.params['nprocs'], mpiutil.size)
        procs = set(range(mpiutil.size))
        aprocs = set(self.params['aprocs']) & procs
        self.aprocs = (list(aprocs) + list(set(range(nprocs)) - aprocs))[:nprocs]
        assert 0 in self.aprocs, 'Process 0 must be active'
        self.comm = mpiutil.active_comm(self.aprocs) # communicator consists of active processes

    def execute(self):

        output_dir = self.params['output_dir']
        if mpiutil.rank0:
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

        with h5py.File(self.params['data_phs2zen_file'], 'r') as f:
            dset = f['data_phs2zen']
            data_phs2zen = dset[...]
            ants = dset.attrs['ants']
            ts = dset.attrs['ts']
            freq = dset.attrs['freq']
            bls = pickle.loads(dset.attrs['bls']) # as list
            az = dset.attrs['az']
            alt = dset.attrs['alt']
        with h5py.File(self.params['data_int_time_file'], 'r') as f:
            data_int_time = f['data_int_time'][...]

        npol = data_phs2zen.shape[2]
        nt = len(ts)
        nfreq = len(freq)
        nants = len(ants)
        nbls = len(bls)


        if self.comm is not None:
            assert self.comm.size <= nfreq, 'Can not have nprocs (%d) > nfreq (%d)' % (self.comm.size, nfreq)


        data_cal = np.zeros_like(data_phs2zen) # save data after cal
        # construct visiblity matrix for a single freq
        Vmat = np.zeros((2*nants, 2*nants), dtype=data_phs2zen.dtype)
        for fi in mpiutil.mpirange(nfreq): # mpi among freq
            for i, ai in enumerate(ants):
                for j, aj in enumerate(ants):
                    try:
                        ind = bls.index((ai, aj))
                        Vmat[2*i, 2*j] = data_int_time[ind, 0, fi] # xx
                        Vmat[2*i+1, 2*j+1] = data_int_time[ind, 1, fi] # yy
                        Vmat[2*i, 2*j+1] = data_int_time[ind, 2, fi] # xy
                        Vmat[2*i+1, 2*j] = data_int_time[ind, 3, fi] # yx
                    except ValueError:
                        ind = bls.index((aj, ai))
                        Vmat[2*i, 2*j] = data_int_time[ind, 0, fi].conj() # xx
                        Vmat[2*i+1, 2*j+1] = data_int_time[ind, 1, fi].conj() # yy
                        Vmat[2*i, 2*j+1] = data_int_time[ind, 2, fi].conj() # xy
                        Vmat[2*i+1, 2*j] = data_int_time[ind, 3, fi].conj() # yx
            # Eigen decomposition
            s, U = eigh(Vmat)
            # # plot eig val
            # plt.figure()
            # plt.plot(s[::-1], 'o') # in descending order
            # plt.ylabel('Eigen value')
            # plt.savefig(output_dir + 'eig_val_%d.png' % fi)
            # plt.close()
            # the gain matrix for this freq
            Gmat = U[:, -2:] * np.sqrt(s[-2:]) # only the 2 maximum eigen-vals
            # calibrate for this freq
            # construct nt x 2 x 2 visibility for this freq
            Vij = np.zeros((nt, 2, 2), dtype=data_phs2zen.dtype)
            for i, ai in enumerate(ants):
                for j, aj in enumerate(ants):
                    try:
                        ind = bls.index((ai, aj))
                        Vij[:, 0, 0] = data_phs2zen[:, ind, 0, fi] # xx
                        Vij[:, 1, 1] = data_phs2zen[:, ind, 1, fi] # yy
                        Vij[:, 0, 1] = data_phs2zen[:, ind, 2, fi] # xy
                        Vij[:, 1, 0] = data_phs2zen[:, ind, 3, fi] # yx
                    except ValueError:
                        ind = bls.index((aj, ai))
                        Vij[:, 0, 0] = data_phs2zen[:, ind, 0, fi].conj() # xx
                        Vij[:, 1, 1] = data_phs2zen[:, ind, 1, fi].conj() # yy
                        Vij[:, 0, 1] = data_phs2zen[:, ind, 2, fi].conj() # xy
                        Vij[:, 1, 0] = data_phs2zen[:, ind, 3, fi].conj() # yx
                    # 2x2 gain for this freq
                    Gi = Gmat[2*i:2*(i+1)]
                    Gj = Gmat[2*j:2*(j+1)]
                    try:
                        Giinv = inv(Gi)
                    except LinAlgError:
                        Giinv = pinv2(Gi)
                    try:
                        GjHinv = inv(Gj.T.conj())
                    except LinAlgError:
                        GjHinv = pinv2(Gj.T.conj())
                    # nt x 2 x 2 visibility after calibrate
                    VijGj = np.dot(Vij, GjHinv)
                    Vij_cal = np.dot(Giinv[np.newaxis, :, :], VijGj)[0].swapaxes(0, 1)

                    data_cal[:, ind, 0, fi] = Vij_cal[:, 0, 0] # xx
                    data_cal[:, ind, 1, fi] = Vij_cal[:, 1, 1] # yy
                    data_cal[:, ind, 2, fi] = Vij_cal[:, 0, 1] # xy
                    data_cal[:, ind, 3, fi] = Vij_cal[:, 1, 0] # yx


        # Reduce data in separate processes
        if self.comm is not None and self.comm.size > 1: # Reduce only when there are multiple processes
            if mpiutil.rank0:
                self.comm.Reduce(mpiutil.IN_PLACE, data_cal, op=mpiutil.SUM, root=0)
            else:
                self.comm.Reduce(data_cal, data_cal, op=mpiutil.SUM, root=0)

        # save data after cal
        if mpiutil.rank0:
            with h5py.File(output_dir + 'data_cal.hdf5', 'w') as f:
                dset = f.create_dataset('data_cal', data=data_cal)
                dset.attrs['ants'] = ants
                dset.attrs['ts'] = ts
                dset.attrs['freq'] = freq
                dset.attrs['bls'] = pickle.dumps(bls) # save as list
                dset.attrs['az'] = az
                dset.attrs['alt'] = alt

            # convert to Stokes I, Q, U, V
            data_cal_stokes = np.zeros_like(data_cal)
            data_cal_stokes[:, :, 0, :] = 0.5 * (data_cal[:, :, 0] + data_cal[:, :, 1]) # I
            data_cal_stokes[:, :, 1, :] = 0.5 * (data_cal[:, :, 0] - data_cal[:, :, 1]) # Q
            data_cal_stokes[:, :, 2, :] = 0.5 * (data_cal[:, :, 2] + data_cal[:, :, 3]) # U
            data_cal_stokes[:, :, 3, :] = -0.5J * (data_cal[:, :, 2] - data_cal[:, :, 3]) # V

            # save stokes data
            with h5py.File(output_dir + 'data_cal_stokes.hdf5', 'w') as f:
                dset = f.create_dataset('data_cal_stokes', data=data_cal_stokes)
                dset.attrs['ants'] = ants
                dset.attrs['ts'] = ts
                dset.attrs['freq'] = freq
                dset.attrs['bls'] = pickle.dumps(bls) # save as list
                dset.attrs['az'] = az
                dset.attrs['alt'] = alt
