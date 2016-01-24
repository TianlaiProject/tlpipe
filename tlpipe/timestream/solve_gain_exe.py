"""Module to solve antenna gain."""

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
from tlpipe.core import tldishes
from tlpipe.utils.pickle_util import get_value
from tlpipe.utils.date_util import get_ephdate
from tlpipe.utils.path_util import input_path, output_path


# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
params_init = {
               'nprocs': mpiutil.size, # number of processes to run this module
               'aprocs': range(mpiutil.size), # list of active process rank no.
               'input_file': ['data1_conv.hdf5', 'data2_conv.hdf5'],
               'output_file': 'gain.hdf5',
               'calibrator': 'cas',
               'catalog': 'misc,helm,nvss',
               'local_transit_time': None, # if None, use transit time in the input_file
               'local_time_zone': 'UTC+08', # time zone of local_transit_time
               'span': 60, # second
               'extra_history': '',
              }
prefix = 'sg_'


class SolveGain(Base):
    """Module to solve antenna gain."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(SolveGain, self).__init__(parameter_file_or_dict, params_init, prefix, feedback)

    def execute(self):

        input_file = input_path(self.params['input_file'])
        output_file = output_path(self.params['output_file'])
        calibrator = self.params['calibrator']
        catalog = self.params['catalog']
        local_transit_time = self.params['local_transit_time']
        local_time_zone = self.params['local_time_zone']
        span = self.params['span']

        with h5py.File(input_file[0], 'r') as f:
            dset = f['data']
            data_shp = dset.shape
            data_type = dset.dtype
            ants = dset.attrs['ants']
            freq = dset.attrs['freq']
            pols = dset.attrs['pol'].tolist()
            az, alt = get_value(dset.attrs['az_alt'])[0]
            az = np.radians(az)
            alt = np.radians(alt)
            npol = dset.shape[2]
            nfreq = len(freq)
            nants = len(ants)
            bls = [(ants[i], ants[j]) for i in range(nants) for j in range(i, nants)]
            nbls = len(bls)

            start_time = get_value(dset.attrs['start_time'])
            # print 'start_time:', start_time
            end_time = get_value(dset.attrs['end_time'])
            # print 'end_time:', end_time
            transit_time_lst = get_value(dset.attrs['transit_time'])
            # print 'transit_time:', transit_time_lst[0]
            int_time = get_value(dset.attrs['int_time'])
            time_zone = get_value(dset.attrs['timezone'])

            start_time = get_ephdate(start_time, time_zone) # utc
            end_time = get_ephdate(end_time, time_zone) # utc
            if local_transit_time is not None:
                transit_time = get_ephdate(local_transit_time, local_time_zone) # utc
            else:
                transit_time = get_ephdate(transit_time_lst[0], time_zone) # utc
            new_start_utc_time = transit_time - span * ephem.second # uct
            new_end_utc_time = transit_time + span * ephem.second # utc
            if new_end_utc_time > end_time:
                need_second_file = True
            else:
                need_second_file = False
            tz = int(time_zone[3:])
            new_start_time = str(ephem.Date(new_start_utc_time + tz * ephem.hour))
            # print 'new_start_time:', new_start_time
            new_end_time = str(ephem.Date(new_end_utc_time + tz * ephem.hour))
            # print 'new_end_time:', new_end_time

            # cut data
            eph_time = np.arange(start_time, end_time, int_time * ephem.second)
            transit_ind = np.searchsorted(eph_time, transit_time)
            # print 'transit_ind:', transit_ind
            start_ind = transit_ind - int(span * int_time)
            end_ind = transit_ind + int(span * int_time)
            eph_time = eph_time[start_ind:end_ind]
            ts = np.array([ephem.julian_date(d) for d in eph_time])
            nt = len(ts)

            lfreq, sfreq, efreq = mpiutil.split_local(nfreq)
            local_freq = range(sfreq, efreq)

            # get data to the local process
            local_data = dset[start_ind:end_ind, :, :, sfreq:efreq]

        if need_second_file == True:
            with h5py.File(input_file[1], 'r') as f:
                local_data = np.concatenate((local_data, f['data'][:nt-local_data.shape[0], :, :, sfreq:efreq]))

        assert nt == local_data.shape[0], 'Incorrect data shape'

        local_eigval = np.zeros((nt, nants, 2, lfreq), dtype=np.float64)
        local_gain = np.zeros((nt, nants, 2, lfreq), dtype=np.complex128)

        # global array to save eigval, gain
        if mpiutil.rank0:
            eigval = np.zeros((nt, nants, 2, nfreq), dtype=np.float64)
            gain = np.zeros((nt, nants, 2, nfreq), dtype=np.complex128)
        else:
            eigval = None
            gain = None

        # calibrator
        srclist, cutoff, catalogs = a.scripting.parse_srcs(calibrator, catalog)
        cat = a.src.get_catalog(srclist, cutoff, catalogs)
        assert(len(cat) == 1), 'Allow only one calibrator'
        s = cat.values()[0]
        if mpiutil.rank0:
            print 'Calibrating for source %s with' % calibrator,
            print 'strength', s._jys,
            print 'measured at', s.mfreq, 'GHz',
            print 'with index', s.index

        # array
        aa = tldishes.get_aa(1.0e-3 * freq) # use GHz
        # make all antennas point to the pointing direction
        for ai in aa:
            ai.set_pointing(az=az, alt=alt, twist=0)

        # construct visibility matrix for a single time, pol, freq
        Vmat = np.zeros((nants, nants), dtype=data_type)
        for ti, t in enumerate(ts):
            aa.set_jultime(t)
            s.compute(aa)
            # get fluxes vs. freq of the calibrator
            Sc = s.get_jys()
            # get the topocentric coordinate of the calibrator at the current time
            s_top = s.get_crds('top', ncrd=3)
            aa.sim_cache(cat.get_crds('eq', ncrd=3)) # for compute bm_response and sim
            for pol in [pols.index('xx'), pols.index('yy')]: # xx, yy
                aa.set_active_pol(pols[pol])
                for fi, freq_ind in enumerate(local_freq): # mpi among freq
                    for i, ai in enumerate(ants):
                        for j, aj in enumerate(ants):
                            uij = aa.gen_uvw(ai-1, aj-1, src='z').squeeze() # (rj - ri)/lambda
                            bmij = aa.bm_response(ai-1, aj-1).squeeze()
                            try:
                                ind = bls.index((ai, aj))
                                # Vmat[i, j] = local_data[ti, ind, pol, fi] / (Sc[freq_ind] * bmij[freq_ind] * np.exp(-2.0J * np.pi * np.dot(s_top, uij[:, freq_ind]))) # xx, yy
                                Vmat[i, j] = local_data[ti, ind, pol, fi] / (Sc[freq_ind] * bmij[freq_ind] * np.exp(2.0J * np.pi * np.dot(s_top, uij[:, freq_ind]))) # xx, yy
                            except ValueError:
                                ind = bls.index((aj, ai))
                                # Vmat[i, j] = np.conj(local_data[ti, ind, pol, fi] / (Sc[freq_ind] * bmij[freq_ind] * np.exp(-2.0J * np.pi * np.dot(s_top, uij[:, freq_ind])))) # xx, yy
                                Vmat[i, j] = np.conj(local_data[ti, ind, pol, fi] / (Sc[freq_ind] * bmij[freq_ind] * np.exp(2.0J * np.pi * np.dot(s_top, uij[:, freq_ind])))) # xx, yy

                    # Eigen decomposition
                    e, U = eigh(Vmat)
                    local_eigval[ti, :, pol, fi] = e[::-1] # descending order
                    # max eigen-val
                    lbd = e[-1] # lambda
                    # the gain vector for this freq
                    gvec = np.sqrt(lbd) * U[:, -1] # only eigen-vector corresponding to the maximum eigen-val
                    local_gain[ti, :, pol, fi] = gvec


        # Gather data in separate processes
        mpiutil.gather_local(eigval, local_eigval, (0, 0, 0, sfreq), root=0, comm=self.comm)
        mpiutil.gather_local(gain, local_gain, (0, 0, 0, sfreq), root=0, comm=self.comm)

        # save data
        if mpiutil.rank0:
            with h5py.File(output_file, 'w') as f:
                f.create_dataset('eigval', data=eigval)
                dset = f.create_dataset('gain', data=gain)
                dset.attrs['time'] = ts
                dset.attrs['ants'] = ants
                dset.attrs['pol'] = ['xx', 'yy']
                dset.attrs['freq'] = freq
                dset.attrs['history'] = self.history
