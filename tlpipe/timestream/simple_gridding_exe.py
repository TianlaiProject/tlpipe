"""Module to gridding visibilities to uv plane. This module does not do the gauss convolution when gridding."""

try:
    import cPickle as pickle
except ImportError:
    import pickle

import os
import numpy as np
import aipy as a
import h5py

from tlpipe.kiyopy import parse_ini
from tlpipe.utils import mpiutil
from tlpipe.core import tldishes


# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
params_init = {
               'nprocs': mpiutil.size, # number of processes to run this module
               'aprocs': range(mpiutil.size), # list of active process rank no.
               'data_file': 'data_cal_stokes.hdf5',
               'pol': 'I',
               'res': 1.0, # resolution, unit: wavelength
               'max_wl': 200.0, # max wavelength
               'sigma': 0.07,
              }
prefix = 'sgr_'


pol_dict = {'I': 0, 'Q': 1, 'U': 2, 'V': 3}


class Gridding(object):
    """Simple gridding without Gauss convolution."""

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

        output_dir = os.environ['TL_OUTPUT']
        data_file = self.params['data_file']

        with h5py.File(data_file, 'r') as f:
            dset = f['data']
            # data_cal_stokes = dset[...]
            ants = dset.attrs['ants']
            ts = f['time'][...]
            freq = dset.attrs['freq']
            az = np.radians(dset.attrs['az_alt'][0][0])
            alt = np.radians(dset.attrs['az_alt'][0][1])

            npol = dset.shape[2]
            nt = len(ts)
            nfreq = len(freq)
            nants = len(ants)
            bls = [(ants[i], ants[j]) for i in range(nants) for j in range(i, nants)] # start from 1
            nbls = len(bls)

            lt, st, et = mpiutil.split_local(nt)
            local_data = dset[st:et] # data section used only in this process


        res = self.params['res']
        max_wl = self.params['max_wl']
        max_lm = 0.5 * 1.0 / res
        size = np.int(2 * max_wl / res) + 1
        center = np.int(max_wl / res) # the central pixel
        sigma = self.params['sigma']

        uv = np.zeros((size, size), dtype=np.complex128)
        uv_cov = np.zeros((size, size), dtype=np.complex128)

        src = 'cas'
        cat = 'misc'
        # calibrator
        srclist, cutoff, catalogs = a.scripting.parse_srcs(src, cat)
        cat = a.src.get_catalog(srclist, cutoff, catalogs)
        assert(len(cat) == 1), 'Allow only one calibrator'
        s = cat.values()[0]
        if mpiutil.rank0:
            print 'Calibrating for source with',
            print 'strength', s._jys,
            print 'measured at', s.mfreq, 'GHz',
            print 'with index', s.index


        # array
        aa = tldishes.get_aa(1.0e-3 * freq) # use GHz
        for ti, t_ind in enumerate(range(st, et)): # mpi among time
            t = ts[t_ind]
            aa.set_jultime(t)
            s.compute(aa)

            for bl_ind in range(nbls):
                i, j = bls[bl_ind]
                if i == j:
                    continue
                us, vs, ws = aa.gen_uvw(i-1, j-1, src=s) # NOTE start from 0
                for fi, (u, v) in enumerate(zip(us.flat, vs.flat)):
                    val = local_data[ti, bl_ind, 0, fi] # only I here
                    if np.isfinite(val):
                        up = np.int(u / res)
                        vp = np.int(v / res)
                        uv_cov[center+vp, center+up] += 1.0
                        uv_cov[center-vp, center-up] += 1.0 # append conjugate
                        uv[center+vp, center+up] += val
                        uv[center-vp, center-up] += np.conj(val)# append conjugate


        # Reduce data in separate processes
        if self.comm is not None and self.comm.size > 1: # Reduce only when there are multiple processes
            if mpiutil.rank0:
                self.comm.Reduce(mpiutil.IN_PLACE, uv_cov, op=mpiutil.SUM, root=0)
            else:
                self.comm.Reduce(uv_cov, uv_cov, op=mpiutil.SUM, root=0)
            if mpiutil.rank0:
                self.comm.Reduce(mpiutil.IN_PLACE, uv, op=mpiutil.SUM, root=0)
            else:
                self.comm.Reduce(uv, uv, op=mpiutil.SUM, root=0)


        if mpiutil.rank0:
            uv_cov_fft = np.fft.ifft2(np.fft.ifftshift(uv_cov))
            uv_cov_fft = np.fft.ifftshift(uv_cov_fft)
            uv_fft = np.fft.ifft2(np.fft.ifftshift(uv))
            uv_fft = np.fft.ifftshift(uv_fft)
            uv_imag_fft = np.fft.ifft2(np.fft.ifftshift(1.0J * uv.imag))
            uv_imag_fft = np.fft.ifftshift(uv_imag_fft)

            # save data
            with h5py.File(output_dir + 'uv_imag_noconv.hdf5', 'w') as f:
                f.create_dataset('uv_cov', data=uv_cov)
                f.create_dataset('uv', data=uv)
                f.create_dataset('uv_cov_fft', data=uv_cov_fft)
                f.create_dataset('uv_fft', data=uv_fft)
                f.create_dataset('uv_imag_fft', data=uv_imag_fft)
                f.attrs['max_wl'] = max_wl
                f.attrs['max_lm'] = max_lm
