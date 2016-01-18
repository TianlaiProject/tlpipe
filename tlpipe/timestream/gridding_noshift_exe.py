"""Module to gridding visibilities to uv plane with no shift term in the convolution kernel."""

try:
    import cPickle as pickle
except ImportError:
    import pickle

import os
import numpy as np
import aipy as a
import h5py

from tlpipe.utils import mpiutil
from tlpipe.core.base_exe import Base
from tlpipe.core import tldishes
from tlpipe.utils.path_util import input_path, output_path


# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
params_init = {
               'nprocs': mpiutil.size, # number of processes to run this module
               'aprocs': range(mpiutil.size), # list of active process rank no.
               'input_file': 'data_cal_stokes.hdf5',
               'output_file': 'uv_imag_noshift.hdf5',
               'cut': [None, None],
               'pol': 'I',
               'res': 1.0, # resolution, unit: wavelength
               'max_wl': 200.0, # max wavelength
               'sigma': 0.07,
               'extra_history': '',
              }
prefix = 'ngr_'


pol_dict = {'I': 0, 'Q': 1, 'U': 2, 'V': 3}


def conv_kernal(u, v, sigma):
    # return np.exp(-0.5 * (2 * np.pi * sigma)**2 * (u**2 + v**2))
    return np.exp(-(2 * np.pi * sigma)**2 * (u**2 + v**2))

def conv_gauss(arr, c, vp, up, sigma, val=1.0, pix=1, npix=4):
    for ri in range(-npix, npix):
        for ci in range(-npix, npix):
            tmp = val * conv_kernal(ri*pix, ci*pix, sigma)
            # tmp = val * conv_kernal((vp+ri)*pix, (up+ci)*pix, sigma)
            arr[c+(vp+ri), c+(up+ci)] += tmp
            arr[c-(vp+ri), c-(up+ci)] += np.conj(tmp) # append conjugate



class Gridding(Base):
    """Gridding with no shift term in the convolution kernel."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(Gridding, self).__init__(parameter_file_or_dict, params_init, prefix, feedback)

    def execute(self):

        input_file = input_path(self.params['input_file'])
        output_file = output_path(self.params['output_file'])
        cut = self.params['cut']

        with h5py.File(input_file, 'r') as f:
            dset = f['data']
            # data_cal_stokes = dset[...]
            ants = dset.attrs['ants']
            ts = f['time'][...]
            freq = dset.attrs['freq']
            # az = np.radians(dset.attrs['az_alt'][0][0])
            # alt = np.radians(dset.attrs['az_alt'][0][1])

            # cut head and tail
            nt = len(ts)
            t_inds = range(nt)
            if cut[0] is not None and cut[1] is not None:
                t_inds = t_inds[:int(cut[0] * nt)] + t_inds[-int(cut[1] * nt):]
            elif cut[0] is not None:
                t_inds = t_inds[:int(cut[0] * nt)]
            elif cut[1] is not None:
                t_inds[-int(cut[1] * nt):]

            npol = dset.shape[2]
            nt = len(ts)
            nfreq = len(freq)
            nants = len(ants)
            bls = [(ants[i], ants[j]) for i in range(nants) for j in range(i, nants)] # start from 1
            nbls = len(bls)

            lt_inds = mpiutil.mpilist(t_inds)
            local_data = dset[lt_inds, :, :, :] # data section used only in this process

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
        for ti, t_ind in enumerate(lt_inds): # mpi among time
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
                        conv_gauss(uv_cov, center, vp, up, sigma, 1.0, res)
                        conv_gauss(uv, center, vp, up, sigma, val, res)


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
            with h5py.File(output_file, 'w') as f:
                f.create_dataset('uv_cov', data=uv_cov)
                f.create_dataset('uv', data=uv)
                f.create_dataset('uv_cov_fft', data=uv_cov_fft)
                f.create_dataset('uv_fft', data=uv_fft)
                f.create_dataset('uv_imag_fft', data=uv_imag_fft)
                f.attrs['max_wl'] = max_wl
                f.attrs['max_lm'] = max_lm
