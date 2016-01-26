"""Module to gridding visibilities to uv plane. This module does not do the gauss convolution when gridding."""

try:
    import cPickle as pickle
except ImportError:
    import pickle

import os
import numpy as np
import ephem
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
               'output_file': 'uv_imag_noconv.hdf5',
               'cut': [None, None],
               'pol': 'I',
               'res': 1.0, # resolution, unit: wavelength
               'max_wl': 200.0, # max wavelength
               'sigma': 0.07,
               'phase_center': 'cas',
               'catalog': 'misc,helm,nvss',
               'extra_history': '',
              }
prefix = 'sgr_'


class Gridding(Base):
    """Simple gridding without Gauss convolution."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(Gridding, self).__init__(parameter_file_or_dict, params_init, prefix, feedback)

    def execute(self):

        input_file = input_path(self.params['input_file'])
        output_file = output_path(self.params['output_file'])
        cut = self.params['cut']
        pol = self.params['pol']
        phase_center = self.params['phase_center']
        catalog = self.params['catalog']

        with h5py.File(input_file, 'r') as f:
            dset = f['data']
            # data_cal_stokes = dset[...]
            ants = dset.attrs['ants']
            ts = f['time'][...]
            freq = dset.attrs['freq']
            pols = dset.attrs['pol'].tolist()
            assert pol in pols, 'Required pol %s is not in this data set with pols %s' % (pol, pols)
            # az = np.radians(dset.attrs['az_alt'][0][0])
            # alt = np.radians(dset.attrs['az_alt'][0][1])
            start_time = dset.attrs['start_time']
            history = dset.attrs['history']

            # cut head and tail
            nt = len(ts)
            t_inds = range(nt)
            if cut[0] is not None and cut[1] is not None:
                t_inds = t_inds[:int(cut[0] * nt)] + t_inds[-int(cut[1] * nt):]
            elif cut[0] is not None:
                t_inds = t_inds[:int(cut[0] * nt)]
            elif cut[1] is not None:
                t_inds = t_inds[-int(cut[1] * nt):]

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

        # phase center
        srclist, cutoff, catalogs = a.scripting.parse_srcs(phase_center, catalog)
        cat = a.src.get_catalog(srclist, cutoff, catalogs)
        assert(len(cat) == 1), 'Allow only one phase center'
        s = cat.values()[0]
        if mpiutil.rank0:
            print 'Imaging relative to phase center %s.' % phase_center

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
                    val = local_data[ti, bl_ind, pols.index(pol), fi]
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

            cen = ephem.Equatorial(s.ra, s.dec, epoch=aa.epoch)
            # We precess the coordinates of the center of the image here to
            # J2000, just to have a well-defined epoch for them.  For image coords to
            # be accurately reconstructed, precession needs to be applied per pixel
            # and not just per phase-center because ra/dec axes aren't necessarily
            # aligned between epochs.  When reading these images, to be 100% accurate,
            # one should precess the ra/dec coordinates back to the date of the
            # observation, infer the coordinates of all the pixels, and then
            # precess the coordinates for each pixel independently.
            cen = ephem.Equatorial(cen, epoch=ephem.J2000)

            # save data
            with h5py.File(output_file, 'w') as f:
                f.create_dataset('uv_cov', data=uv_cov)
                f.create_dataset('uv', data=uv)
                f.create_dataset('uv_cov_fft', data=uv_cov_fft)
                f.create_dataset('uv_fft', data=uv_fft)
                f.create_dataset('uv_imag_fft', data=uv_imag_fft)
                f.attrs['pol'] = pol
                f.attrs['max_wl'] = max_wl
                f.attrs['max_lm'] = max_lm
                f.attrs['src_name'] = s.src_name
                f.attrs['obs_date'] = start_time
                f.attrs['ra'] = np.degrees(cen.ra)
                f.attrs['dec'] = np.degrees(cen.dec)
                f.attrs['epoch'] = 'J2000'
                f.attrs['d_ra'] = np.degrees(2.0 * max_lm / size)
                f.attrs['d_dec'] = np.degrees(2.0 * max_lm / size)
                f.attrs['freq'] = freq[nfreq/2]
                f.attrs['history'] = history + self.history
