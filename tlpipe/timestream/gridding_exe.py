"""Module to gridding visibilites to uv plane."""

try:
    import cPickle as pickle
except ImportError:
    import pickle

import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy.linalg import eigh, inv
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
               'data_cal_stokes_file': 'data_cal_stokes.hdf5',
               'output_dir': './output/', # output directory
               'pol': 'I',
               'res': 1.0, # resolution, unit: wavelength
               'max_wl': 200.0, # max wavelength
               'sigma': 0.07,
              }
prefix = 'gr_'


pol_dict = {'I': 0, 'Q': 1, 'U': 2, 'V': 3}

def get_uvvec(s0_top, n_top):
    """Compute unit vector in u,v direction in topocentric coordinate.
    s0_top: unit vector of the phase center in topocentric coordinate;
    n_top: unit vector of the north celestial pole in topocentric coordinate.

    Return unit vector in u and v direction.
    """
    s0 = s0_top
    n = n_top
    s0x, s0y, s0z = s0[0], s0[1], s0[2]
    nx, ny, nz = n[0], n[1], n[2]
    # uvec is perpendicular to both s0 and n, and have ux >= 0 to point to East
    ux = 1.0 / np.sqrt(1.0 + ((nz*s0x - nx*s0z) / (ny*s0z - nz*s0y))**2 + ((ny*s0x - nx*s0y) / (nz*s0y - ny*s0z))**2)
    uy = ux * ((nz*s0x - nx*s0z) / (ny*s0z - nz*s0y))
    uz = ux * ((ny*s0x - nx*s0y) / (nz*s0y - ny*s0z))
    uvec = np.array([ux, uy, uz])
    # vvec is in the plane spanned by s0 and n, and have dot(n, vvec) > 0
    ns0 = np.dot(n, s0)
    l1 = 1.0 / np.sqrt(1.0 - ns0**2)
    l2 = - l1 * ns0
    vvec = l1*n + l2*s0

    return uvec, vvec


def conv_kernal(u, v, sigma, l0=0, m0=0):
    return np.exp(-2.0J * np.pi * (u * l0 + v * m0)) * np.exp(-0.5 * (2 * np.pi * sigma)**2 * (u**2 + v**2))

def conv_gauss(arr, rc, cc, sigma, val=1.0, l0=0, m0=0, pix=1, npix=4):
    for r in range(-npix, npix):
        for c in range(-npix, npix):
            arr[rc+r, cc+c] += val * conv_kernal(r*pix, c*pix, sigma, l0, m0)



class Gridding(object):
    """Gridding."""

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

        with h5py.File(self.params['data_cal_stokes_file'], 'r') as f:
            dset = f['data_cal_stokes']
            data_cal_stokes = dset[...]
            ants = dset.attrs['ants']
            ts = dset.attrs['ts']
            freq = dset.attrs['freq']
            bls = pickle.loads(dset.attrs['bls']) # as list
            az = dset.attrs['az']
            alt = dset.attrs['alt']

        npol = data_cal_stokes.shape[2]
        nt = len(ts)
        nfreq = len(freq)
        nants = len(ants)
        nbls = len(bls)


        assert self.comm.size <= nt, 'Can not have nprocs (%d) > nt (%d)' % (self.comm.size, nt)


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
        cal = 'tldishes'
        # calibrator
        srclist, cutoff, catalogs = a.scripting.parse_srcs(src, cat)
        cat = a.cal.get_catalog(cal, srclist, cutoff, catalogs)
        assert(len(cat) == 1), 'Allow only one calibrator'
        s = cat.values()[0]
        if mpiutil.rank0:
            print 'Calibrating for source with',
            print 'strength', s._jys,
            print 'measured at', s.mfreq, 'GHz',
            print 'with index', s.index


        # pointting vector in topocentric coord
        pt_top = a.coord.azalt2top((np.radians(az), np.radians(alt)))

        # array
        aa = a.cal.get_aa(cal, 1.0e-3 * freq) # use GHz
        for ti in mpiutil.mpirange(nt): # mpi among time
            t = ts[ti]
            aa.set_jultime(t)
            s.compute(aa)
            # get the topocentric coordinate of the calibrator at the current time
            s_top = s.get_crds('top', ncrd=3)
            # the north celestial pole
            NP = a.phs.RadioFixedBody(0.0, np.pi/2.0, name='north pole', epoch=str(aa.epoch))

            # get the topocentric coordinate of the north celestial pole at the current time
            NP.compute(aa)
            n_top = NP.get_crds('top', ncrd=3)

            # unit vector in u,v direction in topocentric coordinate at current time relative to the calibrator
            uvec, vvec = get_uvvec(s_top, n_top)

            # l,m of the pointing relative to phase center (the calibrator)
            l0 = np.dot(pt_top, uvec)
            m0 = np.dot(pt_top, vvec)

            for bl_ind in range(len(bls)):
                i, j = bls[bl_ind]
                if i == j:
                    continue
                us, vs, ws = aa.gen_uvw(i-1, j-1, src=s) # NOTE start from 0
                for fi, (u, v) in enumerate(zip(us.flat, vs.flat)):
                    val = data_cal_stokes[ti, bl_ind, 0, fi] # only I here
                    if np.isfinite(val):
                        up = np.int(u / res)
                        vp = np.int(v / res)
                        # uv_cov[center+vp, center+up] += 1.0
                        # uv_cov[center-vp, center-up] += 1.0 # append conjugate
                        # uv[center+vp, center+up] += val
                        # uv[center-vp, center-up] += np.conj(val)# append conjugate
                        conv_gauss(uv_cov, center+vp, center+up, sigma, 1.0, l0, m0, res)
                        conv_gauss(uv_cov, center-vp, center-up, sigma, 1.0, l0, m0, res) # append conjugate
                        conv_gauss(uv, center+vp, center+up, sigma, val, l0, m0, res)
                        conv_gauss(uv, center-vp, center-up, sigma, np.conj(val), l0, m0, res) # append conjugate


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
            with h5py.File(output_dir + 'uv_imag.hdf5', 'w') as f:
                f.create_dataset('uv_cov', data=uv_cov)
                f.create_dataset('uv', data=uv)
                f.create_dataset('uv_cov_fft', data=uv_cov_fft)
                f.create_dataset('uv_fft', data=uv_fft)
                f.create_dataset('uv_imag_fft', data=uv_imag_fft)
                f.attrs['max_wl'] = max_wl
                f.attrs['max_lm'] = max_lm
