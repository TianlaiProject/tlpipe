"""Module to do the map-making."""

import matplotlib.pyplot as plt

from caput import mpiutil
#from tlpipe.pipeline.pipeline import OneAndOne
from tlpipe.timestream import timestream_task
from tlpipe.utils.path_util import output_path
from tlpipe.map import algebra as al
from tlpipe.map.pointing import Pointing
from tlpipe.map.noise_model import Noise
import healpy as hp
import numpy as np
import scipy as sp
from scipy import linalg, special
from scipy.ndimage import gaussian_filter
import h5py
import sys
import gc

from constants import T_infinity, T_huge, T_large, T_medium, T_small, T_sys
from constants import f_medium, f_large

class DirtyMap_GBT(timestream_task.TimestreamTask):

    params_init = {
            #'ra_range' :  [0., 25.],
            #'ra_delta' :  0.5,
            #'dec_range' : [-4.0, 5.0],
            #'dec_delta' : 0.5,
            'field_centre' : (12., 0.,),
            'pixel_spacing' : 0.5,
            'map_shape'     : (10, 10),
            'interpolation' : 'linear',
            'tblock_len' : 100,
            'data_sets'  : 'vis',
            'corr' : 'auto',
            'deweight_time_slope' : False,
            'deweight_time_mean'  : True,
            'pol_select': (0, 2), # only useful for ts
            'freq_select' : (0, 4), 
            }

    prefix = 'dm_'

    def setup(self):

        params = self.params
        self.n_ra, self.n_dec = params['map_shape']
        self.map_shp = (self.n_ra, self.n_dec)
        self.spacing = params['pixel_spacing']
        self.dec_spacing = self.spacing
        # Negative sign because RA increases from right to left.
        self.ra_spacing = -self.spacing/sp.cos(params['field_centre'][1]*sp.pi/180.)


    def process(self, ts):

        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']

        ts.main_data_name = self.params['data_sets']
        n_time, n_freq, n_pol, n_bl = ts.main_data.shape
        print n_time, n_freq, n_pol, n_bl
        tblock_len = self.params['tblock_len']

        freq = ts['freq']
        freq_c = freq[n_freq//2]
        freq_d = freq[1] - freq[0]

        field_centre = self.params['field_centre']

        self.pol = ts['pol'][:]
        self.bl  = ts['blorder'][:]

        # for now, we assume no frequency corr, and thermal noise only.

        ra_spacing = self.ra_spacing
        dec_spacing = self.dec_spacing

        axis_names = ('bl', 'pol', 'freq', 'ra', 'dec')
        dirty_map = np.zeros((n_bl, n_pol, n_freq) +  self.map_shp)
        dirty_map = al.make_vect(dirty_map, axis_names=axis_names)
        dirty_map.set_axis_info('bl',   np.arange(n_bl)[n_bl//2],   1)
        dirty_map.set_axis_info('pol',  np.arange(n_pol)[n_pol//2], 1)
        dirty_map.set_axis_info('freq', freq_c, freq_d)
        dirty_map.set_axis_info('ra',   field_centre[0], self.ra_spacing)
        dirty_map.set_axis_info('dec',  field_centre[1], self.dec_spacing)
        self.dirty_map = dirty_map

        clean_map = np.zeros((n_bl, n_pol, n_freq) +  self.map_shp)
        clean_map = al.make_vect(clean_map, axis_names=axis_names)
        clean_map.copy_axis_info(dirty_map)
        self.clean_map = clean_map

        noise_diag = np.zeros((n_bl, n_pol, n_freq) +  self.map_shp)
        noise_diag = al.make_vect(noise_diag, axis_names=axis_names)
        noise_diag.copy_axis_info(dirty_map)
        self.noise_diag = noise_diag

        func = ts.freq_pol_and_bl_data_operate
        func(self.make_map_freq_pol_bl, full_data=True, copy_data=True, 
                show_progress=show_progress, 
                progress_step=progress_step, keep_dist_axis=False)

        return 0
    
    def make_map_freq_pol_bl(self, vis, vis_mask, li, gi, bl, ts, **kwargs):

        #print vis.shape, li, gi, bl
        fi, pi, bi = gi
        #print ts.keys()
        print li, gi

        tblock_len = self.params['tblock_len']
        n_time = vis.shape[0]

        cov_inv_block_shp = (self.n_ra, self.n_dec, self.n_ra, self.n_dec)
        cov_inv_block = np.zeros(cov_inv_block_shp, dtype=float)
        cov_inv_block.flat[::self.n_ra * self.n_dec + 1] += 1.0 / T_large ** 2.

        dirty_map = np.zeros(self.map_shp)

        cov_inv_block_tmp = np.zeros(cov_inv_block_shp, dtype=float)

        for st in range(0, n_time, tblock_len):
            et = st + tblock_len
            _vis = vis[st:et]
            _vis_mask = vis_mask[st:et]
            _time = ts['sec1970'][st:et]
            _ra   = ts['ra'][st:et]
            _dec  = ts['dec'][st:et]
            if _vis.dtype == np.complex:
                _vis = np.abs(_vis)

            # subtract mean, slope
            if self.params['deweight_time_slope']:
                n_poly = 2
            else:
                n_poly = 1
            polys = ortho_poly(_time, n_poly, ~_vis_mask, 0)
            amps = sp.sum(polys * _vis[None, :], -1)
            _vis_fit = np.sum(amps[:, None] * polys, 0)

            #xx  = np.arange(_time.shape[0])
            #plt.plot(xx, _vis)
            #plt.plot(xx, _vis_fit)
            #plt.show()

            # calculate the variances
            _vars = sp.sum(_vis ** 2.)
            _cont = sp.sum(~_vis_mask)
            if _cont != 0:
                _vars /= _cont
            else:
                _vars = T_infinity ** 2.

            _good  = ( _ra < max(self.dirty_map.get_axis('ra')))
            _good *= ( _ra > min(self.dirty_map.get_axis('ra')))
            _good *= (_dec < max(self.dirty_map.get_axis('dec')))
            _good *= (_dec > min(self.dirty_map.get_axis('dec')))
            _ra   = _ra[  _good]
            _dec  = _dec[ _good]
            _vis  = _vis[ _good]
            _time = _time[_good]

            if _vis.shape[0] < 5: continue

            _vis = al.make_vect(_vis[None, :], axis_names=('freq', 'time'))
            #_vis.set_axis_info('freq', bl[0], 0)
            #_vis.set_axis_info('time', _time[tbloc_len//2], _time[1]-_time[0])

            P = Pointing(("ra", "dec"), (_ra, _dec), self.dirty_map,
                    self.params['interpolation'])
            N = Noise(_vis, _time)
            #N.add_mask(~_vis_mask)
            thermal_noise = _vars
            N.add_thermal(thermal_noise)
            # Things to do along the time axis.  With no frequency
            # correlations, things are more stable.  Also, thermal noise
            # estimate will be high, so you need to deweight things extra.
            if self.params['deweight_time_mean']:
                N.deweight_time_mean(T_huge**2)
            if self.params['deweight_time_slope']:
                N.deweight_time_slope(T_huge**2)
            N.finalize(frequency_correlations=False, preserve_matrices=False)

            _vis_weighted = N.weight_time_stream(_vis)
            dirty_map += P.apply_to_time_axis(_vis_weighted)[0,...]

            cov_inv_block_tmp *= 0.
            P.noise_channel_to_map(N, 0, cov_inv_block_tmp)
            cov_inv_block += cov_inv_block_tmp

        self.dirty_map[bi, pi, fi, ...] = dirty_map

        del cov_inv_block_tmp

        dirty_map.shape = (np.prod(self.map_shp), )
        cov_inv_block.shape = (self.n_ra * self.n_dec, self.n_ra * self.n_dec)
        cov_inv_diag, Rot = linalg.eigh(cov_inv_block, overwrite_a=True)
        map_rotated = sp.dot(Rot.T, dirty_map)
        bad_modes = cov_inv_diag < 1.e-5 * cov_inv_diag.max()
        map_rotated[bad_modes] = 0.
        cov_inv_diag[bad_modes] = 1.
        map_rotated /= cov_inv_diag
        clean_map = sp.dot(Rot, map_rotated)
        clean_map.shape = self.map_shp
        self.clean_map[bi, pi, fi, ...] = clean_map

        noise_diag = 1./cov_inv_diag
        noise_diag[bad_modes] = 0.

        tmp_mat = Rot * noise_diag
        for jj in range(np.prod(self.map_shp)):
            noise_diag[jj] = sp.dot(tmp_mat[jj, :], Rot[jj, :])
        noise_diag.shape = self.map_shp
        self.noise_diag[bi, pi, fi, ...] = noise_diag


        del cov_inv_block, dirty_map, clean_map, noise_diag
        gc.collect()

    def write_output(self, output):

        #dirty_map, noise_inv_diag, clean_map = output
        dirty_map = self.dirty_map
        clean_map = self.clean_map
        noise_diag = self.noise_diag

        suffix = '_%s.h5'%self.params['data_sets']
        output_file = self.output_files[0]
        output_file = output_path(output_file + suffix, 
                relative = not output_file.startswith('/'))

        with h5py.File(output_file, 'w') as f:

            al.save_h5(f, 'dirty_map',  dirty_map)
            al.save_h5(f, 'clean_map',  clean_map)
            al.save_h5(f, 'noise_diag', noise_diag)

            f['pol'] = self.pol
            f['bl']  = self.bl

            #f['dirty_map'] = dirty_map
            #f['clean_map'] = clean_map
            #f['noise_diag']= noise_diag
            #f['ra']        = dirty_map.get_axis('ra')
            #f['dec']       = dirty_map.get_axis('dec')
            #f['freq']      = dirty_map.get_axis('freq')

    def finish(self):
        if mpiutil.rank0:
            print 'Finishing MapMaking.'

class DirtyMap_Flat(timestream_task.TimestreamTask):
    """Class to do the map-making."""

    params_init = {
            'ra_range' :  [0., 25.],
            'ra_delta' :  0.5,
            'dec_range' : [-4.0, 5.0],
            'dec_delta' : 0.5,
            'tblock_len' : 100,
            'data_sets'  : 'vis',
            'corr' : 'auto',
            }

    prefix = 'dm_'

    def setup(self):
        if mpiutil.rank0:
            print 'Setting up MapMaking.'

        self.beam_size = 1.
        ra_min,  ra_max  = self.params['ra_range']
        dec_min, dec_max = self.params['dec_range']
        ra_delta  = self.params['ra_delta']
        dec_delta = self.params['dec_delta']
        self.ra_axis  = np.arange(ra_min, ra_max, ra_delta)
        self.dec_axis = np.arange(dec_min, dec_max, dec_delta)
        self.map_shp = [ self.ra_axis.shape[0]-1, self.dec_axis.shape[0]-1 ]


    def process(self, ts):

        ts.main_data_name = self.params['data_sets']
        vis = ts.main_data

        ntime, nfreq, npol, nbl = vis.shape
        tblock_len = self.params['tblock_len']
        print nbl
        
        ra_bin = self.ra_axis
        dec_bin = self.dec_axis

        noise_inv_diag = np.zeros([nbl, nfreq, np.prod(self.map_shp)])
        dirty_map = np.zeros([nbl, nfreq, np.prod(self.map_shp)])
        clean_map = np.zeros([nbl, nfreq, np.prod(self.map_shp)])
        #pixle_bin = np.arange(self.npixl + 1.) - 0.5

        print ts['pol'][:]
        if ts['pol'][0] == 'hh' and ts['pol'][1] == 'vv':

            if mpiutil.rank0:
                print 'Rotate HH VV to I'
            vis[:, :, 0, :] = np.sum(vis[:, :, :2, :], axis=2)

        for bi in range(nbl):
            for fi in range(nfreq)[:1]:
                noise_inv = np.zeros((np.prod(self.map_shp),)*2)
                print bi, fi
                for st in range(0, ntime, tblock_len):
                    et = st + tblock_len
                    _vis = vis[:, fi, 0, bi][st:et]
                    if _vis.dtype == np.complex:
                        _vis = np.abs(_vis)
                    _ra  = ts['ra'][st:et]
                    _dec = ts['dec'][st:et]

                    #_vis -= np.mean(_vis, axis=0)[None, ...]

                    var = np.var(_vis)
                    var[var==0] = np.inf
                    var = 1./ var
                    n = np.mat(np.eye(_vis.shape[0]) * var)

                    p = np.mat(self.est_pointing_matrix(_ra, _dec))

                    #_ni = np.histogram2d(_ra, _dec, bins=[ra_bin, dec_bin])[0]
                    #_dm = np.histogram2d(_ra, _dec, bins=[ra_bin, dec_bin],
                    #        weights=_vis[:, fi, bi])[0]
                    #noise_inv[bi, fi] += _ni * wet[fi, bi]
                    #dirty_map[bi, fi] += _dm * wet[fi, bi]

                    d = np.mat(_vis[:, None])
                    #d = np.mat(np.ones_like(_vis[:, None]))
                    pn = p.T * n
                    dirty_map[bi, fi, :][:, None] += (pn * d)
                    noise_inv += pn * p

                    del n, pn, d, p
                    gc.collect()

                #bad = np.diagonal(noise_inv) < 1.e-30
                #bad = bad[:, None] * np.eye(noise_inv.shape[0])
                #bad = bad.astype('bool')
                #noise_inv[bad] = np.inf
                #noise_inv_inv = np.linalg.inv(noise_inv)
                #u, s, v = np.linalg.svd(noise_inv)
                #s[s==0] = np.inf
                #s = 1./s
                #noise_inv_inv = np.dot(np.dot(v.T, s), u.T)
                noise_inv_inv = np.linalg.pinv(noise_inv)

                clean_map[bi, fi] = np.sum( noise_inv_inv\
                        * dirty_map[bi, fi, :][None, :], axis=0)
                noise_inv_diag[bi, fi] = np.diag(noise_inv)
                del noise_inv_inv, noise_inv #, bad
                gc.collect()


        dirty_map.shape      = [nbl, nfreq,] + self.map_shp
        clean_map.shape      = [nbl, nfreq,] + self.map_shp
        noise_inv_diag.shape = [nbl, nfreq,] + self.map_shp
        return dirty_map, noise_inv_diag, clean_map

    def write_output(self, output):

        dirty_map, noise_inv_diag, clean_map = output
        print dirty_map.shape, noise_inv_diag.shape

        suffix = '_%s.h5'%self.params['data_sets']
        output_file = self.output_files[0]
        output_file = output_path(output_file + suffix, 
                relative = not output_file.startswith('/'))

        with h5py.File(output_file, 'w') as f:

            f['dirty_map'] = dirty_map
            f['clean_map'] = clean_map
            f['noise_inv_diag'] = noise_inv_diag
            f['ra_axis']   = self.ra_axis
            f['dec_axis']  = self.dec_axis

    def finish(self):
        if mpiutil.rank0:
            print 'Finishing MapMaking.'

    def est_pointing_matrix(self, ra, dec):

        block_length = ra.shape[0]
        map_tmp_shp = self.map_shp

        ra_idx  = np.digitize(ra, self.ra_axis)
        dec_idx = np.digitize(dec, self.dec_axis)

        map_idx  = ra_idx * map_tmp_shp[1] + dec_idx
        map_idx += np.arange(block_length) * np.prod(map_tmp_shp)

        p = np.zeros(block_length * np.prod(map_tmp_shp))
        p[map_idx] = 1.

        #p.shape = [block_length, ] + map_tmp_shp
        #ra_sigma  = self.beam_size / self.params['ra_delta']
        #dec_sigma = self.beam_size / self.params['dec_delta']
        #p = gaussian_filter(p, sigma=(0, ra_sigma, dec_sigma), mode='constant', 
        #        truncate=2.0)
        #p /= p.max()
        #p[p<0.1] = 0
        #np.save('/data/users/ycli/meerkat/p.npy', p)

        p.shape = (block_length, np.prod(map_tmp_shp))

        return p

class DirtyMap_Healpix(timestream_task.TimestreamTask):
    """Class to do the map-making."""

    params_init = {
            'nside' : 64,
            'tblock_len' : 100,
            'data_sets'  : 'vis',
            }

    prefix = 'dm_'

    def setup(self):
        if mpiutil.rank0:
            print 'Setting up MapMaking.'

        self.nside = self.params['nside']
        self.npixl = hp.nside2npix(self.nside)

    def process(self, ts):

        ts.main_data_name = self.params['data_sets']
        vis = ts.main_data

        ntime, nfreq, npol, nbl = vis.shape
        
        dirty_map = np.zeros([nbl, nfreq, self.npixl])
        noise_inv = np.zeros([nbl, nfreq, self.npixl])
        pixle_bin = np.arange(self.npixl + 1.) - 0.5

        print ts['pol'][:]
        if ts['pol'][0] == 'hh' and ts['pol'][1] == 'vv':

            if mpiutil.rank0:
                print 'Rotate HH VV to I'
            vis[:, :, 0, :] = np.sum(vis[:, :, :2, :], axis=2)

        tblock_len = self.params['tblock_len']
        st = 0
        for st in range(0, ntime, tblock_len):
            et = st + tblock_len
            _vis = vis[:, :, 0, :][st:et,...]
            if _vis.dtype == np.complex:
                _vis = np.abs(_vis)
            _ra  = ts['ra'][st:et]
            _dec = ts['dec'][st:et]

            #print _ra.shape, _dec.shape
            _pix = hp.ang2pix(self.nside, _ra, _dec, lonlat=True)

            #_vis -= np.mean(_vis, axis=0)[None, ...]

            wet = 1. / np.var(_vis, axis=0)
            #wet[:] = 1.

            for bi in range(nbl):
                for fi in range(nfreq):
                    _ni = np.histogram(_pix, bins=pixle_bin)[0]
                    _dm = np.histogram(_pix, bins=pixle_bin, weights=_vis[:, fi, bi])[0]
                    noise_inv[bi, fi] += _ni * wet[fi, bi]
                    dirty_map[bi, fi] += _dm * wet[fi, bi]

        return dirty_map, noise_inv

    def write_output(self, output):

        dirty_map, noise_inv = output
        print dirty_map.shape, noise_inv.shape

        suffix = '_%s.h5'%self.params['data_sets']
        output_file = self.output_files[0]
        output_file = output_path(output_file + suffix, 
                relative = not output_file.startswith('/'))

        with h5py.File(output_file, 'w') as f:

            f['dirty_map'] = dirty_map
            f['noise_inv'] = noise_inv

    def finish(self):
        if mpiutil.rank0:
            print 'Finishing MapMaking.'



def ortho_poly(x, n, window=1., axis=-1):
    """Generate orthonormal basis polynomials.

    Generate the first `n` orthonormal basis polynomials over the given domain
    and for the given window using the Gram-Schmidt process.
    
    Parameters
    ----------
    x : 1D array length m
        Functional domain.
    n : integer
        number of polynomials to generate. `n` - 1 is the maximum order of the
        polynomials.
    window : 1D array length m
        Window (weight) function for which the polynomials are orthogonal.

    Returns
    -------
    polys : n by m array
        The n polynomial basis functions. Normalization is such that
        np.sum(polys[i,:] * window * polys[j,:]) = delta_{ij}
    """
    
    if np.any(window < 0):
        raise ValueError("Window function must never be negative.")
    # Check scipy versions. If there is a stable polynomial package, use it.
    s_ver = sp.__version__.split('.')
    major = int(s_ver[0])
    minor = int(s_ver[1])
    if major <= 0 and minor < 8:
        new_sp = False
        if n > 20:
            raise NotImplementedError("High order polynomials unstable.")
    else:
        new_sp = True
    # Get the broadcasted shape of `x` and `window`.
    # The following is the only way I know how to get the broadcast shape of
    # x and window.
    # Turns out I could use np.broadcast here.  Fix this later.
    shape = np.broadcast(x, window).shape
    m = shape[axis]
    # Construct a slice tuple for up broadcasting arrays.
    upbroad = [slice(sys.maxsize)] * len(shape)
    upbroad[axis] = None
    upbroad = tuple(upbroad)
    # Allocate memory for output.
    polys = np.empty((n,) + shape, dtype=float)
    # For stability, rescale the domain.
    x_range = np.amax(x, axis) - np.amin(x, axis)
    x_mid = (np.amax(x, axis) + np.amin(x, axis)) / 2.
    x = (x - x_mid[upbroad]) / x_range[upbroad] * 2
    # Reshape x to be the final shape.
    x = np.zeros(shape, dtype=float) + x
    # Now loop through the polynomials and construct them.
    # This array will be the starting polynomial, before orthogonalization
    # (only used for earlier versions of scipy).
    if not new_sp:
        basic_poly = np.ones(shape, dtype=float) / np.sqrt(m)
    for ii in range(n):
        # Start with the basic polynomial.
        # If we have an up-to-date scipy, start with nearly orthogonal
        # functions.  Otherwise, just start with the next polynomial.
        if not new_sp:
            new_poly = basic_poly.copy()
        else:
            new_poly = special.eval_legendre(ii, x)
        # Orthogonalize against all lower order polynomials.
        for jj in range(ii):
            new_poly -= (np.sum(new_poly * window * polys[jj,:], axis)[upbroad]
                         * polys[jj,:])
        # Normalize, accounting for possibility that all data is masked. 
        norm = np.array(np.sqrt(np.sum(new_poly**2 * window, axis)))
        if norm.shape == ():
            if norm == 0:
                bad_inds = np.array(True)
                norm = np.array(1.)
            else:
                bad_inds = np.array(False)
        else:
            bad_inds = norm == 0
            norm[bad_inds] = 1.
        new_poly /= norm[upbroad]
        #new_poly[bad_inds[None,]] = 0
        new_poly *= ~bad_inds[upbroad]
        # Copy into output.
        polys[ii,:] = new_poly
        # Increment the base polynomial with another power of the domain for
        # the next iteration.
        if not new_sp:
            basic_poly *= x
    return polys

