"""Module to do the map-making."""

import matplotlib.pyplot as plt

from caput import mpiutil
#from tlpipe.pipeline.pipeline import OneAndOne
from tlpipe.timestream import timestream_task
from tlpipe.utils.path_util import output_path
from tlpipe.map import algebra as al
from tlpipe.map.pointing import Pointing
from tlpipe.map.noise_model import Noise
from tlpipe.map import mapbase
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

class DirtyMap_GBT(mapbase.MapBase, timestream_task.TimestreamTask):

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

            'save_cov' : False,
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

        axis_names = ('ra', 'dec')
        map_tmp = np.zeros(self.map_shp)
        map_tmp = al.make_vect(map_tmp, axis_names=axis_names)
        map_tmp.set_axis_info('ra',   params['field_centre'][0], self.ra_spacing)
        map_tmp.set_axis_info('dec',  params['field_centre'][1], self.dec_spacing)
        self.map_tmp = map_tmp


    def process(self, ts):

        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']

        self.init_output()

        func = self.init_ps_datasets(ts)

        if not func is None:
            func(self.make_map, full_data=True, copy_data=True, 
                    show_progress=show_progress, 
                    progress_step=progress_step, keep_dist_axis=False)

    def init_output(self):

        suffix = '_%s.h5'%self.params['data_sets']
        output_file = self.output_files[0]
        output_file = output_path(output_file + suffix, 
                relative = not output_file.startswith('/'))
        self.allocate_output(output_file, 'w')

    def init_ps_datasets(self, ts):

        ts.main_data_name = self.params['data_sets']
        n_time, n_freq, n_pol, n_bl = ts.main_data.shape
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
        dirty_map_tmp = np.zeros((n_bl, n_pol, n_freq) +  self.map_shp)
        dirty_map_tmp = al.make_vect(dirty_map_tmp, axis_names=axis_names)
        dirty_map_tmp.set_axis_info('bl',   np.arange(n_bl)[n_bl//2],   1)
        dirty_map_tmp.set_axis_info('pol',  np.arange(n_pol)[n_pol//2], 1)
        dirty_map_tmp.set_axis_info('freq', freq_c, freq_d)
        dirty_map_tmp.set_axis_info('ra',   field_centre[0], self.ra_spacing)
        dirty_map_tmp.set_axis_info('dec',  field_centre[1], self.dec_spacing)

        self.create_dataset_like('dirty_map',  dirty_map_tmp)

        self.create_dataset_like('clean_map',  dirty_map_tmp)

        self.create_dataset_like('noise_diag', dirty_map_tmp)

        self.df['mask'] = np.zeros([n_bl, n_pol, n_freq])

        if self.params['save_cov']:
            axis_names = ('bl', 'pol', 'freq', 'ra', 'dec', 'ra', 'dec')
            cov_tmp = np.zeros((n_bl, n_pol, n_freq) +  self.map_shp + self.map_shp)
            cov_tmp = al.make_vect(cov_tmp, axis_names=axis_names)
            cov_tmp.set_axis_info('bl',   np.arange(n_bl)[n_bl//2],   1)
            cov_tmp.set_axis_info('pol',  np.arange(n_pol)[n_pol//2], 1)
            cov_tmp.set_axis_info('freq', freq_c, freq_d)
            cov_tmp.set_axis_info('ra',   field_centre[0], self.ra_spacing)
            cov_tmp.set_axis_info('dec',  field_centre[1], self.dec_spacing)
            self.create_dataset_like('cov_inv', cov_tmp)

        self.df['pol'] = self.pol
        self.df['bl']  = self.bl

        func = ts.freq_pol_and_bl_data_operate

        return func

    def make_map(self, vis, vis_mask, li, gi, bl, ts, **kwargs):

        #print vis.shape, li, gi, bl
        #fi, pi, bi = gi
        if not isinstance(li, tuple):
            li = (li, )
        if not isinstance(gi, tuple):
            gi = (gi, )
        idx = gi[::-1]
        #print ts.keys()
        print "RANK%03d:"%mpiutil.rank + \
                " Local  (" + ("%03d, "*len(li))%li + ")," +\
                " Global (" + ("%03d, "*len(gi))%gi + ")"
        if np.all(vis_mask):
            print "\t All masked, continue"
            self.df['mask'][idx] = 1
            return

        vis_shp = vis.shape
        ra   = ts['ra'][:]
        dec  = ts['dec'][:]
        if len(vis_shp) == 1:
            vis      = vis[:, None]
            vis_mask = vis_mask[:, None]
            vis_shp  = vis.shape
        else:
            #print vis_shp
            #print ra.shape
            _bc = [None, ] * len(vis_shp)
            _bc[0] = slice(None)
            if len(ra.shape) == 2:
                _bc[-1] = slice(None)
            ra  = ra[_bc]  * np.ones(vis_shp)
            dec = dec[_bc] * np.ones(vis_shp)

            vis_shp        = (vis_shp[0], np.prod(vis_shp[1:]))
            vis.shape      = vis_shp
            vis_mask.shape = vis_shp
            ra.shape       = vis_shp
            dec.shape      = vis_shp

        tblock_len = self.params['tblock_len']
        if self.params['deweight_time_slope']:
            n_poly = 2
        else:
            n_poly = 1

        time = ts['sec1970'][:]
        dirty_map, cov_inv_block = make_dirtymap(vis, vis_mask, time, ra, dec, 
                self.map_tmp, tblock_len, n_poly, self.params['interpolation'])

        self.df['dirty_map' ][idx + (slice(None), )]  = dirty_map

        if self.params['save_cov']:
            self.df['cov_inv'][idx + (slice(None), )] = cov_inv_block

        clean_map, noise_diag = make_cleanmap(dirty_map, cov_inv_block)

        self.df['clean_map' ][idx + (slice(None), )] = clean_map
        self.df['noise_diag'][idx + (slice(None), )] = noise_diag

        del cov_inv_block, dirty_map, clean_map, noise_diag
        gc.collect()

    def finish(self):
        if mpiutil.rank0:
            print 'Finishing MapMaking.'

        mpiutil.barrier()

#def make_dirtymap(vis, vis_mask, ts, map_tmp, tblock_len, n_poly=1, 
def make_dirtymap(vis, vis_mask, time, ra, dec, map_tmp, tblock_len, n_poly=1, 
        interpolation = 'linear'):

    dirty_map = al.zeros_like(map_tmp)
    cov_inv = np.zeros(map_tmp.shape * 2, dtype=float)

    n_time = vis.shape[0]
    n_extr = vis.shape[1]
    if tblock_len is None:
        tblock_len = n_time

    for st in range(0, n_time, tblock_len):
        et = st + tblock_len
        _time = time[st:et] #ts['sec1970'][st:et]
        #_ra   = ts['ra'][st:et, 0]
        #_dec  = ts['dec'][st:et, 0]
        for ii in range(n_extr):
            _ra   = ra[st:et, ii]  #ts['ra'][st:et, 0]
            _dec  = dec[st:et, ii] #ts['dec'][st:et, 0]
            _vis  = vis[st:et, ii]
            _vis_mask = vis_mask[st:et, ii]
            if _vis.dtype == np.complex:
                _vis = np.abs(_vis)
            _dm, _ci = timestream2map(_vis, _vis_mask, _time, _ra, _dec, 
                                      map_tmp, n_poly, interpolation)
            dirty_map += _dm
            cov_inv   += _ci

            del _ci, _dm
            gc.collect()

    return dirty_map, cov_inv

def timestream2map(vis_one, vis_mask, time, ra, dec, map_tmp, n_poly = 1, 
        interpolation = 'linear'):

    vis_one = np.array(vis_one)
    vis_one[vis_mask] = 0.

    cov_inv_block = np.zeros(map_tmp.shape * 2, dtype=float)
    
    polys = ortho_poly(time, n_poly, ~vis_mask, 0)
    amps = np.sum(polys * vis_one[None, :], -1)
    vis_fit = np.sum(amps[:, None] * polys, 0)
    vis_one -= vis_fit

    _good  = ( ra  < max(map_tmp.get_axis('ra') ))
    _good *= ( ra  > min(map_tmp.get_axis('ra') ))
    _good *= ( dec < max(map_tmp.get_axis('dec')))
    _good *= ( dec > min(map_tmp.get_axis('dec')))
    _good *= ~vis_mask
    if np.sum(_good) < 5: 
        return al.zeros_like(map_tmp), cov_inv_block

    ra   = ra[_good]
    dec  = dec[_good]
    vis_one  = vis_one[_good]
    vis_mask = vis_mask[_good]
    time = time[_good]

    P = Pointing(('ra', 'dec'), (ra, dec), map_tmp, interpolation)

    _vars = sp.sum(vis_one ** 2.)
    _cont = sp.sum(~vis_mask)
    if _cont != 0:
        _vars /= _cont
    else:
        _vars = T_infinity ** 2.
    if _vars < T_small ** 2:
        print "vars too small"
        _vars = T_small ** 2
    #thermal_noise = np.var(vis_one)
    thermal_noise = _vars
    vis_one = al.make_vect(vis_one[None, :], axis_names=['freq', 'time'])
    N = Noise(vis_one, time)
    N.add_thermal(thermal_noise)
    if n_poly == 1:
        N.deweight_time_mean(T_huge ** 2.)
    elif n_poly == 2:
        N.deweight_time_slope(T_huge ** 2.)
    N.finalize(frequency_correlations=False, preserve_matrices=False)
    vis_weighted = N.weight_time_stream(vis_one)
    dirty_map = P.apply_to_time_axis(vis_weighted)[0,...]

    P.noise_channel_to_map(N, 0, cov_inv_block)

    return dirty_map, cov_inv_block

def make_cleanmap(dirty_map, cov_inv_block):
    
    map_shp = dirty_map.shape
    dirty_map.shape = (np.prod(map_shp), )
    cov_inv_block.shape = (np.prod(map_shp), np.prod(map_shp))
    cov_inv_diag, Rot = linalg.eigh(cov_inv_block, overwrite_a=True)
    map_rotated = sp.dot(Rot.T, dirty_map)
    bad_modes = cov_inv_diag <= 1.e-3 * cov_inv_diag.max()
    print "cov_inv_diag max = %4.1f"%cov_inv_diag.max()
    print "discarded: %4.1f" % (100.0 * sp.sum(bad_modes) / bad_modes.size) +\
                "% of modes"
    map_rotated[bad_modes] = 0.
    cov_inv_diag[bad_modes] = 1.
    print "cov_inv_diag min = %5.4e"%cov_inv_diag.min()
    #cov_inv_diag[cov_inv_diag == 0] = 1.
    #print "cov_inv_diag min = %5.4e"%cov_inv_diag.min()
    map_rotated /= cov_inv_diag
    clean_map = sp.dot(Rot, map_rotated)
    clean_map.shape = map_shp

    noise_diag = 1./cov_inv_diag
    noise_diag[bad_modes] = 0.

    tmp_mat = Rot * noise_diag
    for jj in range(np.prod(map_shp)):
        noise_diag[jj] = sp.dot(tmp_mat[jj, :], Rot[jj, :])
    noise_diag.shape = map_shp
    noise_diag[noise_diag<1.e-20] = 0.

    del cov_inv_diag, Rot, tmp_mat, map_rotated
    gc.collect()

    return clean_map, noise_diag



class MakeMap_Ionly(DirtyMap_GBT):

    def init_ps_datasets(self, ts):

        ts.lin2I()

        func = super(MakeMap_Ionly, self).init_ps_datasets(ts)
        return func

class MakeMap_CombineAll(DirtyMap_GBT):

    def init_ps_datasets(self, ts):

        ts.lin2I()

        ts.main_data_name = self.params['data_sets']
        n_time, n_freq, n_pol, n_bl = ts.main_data.shape
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

        axis_names = ('freq', 'ra', 'dec')
        dirty_map_tmp = np.zeros((n_freq, ) +  self.map_shp)
        dirty_map_tmp = al.make_vect(dirty_map_tmp, axis_names=axis_names)
        dirty_map_tmp.set_axis_info('freq', freq_c, freq_d)
        dirty_map_tmp.set_axis_info('ra',   field_centre[0], self.ra_spacing)
        dirty_map_tmp.set_axis_info('dec',  field_centre[1], self.dec_spacing)
        #self.map_tmp = map_tmp

        self.create_dataset_like('dirty_map',  dirty_map_tmp)
        self.create_dataset_like('clean_map',  dirty_map_tmp)
        self.create_dataset_like('noise_diag', dirty_map_tmp)

        self.df['mask'] = np.zeros(n_freq)

        if self.params['save_cov']:
            axis_names = ('freq', 'ra', 'dec', 'ra', 'dec')
            cov_tmp = np.zeros((n_freq, ) +  self.map_shp + self.map_shp)
            cov_tmp = al.make_vect(cov_tmp, axis_names=axis_names)
            cov_tmp.set_axis_info('freq', freq_c, freq_d)
            cov_tmp.set_axis_info('ra',   field_centre[0], self.ra_spacing)
            cov_tmp.set_axis_info('dec',  field_centre[1], self.dec_spacing)
            self.create_dataset_like('cov_inv', cov_tmp)

        self.df['pol'] = self.pol
        self.df['bl']  = self.bl

        func = ts.freq_data_operate

        return func

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
    #print x.shape, window.shape
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

