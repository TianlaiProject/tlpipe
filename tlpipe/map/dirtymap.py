"""Module to do the map-making."""

from caput import mpiutil
#from tlpipe.pipeline.pipeline import OneAndOne
from tlpipe.timestream import timestream_task
from tlpipe.utils.path_util import output_path
import healpy as hp
import numpy as np
from scipy.ndimage import gaussian_filter
import h5py
import gc

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
