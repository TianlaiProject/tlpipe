"""Module to do the map-making."""

from caput import mpiutil
#from tlpipe.pipeline.pipeline import OneAndOne
from tlpipe.timestream import timestream_task
from tlpipe.utils.path_util import output_path
import healpy as hp
import numpy as np
import h5py


class DirtyMap(timestream_task.TimestreamTask):
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
