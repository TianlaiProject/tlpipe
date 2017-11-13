"""Initialize telescope array, average the timestream and do the map-making.

Inheritance diagram
-------------------

.. inheritance-diagram:: MapMaking
   :parts: 2

"""

import os
import time
import numpy as np
from scipy.linalg import eigh
import h5py
import aipy as a
import timestream_task
from tlpipe.container.timestream import Timestream

from caput import mpiutil
from caput import mpiarray
from cora.util import hputil
from tlpipe.utils.np_util import unique, average
from tlpipe.utils.path_util import output_path
from tlpipe.map.drift.telescope import tl_dish, tl_cylinder
from tlpipe.map.drift.core import beamtransfer
from tlpipe.map.drift.pipeline import timestream


class MapMaking(timestream_task.TimestreamTask):
    """Initialize telescope array, average the timestream and do the map-making.

    This task calls the submodule :mod:`~tlpipe.map.drift` which uses the m-mode
    formalism method to do the map-making.

    """

    params_init = {
                    'mask_daytime': True,
                    'mask_time_range': [6.0, 22.0], # hour
                    'beam_theta_range': [0.0, 135.0],
                    'tsys': 50.0,
                    'accuracy_boost': 1.0,
                    'l_boost': 1.0,
                    'bl_range': [0.0, 1.0e7],
                    'auto_correlations': False,
                    'pol': 'xx', # 'yy' or 'I'
                    'beam_dir': 'map/bt',
                    'gen_invbeam': True,
                    'noise_weight': True,
                    'ts_dir': 'map/ts',
                    'ts_name': 'ts',
                    'no_m_zero': True,
                    'simulate': False,
                    'input_maps': [],
                    'add_noise': True,
                    'dirty_map': False,
                    'nbin': None, # use this if multi-freq synthisize
                    'method': 'svd',
                    'normalize': True, # only used for dirty map-making
                    'threshold': 1.0e3, # only used for dirty map-making
                  }

    prefix = 'mm_'

    def process(self, ts):

        assert isinstance(ts, Timestream), '%s only works for Timestream object' % self.__class__.__name__

        mask_daytime = self.params['mask_daytime']
        mask_time_range = self.params['mask_time_range']
        beam_theta_range = self.params['beam_theta_range']
        tsys = self.params['tsys']
        accuracy_boost = self.params['accuracy_boost']
        l_boost = self.params['l_boost']
        bl_range = self.params['bl_range']
        auto_correlations = self.params['auto_correlations']
        pol = self.params['pol']
        beam_dir = output_path(self.params['beam_dir'])
        gen_inv = self.params['gen_invbeam']
        noise_weight = self.params['noise_weight']
        ts_dir = output_path(self.params['ts_dir'])
        ts_name = self.params['ts_name']
        no_m_zero = self.params['no_m_zero']
        simulate = self.params['simulate']
        input_maps = self.params['input_maps']
        add_noise = self.params['add_noise']
        dirty_map = self.params['dirty_map']
        nbin = self.params['nbin']
        method = self.params['method']
        normalize = self.params['normalize']
        threshold = self.params['threshold']

        ts.redistribute('baseline')

        lat = ts.attrs['sitelat']
        # lon = ts.attrs['sitelon']
        lon = 0.0
        # lon = np.degrees(ts['ra_dec'][0, 0]) # the first ra
        local_origin = False
        freqs = ts.freq[:] # MHz
        nfreq = freqs.shape[0]
        band_width = ts.attrs['freqstep'] # MHz
        try:
            ndays = ts.attrs['ndays']
        except KeyError:
            ndays = 1
        feeds = ts['feedno'][:]
        bl_order = mpiutil.gather_array(ts.local_bl, axis=0, root=None, comm=ts.comm)
        bls = [ tuple(bl) for bl in bl_order ]
        az, alt = ts['az_alt'][0]
        az = np.degrees(az)
        alt = np.degrees(alt)
        pointing = [az, alt, 0.0]
        feedpos = ts['feedpos'][:]

        if ts.is_dish:
            dish_width = ts.attrs['dishdiam']
            tel = tl_dish.TlUnpolarisedDishArray(lat, lon, freqs, band_width, tsys, ndays, accuracy_boost, l_boost, bl_range, auto_correlations, local_origin, dish_width, feedpos, pointing)
        elif ts.is_cylinder:
            # factor = 1.2 # suppose an illumination efficiency, keep same with that in timestream_common
            factor = 0.79 # for xx
            # factor = 0.88 # for yy
            cyl_width = factor * ts.attrs['cywid']
            tel = tl_cylinder.TlUnpolarisedCylinder(lat, lon, freqs, band_width, tsys, ndays, accuracy_boost, l_boost, bl_range, auto_correlations, local_origin, cyl_width, feedpos)
        else:
            raise RuntimeError('Unknown array type %s' % ts.attrs['telescope'])

        if not simulate:
            # mask daytime data
            if mask_daytime:
                day_inds = np.where(np.logical_and(ts['local_hour'][:]>=mask_time_range[0], ts['local_hour'][:]<=mask_time_range[1]))[0]
                ts.local_vis_mask[day_inds] = True # do not change vis directly

            # average data
            nt = ts['sec1970'].shape[0]
            phi_size = 2*tel.mmax + 1
            nt_m = float(nt) / phi_size

            # roll data to have phi=0 near the first
            roll_len = np.int(np.around(0.5*nt_m))
            ts.local_vis[:] = np.roll(ts.local_vis[:], roll_len, axis=0)
            ts.local_vis_mask[:] = np.roll(ts.local_vis_mask[:], roll_len, axis=0)
            ts['ra_dec'][:] = np.roll(ts['ra_dec'][:], roll_len, axis=0)

            repeat_inds = np.repeat(np.arange(nt), phi_size)
            num, start, end = mpiutil.split_m(nt*phi_size, phi_size)

            # phi = np.zeros((phi_size,), dtype=ts['ra_dec'].dtype)
            phi = np.linspace(0, 2*np.pi, phi_size, endpoint=False)
            vis = np.zeros((phi_size,)+ts.local_vis.shape[1:], dtype=ts.vis.dtype)
            # average over time
            for idx in xrange(phi_size):
                inds, weight = unique(repeat_inds[start[idx]:end[idx]], return_counts=True)
                vis[idx] = average(np.ma.array(ts.local_vis[inds], mask=ts.local_vis_mask[inds]), axis=0, weights=weight) # time mean
                # phi[idx] = np.average(ts['ra_dec'][:, 0][inds], axis=0, weights=weight)

            del ts # no longer need ts

            if pol == 'xx':
                vis = vis[:, :, 0, :]
            elif pol == 'yy':
                vis = vis[:, :, 1, :]
            elif pol == 'I':
                vis = 0.5 * (vis[:, :, 0, :] + vis[:, :, 1, :])
            elif pol == 'all':
                vis = np.sum(vis, axis=2) # sum over all pol
            else:
                raise ValueError('Invalid pol: %s' % pol)

            # redistribute vis to time axis
            vis = mpiarray.MPIArray.wrap(vis, axis=2).redistribute(0).local_array

            allpairs = tel.allpairs
            redundancy = tel.redundancy
            nrd = len(redundancy)

            # reorder bls according to allpairs
            vis_tmp = np.zeros_like(vis)
            for ind, (a1, a2) in enumerate(allpairs):
                try:
                    b_ind = bls.index((feeds[a1], feeds[a2]))
                    vis_tmp[:, :, ind] = vis[:, :, b_ind]
                except ValueError:
                    b_ind = bls.index((feeds[a2], feeds[a1]))
                    vis_tmp[:, :, ind] = vis[:, :, b_ind].conj()

            del vis

            # average over redundancy
            vis_stream = np.zeros(vis_tmp.shape[:-1]+(nrd,), dtype=vis_tmp.dtype)
            red_bin = np.cumsum(np.insert(redundancy, 0, 0)) # redundancy bin
            # average over redundancy
            for ind in xrange(nrd):
                vis_stream[:, :, ind] = np.sum(vis_tmp[:, :, red_bin[ind]:red_bin[ind+1]], axis=2) / redundancy[ind]

            del vis_tmp

        # beamtransfer
        bt = beamtransfer.BeamTransfer(beam_dir, tel, noise_weight, True)
        bt.generate()

        if simulate:
            ndays = 733
            print ndays
            tstream = timestream.simulate(bt, ts_dir, ts_name, input_maps, ndays, add_noise=add_noise)
        else:
            # timestream and map-making
            tstream = timestream.Timestream(ts_dir, ts_name, bt, no_m_zero)
            for fi in mpiutil.mpilist(range(nfreq)):
                # Make directory if required
                if not os.path.exists(tstream._fdir(fi)):
                    os.makedirs(tstream._fdir(fi))

                # Write file contents
                with h5py.File(tstream._ffile(fi), 'w') as f:
                    # Timestream data
                    # allocate space for vis_stream
                    shp = (nrd, phi_size)
                    f.create_dataset('/timestream', shp, dtype=vis_stream.dtype)
                    f.create_dataset('/phi', data=phi)

                    # Telescope layout data
                    f.create_dataset('/feedmap', data=tel.feedmap)
                    f.create_dataset('/feedconj', data=tel.feedconj)
                    f.create_dataset('/feedmask', data=tel.feedmask)
                    f.create_dataset('/uniquepairs', data=tel.uniquepairs)
                    f.create_dataset('/baselines', data=tel.baselines)

                    # Telescope frequencies
                    f.create_dataset('/frequencies', data=freqs)

                    # Write metadata
                    f.attrs['beamtransfer_path'] = os.path.abspath(bt.directory)
                    f.attrs['ntime'] = phi_size

            mpiutil.barrier()

            # write vis_stream to files
            num, s, e = mpiutil.split_local(phi_size)
            for fi in xrange(nfreq):
                for i in range(10):
                    try:
                        with h5py.File(tstream._ffile(fi), 'r+') as f:
                            f['/timestream'][:, s:e] = vis_stream[:, fi, :].T
                    except IOError:
                        time.sleep(0.5)
                        continue
                else:
                    raise RuntimeError('Could not open file: %s...' % tstream._ffile(fi))

            del vis_stream
            mpiutil.barrier()

        tstream.generate_mmodes()
        nside = hputil.nside_for_lmax(tel.lmax, accuracy_boost=tel.accuracy_boost)
        if dirty_map:
            tstream.mapmake_full(nside, 'map_full_dirty.hdf5', nbin, dirty=True, method=method, normalize=normalize, threshold=threshold)
        else:
            tstream.mapmake_full(nside, 'map_full.hdf5', nbin, dirty=False, method=method, normalize=normalize, threshold=threshold)

        # ts.add_history(self.history)

        return tstream
