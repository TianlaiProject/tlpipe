"""Initialize telescope array, average the timestream and do the map-making."""

import os
import numpy as np
from scipy.linalg import eigh
import h5py
import aipy as a
import tod_task

from caput import mpiutil
from caput import mpiarray
from caput import memh5
from cora.util import hputil
from tlpipe.utils.np_util import unique, average
from tlpipe.utils.path_util import output_path
from tlpipe.map.drift.telescope import tl_dish, tl_cylinder
from tlpipe.map.drift.core import beamtransfer
from tlpipe.map.drift.pipeline import timestream


class MapMaking(tod_task.SingleTimestream):
    """Initialize telescope array, average the timestream and do the map-making."""

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
                    'simulate': False,
                    'input_maps': [],
                    'add_noise': True,
                  }

    prefix = 'mm_'

    def process(self, ts):

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
        simulate = self.params['simulate']
        input_maps = self.params['input_maps']
        add_noise = self.params['add_noise']

        ts.redistribute('frequency')

        lat = ts.attrs['sitelat']
        # lon = ts.attrs['sitelon']
        lon = 0.0
        # lon = np.degrees(ts['ra_dec'][0, 0]) # the first ra
        local_origin = False
        freq = ts.freq
        freqs = ts.freq.data.to_numpy_array(root=None)
        ndays = 1
        feeds = ts['feedno'][:]
        az, alt = ts['az_alt'][0]
        az = np.degrees(az)
        alt = np.degrees(alt)
        pointing = [az, alt, 0.0]
        feedpos = ts['feedpos'][:]

        if ts.is_dish:
            dish_width = ts.attrs['dishdiam']
            tel = tl_dish.TlUnpolarisedDishArray(lat, lon, freqs, tsys, ndays, accuracy_boost, l_boost, bl_range, auto_correlations, local_origin, dish_width, feedpos, pointing)
        elif ts.is_cylinder:
            cyl_width = ts.attrs['cywid']
            tel = tl_cylinder.TlUnpolarisedCylinder(lat, lon, freqs, tsys, ndays, accuracy_boost, l_boost, bl_range, auto_correlations, local_origin, cyl_width, feedpos)
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
            for idx in range(phi_size):
                inds, weight = unique(repeat_inds[start[idx]:end[idx]], return_counts=True)
                vis[idx] = average(np.ma.array(ts.local_vis[inds], mask=ts.local_vis_mask[inds]), axis=0, weights=weight) # time mean
                # phi[idx] = np.average(ts['ra_dec'][:, 0][inds], axis=0, weights=weight)

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

            allpairs = tel.allpairs
            redundancy = tel.redundancy

            # reorder bls according to allpairs
            vis_tmp = np.zeros_like(vis)
            bls = [ tuple(bl) for bl in ts['blorder'][:] ]
            for ind, (a1, a2) in enumerate(allpairs):
                try:
                    b_ind = bls.index((feeds[a1], feeds[a2]))
                    vis_tmp[:, :, ind] = vis[:, :, b_ind]
                except ValueError:
                    b_ind = bls.index((feeds[a2], feeds[a1]))
                    vis_tmp[:, :, ind] = vis[:, :, b_ind].conj()

            # average over redundancy
            vis_stream = np.zeros(vis.shape[:-1]+(len(redundancy),), dtype=vis_tmp.dtype)
            red_bin = np.cumsum(np.insert(redundancy, 0, 0)) # redundancy bin
            # average over redundancy
            for ind in range(len(redundancy)):
                vis_stream[:, :, ind] = np.sum(vis_tmp[:, :, red_bin[ind]:red_bin[ind+1]], axis=2) / redundancy[ind]

            del vis
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
            tstream = timestream.Timestream(ts_dir, ts_name, bt)
            for lfi, fi in freq.data.enumerate(axis=0):
                # Make directory if required
                if not os.path.exists(tstream._fdir(fi)):
                    os.makedirs(tstream._fdir(fi))

                # Write file contents
                with h5py.File(tstream._ffile(fi), 'w') as f:

                    # Timestream data
                    f.create_dataset('/timestream', data=vis_stream[:, lfi].T)
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

        tstream.generate_mmodes()
        nside = hputil.nside_for_lmax(tel.lmax, accuracy_boost=tel.accuracy_boost)
        tstream.mapmake_full(nside, 'map_full.hdf5')

        # ts.add_history(self.history)

        return tstream
