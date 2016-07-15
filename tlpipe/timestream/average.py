"""Initialize telescope array and average the timestream."""

import os
import numpy as np
from scipy.linalg import eigh
import h5py
import aipy as a
import tod_task

from caput import mpiutil
from caput import mpiarray
from caput import memh5
from tlpipe.core import tldishes
from tlpipe.utils.path_util import input_path, output_path
from tlpipe.map.fmmode.telescope import tldish


class Average(tod_task.SingleTimestream):
    """Initialize telescope array and average the timestream."""

    params_init = {
                    'beam_theta_range': [0.0, 180.0],
                    'tsys': 50.0,
                    'accuracy_boost': 1.0,
                    'l_boost': 1.0,
                    'bl_range': [0.0, 1.0e7],
                    'auto_correlations': False,
                    'pol': 'xx', # 'yy' or 'I'
                  }

    prefix = 'av_'

    def process(self, ts):

        beam_theta_range = self.params['beam_theta_range']
        tsys = self.params['tsys']
        accuracy_boost = self.params['accuracy_boost']
        l_boost = self.params['l_boost']
        bl_range = self.params['bl_range']
        auto_correlations = self.params['auto_correlations']
        pol = self.params['pol']

        ts.redistribute('frequency')

        lat = ts.attrs['sitelat']
        # lon = ts.attrs['sitelon']
        # lon = 0.0
        lon = np.degrees(ts['ra_dec'][0, 0]) # the firsr ra
        freqs = ts.freq.data.to_numpy_array(root=None)
        ndays = 1
        feeds = ts['feedno'][:]
        az, alt = ts['az_alt'][0]
        az = np.degrees(az)
        alt = np.degrees(alt)
        pointing = [az, alt, 0.0]
        dish_width = ts.attrs['dishdiam']


        if ts.is_dish:
            tel = tldish.TlUnpolarisedDishArray(lat, lon, freqs, beam_theta_range, tsys, ndays, accuracy_boost, l_boost, bl_range, auto_correlations, feeds, pointing, dish_width)
        elif ts.is_cylinder:
            pass
        else:
            raise RuntimeError('Unknown array type %s' % ts.attrs['telescope'])

        # print tel.frequencies
        # print tel.redundancy
        # print tel.theta_size, tel.phi_size
        # print tel.allpairs

        # # make data phi from 0 to 2pi
        # phi = ts['ra_dec'][:, 0] # radians

        # average data
        nt = ts['sec1970'].shape[0]
        phi_size = tel.phi_size
        num, start, end = mpiutil.split_m(nt, phi_size)

        on = np.where(ts['ns_on'][:])[0]
        ts['vis'].local_data[on] = complex(np.nan, np.nan)

        # averate over time
        inds = []
        vis = np.zeros((phi_size,)+ts['vis'].local_data.shape[1:], dtype=ts['vis'].dtype)
        for idx in range(phi_size):
            inds.append(int(0.5 * (start[idx] + end[idx])))
            vis[idx] = np.ma.mean(np.ma.masked_invalid(ts['vis'].local_data[start[idx]:end[idx]]), axis=0) # time mean

        if pol == 'xx':
            vis = vis[:, :, 0, :]
        elif pol == 'yy':
            vis = vis[:, :, 1, :]
        elif pol == 'I':
            vis = 0.5 * (vis[:, :, 0, :] + vis[:, :, 1, :])
        else:
            raise ValueError('Invalid pol: %s' % pol)

        allpairs = tel.allpairs
        redundancy = tel.redundancy

        # reorder bls according to allpairs
        vis_tmp = np.zeros_like(vis)
        bls = [ tuple(bl) for bl in ts['blorder'][:] ]
        print feeds
        print bls
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
            vis_stream[:, :, ind] = np.sum(vis_tmp[:, :, red_bin[ind]:red_bin[ind+1]]) / redundancy[ind]

        phi = (ts['ra_dec'][:, 0])[inds]

        vis = mpiarray.MPIArray.wrap(vis, axis=1)
        vis_h5 = memh5.MemGroup(distributed=True)
        vis_h5.create_dataset('/timestream', data=vis_stream)
        vis_h5.create_dataset('/phi', data=phi)

        # Telescope layout data
        vis_h5.create_dataset('/feedmap', data=tel.feedmap)
        vis_h5.create_dataset('/feedconj', data=tel.feedconj)
        vis_h5.create_dataset('/feedmask', data=tel.feedmask)
        vis_h5.create_dataset('/uniquepairs', data=tel.uniquepairs)
        vis_h5.create_dataset('/baselines', data=tel.baselines)

        # Telescope frequencies
        vis_h5.create_dataset('/frequencies', data=freqs)

        # Write metadata
        # vis_h5.attrs['beamtransfer_path'] = os.path.abspath(bt.directory)
        vis_h5.attrs['ntime'] = phi_size

        vis_h5.to_hdf5('ts.hdf5')


        # ts.add_history(self.history)

        return ts
