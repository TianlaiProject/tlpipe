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
from scipy.interpolate import interp1d, Rbf
import h5py
import aipy as a
import timestream_task
from tlpipe.container.timestream import Timestream

from caput import mpiutil
from caput import mpiarray
from caput import memh5
from cora.util import hputil
from tlpipe.utils.np_util import unique, average
from tlpipe.utils.path_util import output_path
from tlpipe.map.drift.core import beamtransfer
from tlpipe.map.drift.pipeline import timestream


class MapMaking(timestream_task.TimestreamTask):
    """Initialize telescope array, average the timestream and do the map-making.

    This task calls the submodule :mod:`~tlpipe.map.drift` which uses the m-mode
    formalism method to do the map-making.

    """

    params_init = {
                    'mask_daytime': False,
                    'mask_time_range': [8.0, 19.5], # hour
                    'tsys': 50.0,
                    'accuracy_boost': 1.0,
                    'l_boost': 1.0,
                    'bl_range': [0.0, 1.0e7],
                    'auto_correlations': False,
                    'time_avg': 'avg', # or 'fft'
                    'pol': 'xx', # 'yy' or 'I'
                    'interp': 'none', # 'linear', 'nearest' or 'rbf'
                    'beam_dir': 'map/bt',
                    'use_existed_beam': False,
                    'gen_invbeam': True,
                    'noise_weight': True,
                    'ts_dir': 'map/ts',
                    'ts_name': 'ts',
                    'no_m_zero': True,
                    'simulate': False,
                    'input_maps': [],
                    'prior_map': None, # or 'prior.hdf5'
                    'add_noise': True,
                    'dirty_map': False,
                    'nbin': None, # use this if multi-freq synthesize
                    'method': 'svd', # or tk
                    'normalize': True, # only used for dirty map-making
                    'threshold': 1.0e3, # only used for dirty map-making
                    'epsilon': 0.0001, # regularization parameter for tk
                    'correct_order': 1, # tk deconv correction order
                  }

    prefix = 'mm_'

    def process(self, ts):

        mask_daytime = self.params['mask_daytime']
        mask_time_range = self.params['mask_time_range']
        tsys = self.params['tsys']
        accuracy_boost = self.params['accuracy_boost']
        l_boost = self.params['l_boost']
        bl_range = self.params['bl_range']
        auto_correlations = self.params['auto_correlations']
        time_avg = self.params['time_avg']
        pol = self.params['pol']
        interp = self.params['interp']
        beam_dir = output_path(self.params['beam_dir'])
        use_existed_beam = self.params['use_existed_beam']
        gen_inv = self.params['gen_invbeam']
        noise_weight = self.params['noise_weight']
        ts_dir = output_path(self.params['ts_dir'])
        ts_name = self.params['ts_name']
        no_m_zero = self.params['no_m_zero']
        simulate = self.params['simulate']
        input_maps = self.params['input_maps']
        prior_map = self.params['prior_map']
        add_noise = self.params['add_noise']
        dirty_map = self.params['dirty_map']
        nbin = self.params['nbin']
        method = self.params['method']
        normalize = self.params['normalize']
        threshold = self.params['threshold']
        eps = self.params['epsilon']
        correct_order = self.params['correct_order']

        if use_existed_beam:
            # load the saved telescope from disk
            tel = None
        else:
            assert isinstance(ts, Timestream), '%s only works for Timestream object' % self.__class__.__name__

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
                from tlpipe.map.drift.telescope import tl_dish

                dish_width = ts.attrs['dishdiam']
                tel = tl_dish.TlUnpolarisedDishArray(lat, lon, freqs, band_width, tsys, ndays, accuracy_boost, l_boost, bl_range, auto_correlations, local_origin, dish_width, feedpos, pointing)
            elif ts.is_cylinder:
                from tlpipe.map.drift.telescope import tl_cylinder

                # factor = 1.2 # suppose an illumination efficiency, keep same with that in timestream_common
                factor = 0.79 # for xx
                # factor = 0.88 # for yy
                cyl_width = factor * ts.attrs['cywid']
                tel = tl_cylinder.TlUnpolarisedCylinder(lat, lon, freqs, band_width, tsys, ndays, accuracy_boost, l_boost, bl_range, auto_correlations, local_origin, cyl_width, feedpos)
            else:
                raise RuntimeError('Unknown array type %s' % ts.attrs['telescope'])

            if not simulate:
                # select the corresponding vis and vis_mask
                if pol == 'xx':
                    local_vis = ts.local_vis[:, :, 0, :]
                    local_vis_mask = ts.local_vis_mask[:, :, 0, :]
                elif pol == 'yy':
                    local_vis = ts.local_vis[:, :, 1, :]
                    local_vis_mask = ts.local_vis_mask[:, :, 1, :]
                elif pol == 'I':
                    xx_vis = ts.local_vis[:, :, 0, :]
                    xx_vis_mask = ts.local_vis_mask[:, :, 0, :]
                    yy_vis = ts.local_vis[:, :, 1, :]
                    yy_vis_mask = ts.local_vis_mask[:, :, 1, :]

                    local_vis = np.zeros_like(xx_vis)
                    for ti in xrange(local_vis.shape[0]):
                        for fi in xrange(local_vis.shape[1]):
                            for bi in xrange(local_vis.shape[2]):
                                if xx_vis_mask[ti, fi, bi] != yy_vis_mask[ti, fi, bi]:
                                    if xx_vis_mask[ti, fi, bi]:
                                        local_vis[ti, fi, bi] = yy_vis[ti, fi, bi]
                                    else:
                                        local_vis[ti, fi, bi] = xx_vis[ti, fi, bi]
                                else:
                                    local_vis[ti, fi, bi] = 0.5 * (xx_vis[ti, fi, bi] + yy_vis[ti, fi, bi])
                    local_vis_mask = xx_vis_mask | yy_vis_mask
                else:
                    raise ValueError('Invalid pol: %s' % pol)

                if interp != 'none':
                    for fi in xrange(local_vis.shape[1]):
                        for bi in xrange(local_vis.shape[2]):
                            # interpolate for local_vis
                            true_inds = np.where(local_vis_mask[:, fi, bi])[0] # masked inds
                            if len(true_inds) > 0:
                                false_inds = np.where(~local_vis_mask[:, fi, bi])[0] # un-masked inds
                                if len(false_inds) > 0.1 * local_vis.shape[0]:
                # nearest interpolate for local_vis
                                    if interp in ('linear', 'nearest'):
                                        itp_real = interp1d(false_inds, local_vis[false_inds, fi, bi].real, kind=interp, fill_value='extrapolate', assume_sorted=True)
                                        itp_imag = interp1d(false_inds, local_vis[false_inds, fi, bi].imag, kind=interp, fill_value='extrapolate', assume_sorted=True)
                                    elif interp == 'rbf':
                                        itp_real = Rbf(false_inds, local_vis[false_inds, fi, bi].real, smooth=10)
                                        itp_imag = Rbf(false_inds, local_vis[false_inds, fi, bi].imag, smooth=10)
                                    else:
                                        raise ValueError('Unknown interpolation method: %s' % interp)
                                    local_vis[true_inds, fi, bi] = itp_real(true_inds) + 1.0J * itp_imag(true_inds) # the interpolated vis
                                else:
                                    local_vis[:, fi, bi] = 0 # TODO: may need to take special care

                # average data
                nt = ts['sec1970'].shape[0]
                phi_size = 2*tel.mmax + 1

                # phi = np.zeros((phi_size,), dtype=ts['ra_dec'].dtype)
                phi = np.linspace(0, 2*np.pi, phi_size, endpoint=False)
                vis = np.zeros((phi_size,)+local_vis.shape[1:], dtype=local_vis.dtype)

                if time_avg == 'avg':
                    nt_m = float(nt) / phi_size
                    # roll data to have phi=0 near the first
                    roll_len = np.int(np.around(0.5*nt_m))
                    local_vis[:] = np.roll(local_vis[:], roll_len, axis=0)
                    if interp == 'none':
                        local_vis_mask[:] = np.roll(local_vis_mask[:], roll_len, axis=0)
                    # ts['ra_dec'][:] = np.roll(ts['ra_dec'][:], roll_len, axis=0)

                    repeat_inds = np.repeat(np.arange(nt), phi_size)
                    num, start, end = mpiutil.split_m(nt*phi_size, phi_size)

                    # average over time
                    for idx in xrange(phi_size):
                        inds, weight = unique(repeat_inds[start[idx]:end[idx]], return_counts=True)
                        if interp == 'none':
                            vis[idx] = average(np.ma.array(local_vis[inds], mask=local_vis_mask[inds]), axis=0, weights=weight) # time mean
                        else:
                            vis[idx] = average(local_vis[inds], axis=0, weights=weight) # time mean
                        # phi[idx] = np.average(ts['ra_dec'][:, 0][inds], axis=0, weights=weight)
                elif time_avg == 'fft':
                    if interp == 'none':
                        raise ValueError('Can not do fft average without first interpolation')
                    Vm = np.fft.fftshift(np.fft.fft(local_vis, axis=0), axes=0)
                    vis[:] = np.fft.ifft(np.fft.ifftshift(Vm[nt/2-tel.mmax:nt/2+tel.mmax+1], axes=0), axis=0) / (1.0 * nt / phi_size)

                    # for fi in xrange(vis.shape[1]):
                    #     for bi in xrange(vis.shape[2]):
                    #         # plot local_vis and vis
                    #         import matplotlib
                    #         matplotlib.use('Agg')
                    #         import matplotlib.pyplot as plt

                    #         phi0 = np.linspace(0, 2*np.pi, nt, endpoint=False)
                    #         phi1 = np.linspace(0, 2*np.pi, phi_size, endpoint=False)
                    #         plt.figure()
                    #         plt.subplot(211)
                    #         plt.plot(phi0, local_vis[:, fi, bi].real, label='v0.real')
                    #         plt.plot(phi1, vis[:, fi, bi].real, label='v1.real')
                    #         plt.legend()
                    #         plt.subplot(212)
                    #         plt.plot(phi0, local_vis[:, fi, bi].imag, label='v0.imag')
                    #         plt.plot(phi1, vis[:, fi, bi].imag, label='v1.imag')
                    #         plt.legend()
                    #         plt.savefig('vis_fft/vis_%d_%d.png' % (fi, bi))
                    #         plt.close()

                else:
                    raise ValueError('Unknown time_avg: %s' % time_avg)

                del local_vis
                del local_vis_mask

                # mask daytime data
                if mask_daytime:
                    day_or_night = np.where(ts['local_hour'][:]>=mask_time_range[0] & ts['local_hour'][:]<=mask_time_range[1], True, False)
                    day_inds = np.where(np.repeat(day_or_night, phi_size).reshape(nt, phi_size).astype(np.int).sum(axis=1).astype(bool))[0]
                    vis[day_inds] = 0

                del ts # no longer need ts

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
        if not use_existed_beam:
            bt.generate()
        if tel is None:
            tel = bt.telescope

        if simulate:
            ndays = 733
            tstream = timestream.simulate(bt, ts_dir, ts_name, input_maps, ndays, add_noise=add_noise)
        else:
            # timestream and map-making
            tstream = timestream.Timestream(ts_dir, ts_name, bt, no_m_zero)
            parent_path = os.path.dirname(tstream._fdir(0))

            if os.path.exists(parent_path + '/COMPLETED'):
                if mpiutil.rank0:
                    print 'Use existed timestream_f files in %s' % parent_path
            else:
                for fi in mpiutil.mpirange(nfreq):
                    # Make directory if required
                    if not os.path.exists(tstream._fdir(fi)):
                        os.makedirs(tstream._fdir(fi))

                # create memh5 object and write data to temporary file
                vis_h5 = memh5.MemGroup(distributed=True)
                vis_h5.create_dataset('/timestream', data=mpiarray.MPIArray.wrap(vis_stream, axis=0))
                tmp_file = parent_path +'/vis_stream_temp.hdf5'
                vis_h5.to_hdf5(tmp_file, hints=False)
                del vis_h5

                # re-organize data as need for tstream
                # make load even among nodes
                for fi in mpiutil.mpirange(nfreq, method='rand'):
                    # read the needed data from the temporary file
                    with h5py.File(tmp_file, 'r') as f:
                        vis_fi = f['/timestream'][:, fi, :]
                    # Write file contents
                    with h5py.File(tstream._ffile(fi), 'w') as f:
                        # Timestream data
                        # allocate space for vis_stream
                        shp = (nrd, phi_size)
                        f.create_dataset('/timestream', data=vis_fi.T)
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

                # remove temp file
                if mpiutil.rank0:
                    os.remove(tmp_file)
                    # mark all frequencies tstream files are saved correctly
                    open(parent_path + '/COMPLETED', 'a').close()

        tstream.generate_mmodes()
        nside = hputil.nside_for_lmax(tel.lmax, accuracy_boost=tel.accuracy_boost)
        if dirty_map:
            tstream.mapmake_full(nside, 'map_full_dirty.hdf5', nbin, dirty=True, method=method, normalize=normalize, threshold=threshold)
        else:
            tstream.mapmake_full(nside, 'map_full.hdf5', nbin, dirty=False, method=method, normalize=normalize, threshold=threshold, eps=eps, correct_order=correct_order, prior_map_file=prior_map)

        # ts.add_history(self.history)

        return tstream


    def read_process_write(self, ts):
        """Overwrite the method of superclass."""

        use_existed_beam = self.params['use_existed_beam']

        if use_existed_beam:
            self.process(ts)
        else:
            return super(MapMaking, self).read_process_write(ts)