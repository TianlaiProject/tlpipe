"""Bandpass calibration by using a strong point source.

Inheritance diagram
-------------------

.. inheritance-diagram:: BpCal
   :parts: 2

"""

import re
import numpy as np
from scipy import signal
import ephem
import h5py
import aipy as a
from . import timestream_task

from caput import mpiutil
from tlpipe.utils.path_util import output_path
from tlpipe.cal import calibrators


class BpCal(timestream_task.TimestreamTask):
    """Bandpass calibration by using a strong point source.
    """

    params_init = {
                    'calibrator': 'cyg',
                    'lower_dt': 900, # delta t in second, 900s = 15min
                    'avg_tis': 7, # average 7 time indices, to avoid empty data for 5 time indeces noise source ON
                    'kernel_size': 13, # smoothing kernel size
                    'smooth_type': 'median', # or 'mean'
                    'apply_bandpass': True,
                    'save_bandpass': False,
                    'bandpass_file': 'bandpass/bandpass.hdf5',
                  }

    prefix = 'bc_'

    def process(self, ts):

        calibrator = self.params['calibrator']
        lower_dt = self.params['lower_dt']
        avg_tis = self.params['avg_tis']
        kernel_size = self.params['kernel_size']
        smooth_type = self.params['smooth_type']
        apply_bandpass = self.params['apply_bandpass']
        save_bandpass = self.params['save_bandpass']
        bandpass_file = self.params['bandpass_file']
        # show_progress = self.params['show_progress']
        # progress_step = self.params['progress_step']
        tag_output_iter = self.params['tag_output_iter']

        if not (apply_bandpass or save_bandpass):
            # need to do nothing
            return super(BpCal, self).process(ts)

        if 'pol' in ts.keys():
            pol_type = ts['pol'].attrs['pol_type']
            if pol_type != 'linear':
                raise RuntimeError('Can not do ps_cal for pol_type: %s' % pol_type)

        # get the calibrator
        try:
            s = calibrators.get_src(calibrator)
        except KeyError:
            if mpiutil.rank0:
                print('Calibrator %s is unavailable, available calibrators are:')
                for key, d in calibrators.src_data.items():
                    print('%8s  ->  %12s' % (key, d[0]))
            raise RuntimeError('Calibrator %s is unavailable')
        if mpiutil.rank0:
            print('Try to calibrate with %s...' % s.src_name)

        # get transit time of calibrator
        # array
        aa = ts.array
        jul_date = mpiutil.gather_array(ts.local_time[:], root=None, comm=ts.comm)
        aa.set_jultime(jul_date[0]) # the first obs time point
        next_transit = aa.next_transit(s)
        transit_time = a.phs.ephem2juldate(next_transit) # Julian date
        # get time zone
        pattern = '[-+]?\d+'
        try:
            tz = re.search(pattern, ts.attrs['timezone'].decode('ascii')).group() # ts.attrs['timezone'] is bytes in python3
        except AttributeError:
            tz = re.search(pattern, ts.attrs['timezone']).group() # ts.attrs['timezone'] is str in python3.10
        tz = int(tz)
        local_next_transit = ephem.Date(next_transit + tz * ephem.hour) # plus 8h to get Beijing time
        # if transit_time > ts['jul_date'][-1]:
        if transit_time > max(jul_date[-1], jul_date.max()):
            raise RuntimeError('Data does not contain local transit time %s of source %s' % (local_next_transit, calibrator))

        # the first transit index
        transit_ind = np.searchsorted(jul_date[:], transit_time)
        if mpiutil.rank0:
            print('transit ind of %s: %s, time: %s' % (s.src_name, transit_ind, local_next_transit))

        ### now only use the first transit point to do the cal
        ### may need to improve in the future
        int_time = ts.attrs['inttime'] # second
        start_ind = transit_ind - int(lower_dt / int_time) - avg_tis // 2
        end_ind = transit_ind + int(lower_dt / int_time) + avg_tis // 2

        if start_ind < 0:
            raise RuntimeError('Data does not contain the left lower end')
        if end_ind >= len(jul_date):
            raise RuntimeError('Data does not contain the right lower end')

        # gather start inds to rank0
        # jd_local_shape = ts.time.local_shape[0]
        jd_local_offset = ts.time.local_offset[0]

        vis_local_left = ts.local_vis[max(0, start_ind-jd_local_offset):max(0, start_ind-jd_local_offset+avg_tis)]
        # print('rank %d has vis_local_left shape = %s' % (mpiutil.rank, vis_local_left.shape))
        vis_left = mpiutil.gather_array(vis_local_left, axis=0, root=0, comm=ts.comm)
        vis_mask_local_left = ts.local_vis_mask[max(0, start_ind-jd_local_offset):max(0, start_ind-jd_local_offset+avg_tis)]
        vis_mask_left = mpiutil.gather_array(vis_mask_local_left, axis=0, root=0, comm=ts.comm)

        # gather peak inds to rank0
        vis_local_peak = ts.local_vis[max(0, transit_ind-jd_local_offset-avg_tis//2):max(0, transit_ind-jd_local_offset-avg_tis//2+avg_tis)]
        # print('rank %d has vis_local_peak shape = %s' % (mpiutil.rank, vis_local_peak.shape))
        vis_peak = mpiutil.gather_array(vis_local_peak, axis=0, root=0, comm=ts.comm)
        vis_mask_local_peak = ts.local_vis_mask[max(0, transit_ind-jd_local_offset-avg_tis//2):max(0, transit_ind-jd_local_offset-avg_tis//2+avg_tis)]
        vis_mask_peak = mpiutil.gather_array(vis_mask_local_peak, axis=0, root=0, comm=ts.comm)

        # gather right inds to rank0
        vis_local_right = ts.local_vis[max(0, end_ind-jd_local_offset-avg_tis):max(0, end_ind-jd_local_offset)]
        # print('rank %d has vis_local_right shape = %s' % (mpiutil.rank, vis_local_right.shape))
        vis_right = mpiutil.gather_array(vis_local_right, axis=0, root=0, comm=ts.comm)
        vis_mask_local_right = ts.local_vis_mask[max(0, end_ind-jd_local_offset-avg_tis):max(0, end_ind-jd_local_offset)]
        vis_mask_right = mpiutil.gather_array(vis_mask_local_right, axis=0, root=0, comm=ts.comm)

        if mpiutil.rank0:
            # time average
            vis_left_avg = np.ma.array(vis_left, mask=vis_mask_left).mean(axis=0) # (freq, bl) or (freq, pol, bl)
            vis_peak_avg = np.ma.array(vis_peak, mask=vis_mask_peak).mean(axis=0) # (freq, bl) or (freq, pol, bl)
            vis_right_avg = np.ma.array(vis_right, mask=vis_mask_right).mean(axis=0) # (freq, bl) or (freq, pol, bl)

            vis_src = vis_peak_avg - 0.5 * (vis_left_avg + vis_right_avg) # (freq, bl) or (freq, pol, bl)
            vis_src_abs = np.ma.abs(vis_src).filled(fill_value=0) # (freq, bl) or (freq, pol, bl)

            if smooth_type == 'median':
                vis_src_abs_smooth = signal.medfilt(vis_src_abs, kernel_size=[kernel_size] + [1]*(len(vis_src_abs.shape) - 1))
            elif smooth_type == 'mean':
                mean_kernel = np.ones(kernel_size) / kernel_size
                shp = vis_src_abs.shape
                vis_src_abs_smooth = np.zeros(shp).reshape[shp[0], -1]
                vis_src_abs = vis_src_abs.reshape[shp[0], -1]
                for i in range(vis_src_abs_smooth.shape[-1]):
                    vis_src_abs_smooth[:, i] = np.convolve(vis_src_abs[:, i], mean_kernel, mode='same')
                vis_src_abs_smooth = vis_src_abs_smooth.reshape(shp)

            if len(vis_src_abs_smooth.shape) == 2:
                bandpass = vis_src_abs_smooth / s.get_jys(1.0e-3 * ts.freq[:])[:, np.newaxis]
            elif len(vis_src_abs_smooth.shape) == 3:
                bandpass = vis_src_abs_smooth / s.get_jys(1.0e-3 * ts.freq[:])[:, np.newaxis, np.newaxis]
            else:
                raise RuntimeError('bandpass should be a two or three dimensional array')

            # normalize bandpass to ~1
            bandpass /= bandpass.mean(axis=0)[np.newaxis]
            bandpass = bandpass.astype(np.float64)

            if save_bandpass:
                if tag_output_iter:
                    bandpass_file = output_path(bandpass_file, iteration=self.iteration)
                else:
                    bandpass_file = output_path(bandpass_file)
                # save file
                with h5py.File(bandpass_file, 'w') as f:
                    f.create_dataset('bandpass', data=bandpass)
                    f.create_dataset('freq', data=ts.freq)
                    f.create_dataset('baseline', data=ts.bl)
                    if len(bandpass.shape) == 2:
                        f['bandpass'].attrs['dims'] = '(freq, baseline)'
                    elif len(bandpass.shape) == 3:
                        f['bandpass'].attrs['dims'] = '(freq, pol, baseline)'
                        f.create_dataset('pol', data=ts.pol)
                    else:
                        raise RuntimeError('bandpass should be a two or three dimensional array')

        else:
            bandpass = np.zeros(ts.vis.shape[1:], dtype=np.float64) # (freq, bl) or (freq, pol, bl)

        if apply_bandpass:
            if mpiutil.size > 1:
                # mpiutil.world.Bcast(bandpass, root=0)
                ts.comm.Bcast(bandpass, root=0)

            ts.local_vis[:] = ts.local_vis / bandpass[np.newaxis]

        return super(BpCal, self).process(ts)
