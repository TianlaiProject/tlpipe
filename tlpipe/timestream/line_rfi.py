"""Line RFI flagging.

Inheritance diagram
-------------------

.. inheritance-diagram:: Flag
   :parts: 2

"""

import warnings
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import timestream_task
from tlpipe.container.raw_timestream import RawTimestream
from tlpipe.container.timestream import Timestream
from tlpipe.utils.sg_filter import savitzky_golay
from tlpipe.utils.path_util import output_path
import tlpipe.plot
import matplotlib.pyplot as plt


class Flag(timestream_task.TimestreamTask):
    """Line RFI flagging.

    This task flags the line RFI along time (then frequency) by first integrate
    data along frequency (and correspondingly time) axis, and mask values that
    exceeds the given threshold.

    """

    params_init = {
                    'freq_window': 15,
                    'time_window': 15,
                    'freq_sigma': 2.0,
                    'time_sigma': 5.0,
                    'plot_fit': False, # plot the smoothing fit
                    'freq_fig_name': 'rfi_freq',
                    'time_fig_name': 'rfi_time',
                  }

    prefix = 'lf_'

    def process(self, ts):

        freq_window = self.params['freq_window']
        time_window = self.params['time_window']

        ts.redistribute('baseline')

        time = ts.time[:]
        nt = len(time)
        freq = ts.freq[:]
        nfreq = len(freq)
        # bl = ts.local_bl[:]

        # freq_window = min(nfreq/2, freq_window)
        # ensure window_size is an odd number
        if freq_window % 2 == 0:
            freq_window += 1
        if nfreq < 2*freq_window:
            warnings.warn('Not enough frequency points to do the smoothing')
            freq_flag = False
        else:
            freq_flag = True

        # time_window = min(nt/2, time_window)
        # ensure window_size is an odd number
        if time_window % 2 == 0:
            time_window += 1
        if nt < 2*time_window:
            warnings.warn('Not enough time points to do the smoothing')
            time_flag = False
        else:
            time_flag = True

        if isinstance(ts, RawTimestream):
            func = ts.bl_data_operate
        elif isinstance(ts, Timestream):
            func = ts.pol_and_bl_data_operate

        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']

        func(self.flag, full_data=True, show_progress=show_progress, progress_step=progress_step, keep_dist_axis=False, freq_flag=freq_flag, time_flag=time_flag)

        return super(Flag, self).process(ts)

    def flag(self, vis, vis_mask, li, gi, bl, ts, **kwargs):
        """Function that does the actual flag."""

        freq_window = self.params['freq_window']
        time_window = self.params['time_window']
        freq_sigma = self.params['freq_sigma']
        time_sigma = self.params['time_sigma']
        plot_fit = self.params['plot_fit']
        freq_fig_prefix = self.params['freq_fig_name']
        time_fig_prefix = self.params['time_fig_name']
        tag_output_iter = self.params['tag_output_iter']
        iteration = self.iteration
        freq_flag = kwargs.get('freq_flag')
        time_flag = kwargs.get('time_flag')

        time = ts.time[:]
        nt = len(time)
        freq = ts.freq[:]
        nfreq = len(freq)

        if isinstance(ts, Timestream): # for Timestream
            pol = bl[0]
            bl = tuple(bl[1])
        elif isinstance(ts, RawTimestream): # for RawTimestream
            pol = None
            bl = tuple(bl)
        else:
            raise ValueError('Need either a RawTimestream or Timestream')

        if freq_flag:
            # time integration
            tm_vis = np.ma.mean(np.ma.array(vis, mask=vis_mask), axis=0) # masked array
            abs_vis = np.ma.abs(tm_vis) # masked array
            if np.ma.count_masked(tm_vis) > 0: # has masked value
                abs_vis_valid = abs_vis[~abs_vis.mask]
                inds_valid = np.arange(nfreq)[~abs_vis.mask]
                itp = InterpolatedUnivariateSpline(inds_valid, abs_vis_valid)
                abs_vis_itp = itp(np.arange(nfreq))
                abs_vis1 = abs_vis_itp.copy()
            else:
                abs_vis1 = abs_vis.data.copy() # convert to ordinary array

            for cnt in xrange(10):
                if cnt != 0:
                    abs_vis1[inds] = smooth[inds]
                smooth = savitzky_golay(abs_vis1, freq_window, 3)

                # flage RFI
                diff = abs_vis1 - smooth
                median = np.median(diff)
                abs_diff = np.abs(diff - median)
                mad = np.median(abs_diff) / 0.6745
                inds = np.where(abs_diff > freq_sigma*mad)[0] # masked inds
                if len(inds) == 0:
                    break

            diff = abs_vis - smooth
            median = np.median(diff)
            abs_diff = np.abs(diff - median)
            mad = np.median(abs_diff) / 0.6745
            inds = np.where(abs_diff > freq_sigma*mad)[0] # masked inds
            vis_mask[:, inds] = True # set mask

            if plot_fit:
                plt.figure()
                plt.plot(freq, abs_vis, label='data')
                plt.plot(freq[inds], abs_vis[inds], 'ro', label='flag')
                plt.plot(freq, smooth, label='smooth')
                plt.xlabel(r'$\nu$ / MHz')
                plt.legend(loc='best')
                if pol is None:
                    fig_name = '%s_%d_%d.png' % (freq_fig_prefix, bl[0], bl[1])
                else:
                    fig_name = '%s_%d_%d_%s.png' % (freq_fig_prefix, bl[0], bl[1], pol)
                if tag_output_iter:
                    fig_name = output_path(fig_name, iteration=iteration)
                else:
                    fig_name = output_path(fig_name)
                plt.savefig(fig_name)
                plt.close()

        if time_flag:
            # freq integration
            fm_vis = np.ma.mean(np.ma.array(vis, mask=vis_mask), axis=1)
            abs_vis = np.ma.abs(fm_vis)
            if np.ma.count_masked(fm_vis) > 0: # has masked value
                abs_vis_valid = abs_vis[~abs_vis.mask]
                inds_valid = np.arange(nt)[~abs_vis.mask]
                itp = InterpolatedUnivariateSpline(inds_valid, abs_vis_valid)
                abs_vis_itp = itp(np.arange(nt))
                abs_vis1 = abs_vis_itp.copy()
            else:
                abs_vis1 = abs_vis.data.copy()

            for cnt in xrange(10):
                if cnt != 0:
                    abs_vis1[inds] = smooth[inds]
                smooth = savitzky_golay(abs_vis1, time_window, 3)

                # flage RFI
                diff = abs_vis1 - smooth
                median = np.median(diff)
                abs_diff = np.abs(diff - median)
                mad = np.median(abs_diff) / 0.6745
                inds = np.where(abs_diff > time_sigma*mad)[0] # masked inds
                if len(inds) == 0:
                    break

            diff = abs_vis - smooth
            median = np.median(diff)
            abs_diff = np.abs(diff - median)
            mad = np.median(abs_diff) / 0.6745
            inds = np.where(abs_diff > time_sigma*mad)[0] # masked inds
            # Addtional threshold
            # inds1 = np.where(np.abs(diff[inds]) > 1.0e-2*np.abs(smooth[inds]))[0]
            # inds = inds[inds1]
            vis_mask[inds] = True # set mask

            if plot_fit:
                plt.figure()
                plt.plot(time, abs_vis, label='data')
                plt.plot(time[inds], abs_vis[inds], 'ro', label='flag')
                plt.plot(time, smooth, label='smooth')
                plt.xlabel(r'$t$ / Julian Date')
                plt.legend(loc='best')
                if pol is None:
                    fig_name = '%s_%d_%d.png' % (time_fig_prefix, bl[0], bl[1])
                else:
                    fig_name = '%s_%d_%d_%s.png' % (time_fig_prefix, bl[0], bl[1], pol)
                if tag_output_iter:
                    fig_name = output_path(fig_name, iteration=iteration)
                else:
                    fig_name = output_path(fig_name)
                plt.savefig(fig_name)
                plt.close()
