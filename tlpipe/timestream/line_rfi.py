"""Line RFI flagging.

Inheritance diagram
-------------------

.. inheritance-diagram:: Flag
   :parts: 2

"""

import warnings
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import tod_task
from raw_timestream import RawTimestream
from timestream import Timestream
from sg_filter import savitzky_golay
from tlpipe.utils.path_util import output_path
import tlpipe.plot
import matplotlib.pyplot as plt


def flag(vis, vis_mask, li, gi, bl, ts, **kwargs):

    freq_window = kwargs.get('freq_window', 15)
    time_window = kwargs.get('time_window', 15)
    freq_sigma = kwargs.get('freq_sigma', 2.0)
    time_sigma = kwargs.get('time_sigma', 5.0)
    plot_fit = kwargs.get('plot_fit', 5.0)
    freq_fig_prefix = kwargs.get('freq_fig_prefix', 'rfi_freq')
    time_fig_prefix = kwargs.get('time_fig_prefix', 'rfi_time')
    tag_output_iter = kwargs.get('tag_output_iter', True)
    iteration = kwargs.get('iteration', None)
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
        tm_vis = np.ma.mean(np.ma.array(vis, mask=vis_mask), axis=0)
        abs_vis = np.abs(tm_vis)
        if np.ma.count_masked(tm_vis) > 0: # has masked value
            abs_vis_valid = abs_vis[~abs_vis.mask]
            inds_valid = np.arange(nfreq)[~abs_vis.mask]
            itp = InterpolatedUnivariateSpline(inds_valid, abs_vis_valid)
            abs_vis_itp = itp(np.arange(nfreq))
            abs_vis1 = abs_vis_itp.copy()
        else:
            abs_vis1 = abs_vis.copy()

        for cnt in xrange(10):
            if cnt != 0:
                abs_vis1[inds] = smooth[inds]
            smooth = savitzky_golay(abs_vis1, freq_window, 3)

            # flage RFI
            diff = abs_vis1 - smooth
            mean = np.mean(diff)
            std = np.std(diff)
            inds = np.where(np.abs(diff - mean) > freq_sigma*std)[0]
            if len(inds) == 0:
                break

        diff = abs_vis - smooth
        mean = np.mean(diff)
        std = np.std(diff)
        inds = np.where(np.abs(diff - mean) > freq_sigma*std)[0] # masked inds
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
        abs_vis = np.abs(fm_vis)
        if np.ma.count_masked(fm_vis) > 0: # has masked value
            abs_vis_valid = abs_vis[~abs_vis.mask]
            inds_valid = np.arange(nt)[~abs_vis.mask]
            itp = InterpolatedUnivariateSpline(inds_valid, abs_vis_valid)
            abs_vis_itp = itp(np.arange(nt))
            abs_vis1 = abs_vis_itp.copy()
        else:
            abs_vis1 = abs_vis.copy()

        for cnt in xrange(10):
            if cnt != 0:
                abs_vis1[inds] = smooth[inds]
            smooth = savitzky_golay(abs_vis1, time_window, 3)

            # flage RFI
            diff = abs_vis1 - smooth
            mean = np.mean(diff)
            std = np.std(diff)
            inds = np.where(np.abs(diff - mean) > time_sigma*std)[0]
            if len(inds) == 0:
                break

        diff = abs_vis - smooth
        mean = np.mean(diff)
        std = np.std(diff)
        inds = np.where(np.abs(diff - mean) > time_sigma*std)[0] # masked inds
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

    return vis, vis_mask


class Flag(tod_task.TaskTimestream):
    """Line RFI flagging."""

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
        freq_sigma = self.params['freq_sigma']
        time_sigma = self.params['time_sigma']
        plot_fit = self.params['plot_fit']
        freq_fig_prefix = self.params['freq_fig_name']
        time_fig_prefix = self.params['time_fig_name']
        tag_output_iter = self.params['tag_output_iter']

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

        func(flag, full_data=True, keep_dist_axis=False, freq_window=freq_window, time_window=time_window, freq_sigma=freq_sigma, time_sigma=time_sigma, plot_fit=plot_fit, freq_fig_prefix=freq_fig_prefix, time_fig_prefix=time_fig_prefix, tag_output_iter=tag_output_iter, freq_flag=freq_flag, time_flag=time_flag, iteration=self.iteration)

        ts.add_history(self.history)

        # ts.info()

        return ts
