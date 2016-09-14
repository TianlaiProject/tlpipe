"""Line RFI flagging."""

import os
import warnings
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import tod_task
from sg_filter import savitzky_golay
from caput import mpiarray

class Flag(tod_task.IterRawTimestream):
    """Line RFI flagging."""

    params_init = {
                    'freq_window': 15,
                    'time_window': 15,
                    'freq_sigma': 2.0,
                    'time_sigma': 7.0,
                    'plot_fit': False, # plot the smoothing fit
                    'freq_fig_name': 'rfi_freq',
                    'time_fig_name': 'rfi_time',
                  }

    prefix = 'lf_'

    def process(self, rt):

        freq_window = self.params['freq_window']
        time_window = self.params['time_window']
        freq_sigma = self.params['freq_sigma']
        time_sigma = self.params['time_sigma']
        plot_fit = self.params['plot_fit']
        freq_fig_prefix = self.params['freq_fig_name']
        time_fig_prefix = self.params['time_fig_name']

        rt.redistribute('baseline')

        time = rt.time[:]
        nt = len(time)
        freq = rt.freq[:]
        nfreq = len(freq)
        bl = rt.local_bl[:]

        # time integration
        tm_vis = np.ma.mean(np.ma.array(rt.local_vis, mask=rt.local_vis_mask), axis=0)
        freq_window = min(nfreq/2, freq_window)
        # ensure window_size is an odd number
        if freq_window % 2 == 0:
            freq_window += 1
        if nfreq < 2*freq_window:
            warnings.warn('Not enough frequency points to do the smoothing')
        else:
            # iterate over local baselines
            for lbi in range(len(bl)):
                abs_vis = np.abs(tm_vis[:, lbi])
                abs_vis1 = abs_vis.copy()

                for cnt in range(10):
                    # abs_vis1 = abs_vis.copy()
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
                rt.local_vis_mask[:, inds, lbi] = True # set mask

                if plot_fit:
                    import tlpipe.plot
                    import matplotlib.pyplot as plt
                    from tlpipe.utils.path_util import output_path

                    plt.figure()
                    plt.plot(freq, abs_vis, label='data')
                    plt.plot(freq[inds], abs_vis[inds], 'ro', label='flag')
                    plt.plot(freq, smooth, label='smooth')
                    plt.xlabel(r'$\nu$ / MHz')
                    plt.legend(loc='best')
                    fig_name = '%s_%d_%d.png' % (freq_fig_prefix, bl[lbi][0], bl[lbi][1])
                    fig_name = output_path(fig_name)
                    plt.savefig(fig_name)
                    plt.clf()

        # freq integration
        fm_vis = np.ma.mean(np.ma.array(rt.local_vis, mask=rt.local_vis_mask), axis=1)
        time_window = min(nt/2, time_window)
        # ensure window_size is an odd number
        if time_window % 2 == 0:
            time_window += 1
        if nt < 2*time_window:
            warnings.warn('Not enough time points to do the smoothing')
        else:
            # iterate over local baselines
            for lbi in range(len(bl)):
                abs_vis = np.abs(fm_vis[:, lbi])
                abs_vis_valid = abs_vis[~abs_vis.mask]
                inds_valid = np.arange(nt)[~abs_vis.mask]
                itp = InterpolatedUnivariateSpline(inds_valid, abs_vis_valid)
                abs_vis_itp = itp(np.arange(nt))
                abs_vis1 = abs_vis_itp.copy()

                for cnt in range(10):
                    # abs_vis1 = abs_vis_itp.copy()
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
                rt.local_vis_mask[inds, :, lbi] = True # set mask

                if plot_fit:
                    import tlpipe.plot
                    import matplotlib.pyplot as plt
                    from tlpipe.utils.path_util import output_path

                    plt.figure()
                    plt.plot(time, abs_vis, label='data')
                    plt.plot(time[inds], abs_vis[inds], 'ro', label='flag')
                    plt.plot(time, smooth, label='smooth')
                    plt.xlabel(r'$t$ / Julian Date')
                    plt.legend(loc='best')
                    fig_name = '%s_%d_%d.png' % (time_fig_prefix, bl[lbi][0], bl[lbi][1])
                    fig_name = output_path(fig_name)
                    plt.savefig(fig_name)
                    plt.clf()


        rt.add_history(self.history)

        # rt.info()

        return rt
