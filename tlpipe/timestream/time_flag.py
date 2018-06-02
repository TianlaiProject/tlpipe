"""Exceptional values flagging along the time axis.

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


class Flag(timestream_task.TimestreamTask):
    """Exceptional values flagging along the time axis.

    This flags the data along the time axis by comparing the data with its
    smoothing, its difference that exceed the given threshold will be masked.

    """

    params_init = {
                    'time_window': 15,
                    'sigma': 5.0,
                  }

    prefix = 'tf_'

    def process(self, ts):

        time_window = self.params['time_window']

        nt = ts.time.shape[0] # global shape

        # time_window = min(nt/2, time_window)
        # ensure window_size is an odd number
        if time_window % 2 == 0:
            time_window += 1
        if nt >= 2*time_window:

            ts.redistribute('baseline')

            if isinstance(ts, RawTimestream):
                func = ts.freq_and_bl_data_operate
            elif isinstance(ts, Timestream):
                func = ts.freq_pol_and_bl_data_operate

            show_progress = self.params['show_progress']
            progress_step = self.params['progress_step']

            func(self.flag, full_data=True, show_progress=show_progress, progress_step=progress_step, keep_dist_axis=False)
        else:
            warnings.warn('Not enough time points to do the smoothing')

        return super(Flag, self).process(ts)

    def flag(self, vis, vis_mask, li, gi, tbl, ts, **kwargs):
        """Function that does the actual flag."""

        sigma = self.params['sigma']
        time_window = self.params['time_window']

        nt = vis.shape[0]
        abs_vis = np.ma.abs(np.ma.array(vis, mask=vis_mask))
        # mask all if valid values less than the given threshold
        if abs_vis.count() < 0.1 * nt or abs_vis.count() <= 3:
            vis_mask[:] = True
            return

        if np.ma.count_masked(abs_vis) > 0: # has masked value
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
            inds = np.where(abs_diff > sigma*mad)[0] # masked inds
            if len(inds) == 0:
                break

        diff = abs_vis - smooth
        median = np.median(diff)
        abs_diff = np.abs(diff - median)
        mad = np.median(abs_diff) / 0.6745
        inds = np.where(abs_diff > sigma*mad)[0] # masked inds
        # Addtional threshold
        # inds1 = np.where(np.abs(diff[inds]) > 1.0e-2*np.abs(smooth[inds]))[0]
        # inds = inds[inds1]
        vis_mask[inds] = True # set mask
