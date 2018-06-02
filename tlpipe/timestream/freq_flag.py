"""Exceptional values flagging along the frequency axis.

Inheritance diagram
-------------------

.. inheritance-diagram:: Flag
   :parts: 2

"""

import warnings
import numpy as np
import timestream_task
from tlpipe.container.raw_timestream import RawTimestream
from tlpipe.container.timestream import Timestream


class Flag(timestream_task.TimestreamTask):
    """Exceptional values flagging along the frequency axis.

    This task does a very simple flagging by masking data points whose
    absolute value exceed a given threshold (in unit of std of the data)
    along the frequency axis.

    """

    params_init = {
                    'sigma': 3.0,
                    'freq_points': 10, # minima freq point to do the flag
                  }

    prefix = 'ff_'

    def process(self, ts):

        freq_points = self.params['freq_points']

        nfreq = ts.freq.shape[0] # global shape
        if nfreq >= freq_points:

            ts.redistribute('time')

            if isinstance(ts, RawTimestream):
                func = ts.time_and_bl_data_operate
            elif isinstance(ts, Timestream):
                func = ts.time_pol_and_bl_data_operate

            show_progress = self.params['show_progress']
            progress_step = self.params['progress_step']

            func(self.flag, full_data=True, show_progress=show_progress, progress_step=progress_step, keep_dist_axis=False)
        else:
            warnings.warn('Not enough frequency points to do the flag')

        return super(Flag, self).process(ts)

    def flag(self, vis, vis_mask, li, gi, tbl, ts, **kwargs):
        """Function that does the actual flag."""

        sigma = self.params['sigma']
        freq_points = self.params['freq_points']

        vis_abs = np.ma.abs(np.ma.array(vis, mask=vis_mask))
        if vis_abs.count() >= freq_points:
            median = np.ma.median(vis_abs)
            abs_diff = np.ma.abs(vis_abs - median)
            mad = np.ma.median(abs_diff) / 0.6745
            inds = np.ma.where(abs_diff > sigma*mad)[0] # masked inds
            vis_mask[inds] = True # set mask
