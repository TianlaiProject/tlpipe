"""Exceptional values flagging along the frequency axis.

Inheritance diagram
-------------------

.. inheritance-diagram:: Flag
   :parts: 2

"""

import warnings
import numpy as np
import tod_task
from raw_timestream import RawTimestream
from timestream import Timestream


class Flag(tod_task.TaskTimestream):
    """Exceptional values flagging along the frequency axis."""

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

            func(self.flag, full_data=True, keep_dist_axis=False)
        else:
            warnings.warn('Not enough frequency points to do the flag')

        ts.add_history(self.history)

        # ts.info()

        return ts

    def flag(self, vis, vis_mask, li, gi, tbl, ts, **kwargs):
        """Function that does the actual flag."""

        sigma = self.params['sigma']
        freq_points = self.params['freq_points']

        vis_abs = np.abs(np.ma.array(vis, mask=vis_mask))
        if vis_abs.count() >= freq_points:
            mean = np.ma.mean(vis_abs)
            std = np.ma.std(vis_abs)
            inds = np.where(np.abs(vis_abs - mean) > sigma*std)[0] # masked inds
            vis_mask[inds] = True # set mask

        return vis, vis_mask
