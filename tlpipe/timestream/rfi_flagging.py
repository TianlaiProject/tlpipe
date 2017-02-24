"""RFI flagging.

Inheritance diagram
-------------------

.. inheritance-diagram:: Flag
   :parts: 2

"""

import numpy as np
import tod_task
from raw_timestream import RawTimestream
from timestream import Timestream
from tlpipe.rfi import interpolate
from tlpipe.rfi import gaussian_filter
from tlpipe.rfi import sum_threshold


class Flag(tod_task.TaskTimestream):
    """RFI flagging.

    RFI flagging by using the SumThreshold method.

    """

    params_init = {
                    'first_threshold': 6.0,
                    'exp_factor': 1.5,
                    'distribution': 'Rayleigh',
                    'max_threshold_len': 1024,
                    'sensitivity': 1.0,
                    'min_connected': 1,
                    'tk_size': 1.0,
                    'fk_size': 3.0,
                    'threshold_num': 2, # number of threshold
                  }

    prefix = 'rf_'

    def process(self, ts):

        ts.redistribute('baseline')

        if isinstance(ts, RawTimestream):
            func = ts.bl_data_operate
        elif isinstance(ts, Timestream):
            func = ts.pol_and_bl_data_operate

        func(self.flag, full_data=True, keep_dist_axis=False)

        return super(Flag, self).process(ts)

    def flag(self, vis, vis_mask, li, gi, tf, ts, **kwargs):
        """Function that does the actual flag."""

        first_threshold = self.params['first_threshold']
        exp_factor = self.params['exp_factor']
        distribution = self.params['distribution']
        max_threshold_len = self.params['max_threshold_len']
        sensitivity = self.params['sensitivity']
        min_connected = self.params['min_connected']
        tk_size = self.params['tk_size']
        fk_size = self.params['fk_size']
        threshold_num = max(0, int(self.params['threshold_num']))

        vis_abs = np.abs(vis) # operate only on the amplitude

        # first round
        # first complete masked vals due to ns by interpolate
        itp = interpolate.Interpolate(vis_abs, vis_mask)
        background = itp.fit()
        # Gaussian fileter
        gf = gaussian_filter.GaussianFilter(background, time_kernal_size=tk_size, freq_kernal_size=fk_size)
        background = gf.fit()
        # sum-threshold
        vis_diff = vis_abs - background
        # an initial run of N = 1 only to remove extremely high amplitude RFI
        st = sum_threshold.SumThreshold(vis_diff, vis_mask, first_threshold, exp_factor, distribution, 1, min_connected)
        st.execute(sensitivity)

        # next rounds
        for i in xrange(threshold_num):
            # Gaussian fileter
            gf = gaussian_filter.GaussianFilter(vis_diff, st.vis_mask, time_kernal_size=tk_size, freq_kernal_size=fk_size)
            background = gf.fit()
            # sum-threshold
            vis_diff = vis_diff - background
            st = sum_threshold.SumThreshold(vis_diff, st.vis_mask, first_threshold, exp_factor, distribution, max_threshold_len, min_connected)
            st.execute(sensitivity)

        # replace vis_mask with the flagged mask
        vis_mask[:] = st.vis_mask
