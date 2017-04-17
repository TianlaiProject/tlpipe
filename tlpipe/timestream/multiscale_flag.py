"""RFI flagging.

Inheritance diagram
-------------------

.. inheritance-diagram:: Flag
   :parts: 2

"""

import numpy as np
import timestream_task
from tlpipe.container.raw_timestream import RawTimestream
from tlpipe.container.timestream import Timestream
from tlpipe.rfi import interpolate
from tlpipe.rfi import sum_threshold
from tlpipe.utils import multiscale
from tlpipe.utils.path_util import output_path


class Flag(timestream_task.TimestreamTask):
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
                    'flag_direction': ('time', 'freq'),
                  }

    prefix = 'mf_'

    def process(self, ts):

        ts.redistribute('baseline')

        if isinstance(ts, RawTimestream):
            func = ts.bl_data_operate
        elif isinstance(ts, Timestream):
            func = ts.pol_and_bl_data_operate

        func(self.flag, full_data=True, keep_dist_axis=False)

        return super(Flag, self).process(ts)

    def flag(self, vis, vis_mask, li, gi, fb, ts, **kwargs):
        """Function that does the actual flag."""

        # if all have been masked, no need to flag again
        if vis_mask.all():
            return

        first_threshold = self.params['first_threshold']
        exp_factor = self.params['exp_factor']
        distribution = self.params['distribution']
        max_threshold_len = self.params['max_threshold_len']
        sensitivity = self.params['sensitivity']
        min_connected = self.params['min_connected']
        flag_direction = self.params['flag_direction']

        vis_abs = np.abs(vis) # operate only on the amplitude

        # first complete masked vals due to ns by interpolate
        itp = interpolate.Interpolate(vis_abs, vis_mask)
        background = itp.fit()
        background1 = background.copy()

        # nt, nf = background.shape
        # for fi in range(nf):
        #     background[:, fi] = multiscale.median_wavelet_smooth(background[:, fi], level=4)
        # for ti in range(nt):
        #     if fb[0] == fb[1]:
        #         background[ti, :] = multiscale.median_wavelet_smooth(background[ti, :], level=2)
        #     else:
        #         background[ti, :] = multiscale.median_wavelet_smooth(background[ti, :], level=3)
        background[:] = multiscale.median_wavelet_smooth(background, level=2)

        # sum-threshold
        # vis_diff = vis_abs - background
        vis_diff = background1 - background
        # an initial run of N = 1 only to remove extremely high amplitude RFI
        st = sum_threshold.SumThreshold(vis_diff, vis_mask, first_threshold, exp_factor, distribution, 1, min_connected)
        st.execute(sensitivity, flag_direction)

        # if all have been masked, no need to flag again
        if st.vis_mask.all():
            vis_mask[:] = st.vis_mask
            return

        vis_diff = np.where(st.vis_mask, 0, vis_diff)

        st = sum_threshold.SumThreshold(vis_diff, st.vis_mask, first_threshold, exp_factor, distribution, max_threshold_len, min_connected)
        st.execute(sensitivity, flag_direction)

        vis_mask[:] = st.vis_mask
