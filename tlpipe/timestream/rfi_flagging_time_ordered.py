"""RFI flagging.

Inheritance diagram
-------------------

.. inheritance-diagram:: Flag
   :parts: 2

"""

import numpy as np
from caput import mpiarray
from caput import mpiutil
from . import timestream_task
from tlpipe.container.raw_timestream import RawTimestream
from tlpipe.container.timestream import Timestream
from tlpipe.rfi import interpolate
from tlpipe.rfi import gaussian_filter
from tlpipe.rfi import sum_threshold
from tlpipe.rfi import sir_operator
from tlpipe.utils import progress


class Flag(timestream_task.TimestreamTask):
    """RFI flagging.

    RFI flagging by using the SumThreshold method.

    """

    params_init = {
                    'first_threshold': 12.0,
                    'exp_factor': 1.5,
                    'distribution': 'Rayleigh',
                    'max_threshold_len': 1024,
                    'sensitivity': 1.0,
                    'min_connected': 1,
                    'flag_direction': ('time', 'freq'),
                    'tk_size': 1.0, # 128.0 for dish
                    'fk_size': 3.0, # 2.0 for dish
                    'threshold_num': 2, # number of threshold
                    'eta': 0.2,
                  }

    prefix = 'rf_'

    def process(self, ts):

        via_memmap = self.params['via_memmap']
        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']

        ts.redistribute('time', via_memmap=via_memmap)

        if isinstance(ts, RawTimestream):
            redistribute_axis = 2
        elif isinstance(ts, Timestream):
            redistribute_axis = 3

        if 'ns_on' in ts.keys():
            ns_on = mpiutil.gather_array(ts['ns_on'].local_data, root=None, comm=ts.comm)
        else:
            ns_on = None
        nbl = len(ts.bl) # number of baselines
        n, r = nbl // mpiutil.size, nbl % mpiutil.size # number of iterations
        if r != 0:
            n = n + 1
        if show_progress and mpiutil.rank0:
            pg = progress.Progress(n, step=progress_step)
        for i in range(n):
            if show_progress and mpiutil.rank0:
                pg.show(i)

            this_vis = ts.local_vis[..., i*mpiutil.size:(i+1)*mpiutil.size].copy()
            this_vis = mpiarray.MPIArray.wrap(this_vis, axis=0, comm=ts.comm).redistribute(axis=redistribute_axis)
            this_vis_mask = ts.local_vis_mask[..., i*mpiutil.size:(i+1)*mpiutil.size].copy()
            this_vis_mask = mpiarray.MPIArray.wrap(this_vis_mask, axis=0, comm=ts.comm).redistribute(axis=redistribute_axis)
            if this_vis.local_array.shape[-1] != 0:
                if isinstance(ts, RawTimestream):
                    self.flag(this_vis.local_array[..., 0], this_vis_mask.local_array[..., 0], 0, 0, 0, ts)
                    self.sir_operate(this_vis.local_array[..., 0], this_vis_mask.local_array[..., 0], 0, 0, 0, ts, ns_on=ns_on)
                elif isinstance(ts, Timestream):
                    for pi in range(this_vis.local_array.shape[2]):
                        self.flag(this_vis.local_array[..., pi, 0], this_vis_mask.local_array[..., pi, 0], 0, 0, 0, ts)
                        self.sir_operate(this_vis.local_array[..., pi, 0], this_vis_mask.local_array[..., pi, 0], 0, 0, 0, ts, ns_on=ns_on)
            # this_vis = this_vis.redistribute(axis=0)
            this_vis_mask = this_vis_mask.redistribute(axis=0)
            ts.local_vis_mask[..., i*mpiutil.size:(i+1)*mpiutil.size] = this_vis_mask.local_array

        return super(Flag, self).process(ts)

    def flag(self, vis, vis_mask, li, gi, tf, ts, **kwargs):
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
        tk_size = self.params['tk_size']
        fk_size = self.params['fk_size']
        threshold_num = max(0, int(self.params['threshold_num']))

        vis_abs = np.abs(vis) # operate only on the amplitude

        # first round
        # first complete masked vals due to ns by interpolate
        itp = interpolate.Interpolate(vis_abs, vis_mask)
        background = itp.fit()
        # Gaussian fileter
        gf = gaussian_filter.GaussianFilter(background, time_kernal_size=tk_size, freq_kernal_size=fk_size, filter_direction=flag_direction)
        background = gf.fit()
        # sum-threshold
        vis_diff = vis_abs - background
        # an initial run of N = 1 only to remove extremely high amplitude RFI
        st = sum_threshold.SumThreshold(vis_diff, vis_mask, first_threshold, exp_factor, distribution, 1, min_connected)
        st.execute(sensitivity, flag_direction)

        # if all have been masked, no need to flag again
        if st.vis_mask.all():
            vis_mask[:] = st.vis_mask
            return

        # next rounds
        for i in range(threshold_num):
            # Gaussian fileter
            gf = gaussian_filter.GaussianFilter(vis_diff, st.vis_mask, time_kernal_size=tk_size, freq_kernal_size=fk_size, filter_direction=flag_direction)
            background = gf.fit()
            # sum-threshold
            vis_diff = vis_diff - background
            st = sum_threshold.SumThreshold(vis_diff, st.vis_mask, first_threshold, exp_factor, distribution, max_threshold_len, min_connected)
            st.execute(sensitivity, flag_direction)

            # if all have been masked, no need to flag again
            if st.vis_mask.all():
                break

        # replace vis_mask with the flagged mask
        vis_mask[:] = st.vis_mask

    def sir_operate(self, vis, vis_mask, li, gi, tf, ts, **kwargs):
        """Function that does the actual operation."""

        eta = self.params['eta']

        has_ns = kwargs['ns_on'] is not None
        if has_ns:
            ns_on = kwargs['ns_on']

        if vis_mask.ndim == 2:
            mask = vis_mask.copy()
            if has_ns:
                mask[ns_on] = False
            mask = sir_operator.vertical_sir(mask, eta)
            if has_ns:
                mask[ns_on] = True
            vis_mask[:] = sir_operator.horizontal_sir(mask, eta)
        elif vis_mask.ndim == 3:
            # This shold be done after the combination of all pols
            mask = vis_mask[:, :, 0].copy()
            if has_ns:
                mask[ns_on] = False
            mask = sir_operator.vertical_sir(mask, eta)
            if has_ns:
                mask[ns_on] = True
            vis_mask[:] = sir_operator.horizontal_sir(mask, eta)[:, :, np.newaxis]
        else:
            raise RuntimeError('Invalid shape of vis_mask: %s' % vis_mask.shape)