"""Night time mean subtract for the visibilities.

Inheritance diagram
-------------------

.. inheritance-diagram:: Subtract
   :parts: 2

"""

import numpy as np
from . import timestream_task
from tlpipe.container.raw_timestream import RawTimestream
from tlpipe.container.timestream import Timestream
from caput import mpiutil


class Subtract(timestream_task.TimestreamTask):
    """Night time mean subtract for the visibilities.

    """

    params_init = {
                    'time_range': [21.5, 5.5] # [t1, t2], local hour, use the mean of t1 < t < t2 if t1 < t2 or {t1 < t < 24.0 and 0.0 < t < t2} if t1 > t2
    }

    prefix = 'su_'

    def process(self, ts):

        ts.redistribute('baseline')

        if isinstance(ts, RawTimestream):
            func = ts.bl_data_operate
        elif isinstance(ts, Timestream):
            func = ts.pol_and_bl_data_operate

        func(self.operate, full_data=True)

        return super(Subtract, self).process(ts)

    def operate(self, vis, vis_mask, li, gi, tf, ts, **kwargs):
        """Function that does the actual operation."""

        t1, t2 = self.params['time_range']

        local_hour = ts['local_hour'].local_data

        if t1 <= t2:
            tis = np.where(np.logical_and(local_hour>=t1, local_hour<=t2))[0]
        else:
            tis1 = np.where(np.logical_and(local_hour>=t1, local_hour<=24.0))[0]
            tis2 = np.where(np.logical_and(local_hour>=0.0, local_hour<=t2))[0]
            tis = np.concatenate([tis1, tis2])

        mean = np.ma.array(vis[tis], mask=vis_mask[tis]).mean(axis=0)

        vis -= mean[np.newaxis, :]
