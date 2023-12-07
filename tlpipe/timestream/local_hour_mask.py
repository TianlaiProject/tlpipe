"""Mask daytime data according to local hour.

Inheritance diagram
-------------------

.. inheritance-diagram:: Mask
   :parts: 2

"""

import numpy as np
from caput import mpiarray
from . import timestream_task


class Mask(timestream_task.TimestreamTask):
    """Mask daytime data according to local hour."""

    params_init = {
                    'exp_factor': 6.0,
                  }

    prefix = 'lm_'

    def process(self, ts):

        exp_factor = self.params['exp_factor']

        local_hour = ts['local_hour'].local_data
        h = local_hour

        f1 = lambda h: 1.0 / (1.0 + np.exp(exp_factor * (h - 8.0)))
        f2 = lambda h: 1.0 - 1.0 / (1.0 + np.exp(exp_factor * (h - 20.0)))

        local_hour_factor = np.where(h < 14.0, f1(h), f2(h))
        local_hour_factor[local_hour_factor<1.0e-6] = 0.0
        local_hour_factor[local_hour_factor>(1.0 - 1.0e-6)] = 1.0

        if 'time' == ts.main_data_axes[ts.main_data_dist_axis]:
            local_hour_factor = mpiarray.MPIArray.wrap(local_hour_factor, axis=0)
        else:
            local_hour_factor = mpiarray.MPIArray.from_numpy_array(local_hour_factor, axis=0)
        ts.create_main_time_ordered_dataset('local_hour_factor', data=local_hour_factor)


        return super(Mask, self).process(ts)
