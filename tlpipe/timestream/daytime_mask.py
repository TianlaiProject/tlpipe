"""Daytime data mask.

Inheritance diagram
-------------------

.. inheritance-diagram:: Mask
   :parts: 2

"""

import numpy as np
import tod_task


class Mask(tod_task.TaskTimestream):
    """Daytime data mask."""

    params_init = {
                    'mask_time_range': [8.0, 19.5], # hour
                  }

    prefix = 'dm_'

    def process(self, ts):

        mask_time_range = self.params['mask_time_range']

        local_hour = ts['local_hour'].local_data
        day_inds = np.where(np.logical_and(local_hour>=mask_time_range[0], local_hour<=mask_time_range[1]))[0]
        ts.local_vis_mask[day_inds] = True # do not change vis directly

        ts.add_history(self.history)

        # ts.info()

        return ts
