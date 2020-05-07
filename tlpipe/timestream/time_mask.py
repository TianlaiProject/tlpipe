"""Mask the given time sections.

Inheritance diagram
-------------------

.. inheritance-diagram:: Mask
   :parts: 2

"""

from datetime import datetime
import numpy as np
import timestream_task
from tlpipe.utils import date_util


class Mask(timestream_task.TimestreamTask):
    """Mask the given time sections."""

    params_init = {
                    'mask_time_range': None # or [ ('yyyymmddhhMMss', 'yyyymmddhhMMss'),  ('yyyymmddhhMMss', 'yyyymmddhhMMss')], in UTC+08h
                  }

    prefix = 'tm_'

    def process(self, ts):

        mask_time_range = self.params['mask_time_range']

        if mask_time_range is not None:
            for st, et in mask_time_range:
                st1 = date_util.get_juldate(datetime.strptime(st, '%Y%m%d%H%M%S'), tzone='UTC+08h')
                et1 = date_util.get_juldate(datetime.strptime(et, '%Y%m%d%H%M%S'), tzone='UTC+08h')

                ts.local_vis_mask[np.logical_and(ts.local_time>=st1, ts.local_time<=et1)] = True # do not change vis directly


        return super(Mask, self).process(ts)
