"""Mask the data involving the specified channels.

Inheritance diagram
-------------------

.. inheritance-diagram:: Mask
   :parts: 2

"""

import numpy as np
import ephem
import aipy as a
from . import timestream_task
from tlpipe.container.raw_timestream import RawTimestream


class Mask(timestream_task.TimestreamTask):
    """Mask the data involving the specified channels."""

    params_init = {
                    'channels': [] # in format [ channo1, channo2, ... ]
                  }

    prefix = 'cm_'

    def process(self, ts):

        assert isinstance(ts, RawTimestream), '%s only works for RawTimestream object, use %s instead for Timestream object' % (self.__class__.__name__, 'feed_mask.py')

        channels = self.params['channels']

        for channo in channels:
            for bi, (ch1, ch2) in enumerate(ts.local_bl):
                if channo == ch1 or channo == ch2:
                    # print 'mask for bi = %d, (%d, %d)' % (bi, ch1, ch2)
                    ts.local_vis_mask[:, :, bi] = True


        return super(Mask, self).process(ts)
