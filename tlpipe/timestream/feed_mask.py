"""Mask the data involving the specified feeds.

Inheritance diagram
-------------------

.. inheritance-diagram:: Mask
   :parts: 2

"""

import numpy as np
import ephem
import aipy as a
from . import timestream_task
from tlpipe.container.timestream import Timestream


class Mask(timestream_task.TimestreamTask):
    """Mask the data involving the specified feeds."""

    params_init = {
                    'feeds': [] # in format [ (feedno, pol), (3, 'x'), (5, 'y), ... ]
                  }

    prefix = 'fm_'

    def process(self, ts):

        assert isinstance(ts, Timestream), '%s only works for Timestream object, use %s instead for RawTimestream object' % (self.__class__.__name__, 'channel_mask.py')

        feeds = self.params['feeds']

        for feedno, pol in feeds:
            for i in [0, 1]:
                bis = np.where(ts.local_bl[:, i] == feedno)[0].tolist()
                # bis = list(set(bis)) # make unique
                pis = []
                for pi, lp in enumerate(ts.local_pol):
                    if pol == ts.pol_dict[lp][i]:
                        pis.append(pi)

                for bi in bis:
                    for pi in pis:
                        # print 'mask for bl = (%d, %d), pol = %s' % (ts.local_bl[bi, 0], ts.local_bl[bi, 1], ts.pol_dict[ts.local_pol[pi]])
                        ts.local_vis_mask[:, :, pi, bi] = True


        return super(Mask, self).process(ts)
