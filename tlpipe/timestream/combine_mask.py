"""Combine RFI masks of all four polarizations.

Inheritance diagram
-------------------

.. inheritance-diagram:: Combine
   :parts: 2

"""

import numpy as np
import tod_task
from timestream import Timestream


class Combine(tod_task.TaskTimestream):
    """Combine RFI masks of all four polarizations.

    If a sample in one polarization is flagged, the sample will be flagged
    in all polarizations. This is done by stacking the masks of the XX, YY,
    XY and YX polarizations.

    """

    prefix = 'cm_'

    def process(self, ts):

        assert isinstance(ts, Timestream), '%s only works for Timestream object' % self.__class__.__name__

        if ts.dist_axis_name == 'polarization':
            ts.redistribute('baseline')

        ts.bl_data_operate(self.combine)

        return super(Combine, self).process(ts)

    def combine(self, vis, vis_mask, li, gi, tf, ts, **kwargs):
        """Function that does the combine operation."""

        vis_mask[:] = np.sum(vis_mask, axis=2).astype(bool)[:, :, np.newaxis]
