"""Combine RFI masks of all four polarizations.

Inheritance diagram
-------------------

.. inheritance-diagram:: Combine
   :parts: 2

"""

import numpy as np
import timestream_task
from tlpipe.container.timestream import Timestream


class Combine(timestream_task.TimestreamTask):
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

        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']

        ts.bl_data_operate(self.combine, show_progress=show_progress, progress_step=progress_step,)

        # set flag to indicate the combination
        ts['vis_mask'].attrs['combined_mask'] = True

        return super(Combine, self).process(ts)

    def combine(self, vis, vis_mask, li, gi, tf, ts, **kwargs):
        """Function that does the combine operation."""

        vis_mask[:] = np.sum(vis_mask, axis=2).astype(bool)[:, :, np.newaxis]
