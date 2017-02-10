"""Local median fit method.

Inheritance diagram
-------------------

.. inheritance-diagram:: LocalMedianFit tlpipe.rfi.local_average_fit.LocalAverageFit tlpipe.rfi.local_minimum_fit.LocalMinimumFit
   :parts: 1

"""

import local_fit
import numpy as np


class LocalMedianFit(local_fit.LocalFitMethod):
    """Local median fit method.

    In this method, the background value is caculated by the local median of a
    sliding window of size :math:`N \\times M` around each data value.

    """

    def _calculate(self, x, y, start_x, end_x, start_y, end_y):

        vis = np.ma.array(self.vis[start_y:end_y, start_x:end_x], mask=self.vis_mask[start_y:end_y, start_x:end_x])

        if vis.count() > 0:
            return np.ma.median(vis)
        else:
            return self.vis[y, x]