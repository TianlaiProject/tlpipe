"""Local minimum fit method.

Inheritance diagram
-------------------

.. inheritance-diagram:: LocalMinimumFit tlpipe.rfi.local_median_fit.LocalMedianFit tlpipe.rfi.local_average_fit.LocalAverageFit
   :parts: 1

"""

import local_fit
import numpy as np


class LocalMinimumFit(local_fit.LocalFitMethod):
    """Local minimum fit method.

    In this method, the background value is caculated by the local minimum of a
    sliding window of size :math:`N \\times M` around each data value.

    """

    def _calculate(self, x, y, start_x, end_x, start_y, end_y):

        vis = np.ma.array(self.vis[start_y:end_y, start_x:end_x], mask=self.vis_mask[start_y:end_y, start_x:end_x])

        if vis.count() > 0:
            return np.ma.min(vis)
        else:
            return self.vis[y, x]