"""Local average fit method.

Inheritance diagram
-------------------

.. inheritance-diagram:: LocalAverageFit tlpipe.rfi.local_median_fit.LocalMedianFit tlpipe.rfi.local_minimum_fit.LocalMinimumFit
   :parts: 1

"""

import local_fit
import numpy as np


class LocalAverageFit(local_fit.LocalFitMethod):
    """Local average fit method.

    In this method, the background value is caculated by the local average of a
    sliding window of size :math:`N \\times M` around each data value.

    """

    def _calculate(self, x, y, start_x, end_x, start_y, end_y):

        vis = np.ma.array(self.vis[start_y:end_y, start_x:end_x], mask=self.vis_mask[start_y:end_y, start_x:end_x])

        if vis.count() > 0:
            return np.ma.mean(vis)
        else:
            return self.vis[y, x]