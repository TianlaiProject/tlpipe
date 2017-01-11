import local_fit
import numpy as np


class LocalMedianFit(local_fit.LocalFitMethod):

    def _calculate(self, x, y, start_x, end_x, start_y, end_y):

        vis = np.ma.array(self.vis[start_y:end_y, start_x:end_x], mask=self.vis_mask[start_y:end_y, start_x:end_x])

        if vis.count() > 0:
            return np.ma.median(vis)
        else:
            return self.vis[y, x]