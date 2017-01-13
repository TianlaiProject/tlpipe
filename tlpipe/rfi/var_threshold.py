import numpy as np
import combinatorial_threshold


class VarThreshold(combinatorial_threshold.CombinatorialThreshold):

    def __init__(self, time_freq_vis, time_freq_vis_mask=None, first_threshold=6.0, exp_factor=1.2, distribution='Rayleigh', max_threshold_length=1024):

        super(SumThreshold, self).__init__(time_freq_vis, time_freq_vis_mask, first_threshold, exp_factor, distribution, max_threshold_length)


    def horizontal_var_threshold(self, length, threshold):

        height, width = self.vis.shape

        if length > width:
            return

        vis = np.ma.array(self.vis, mask=self.vis_mask)

        for y in xrange(height):
            for x in xrange(width-length):
                vis1 = np.ma.compressed(vis[y, x:x+length])
                if vis1.size > 0 and (vis1 > threshold).all():
                    self.vis_mask[y, x:x+length] = True

    def vertical_var_threshold(self, length, threshold):

        height, width = self.vis.shape

        if length > height:
            return

        vis = np.ma.array(self.vis, mask=self.vis_mask)

        for x in xrange(width):
            for y in xrange(height-length):
                vis1 = np.ma.compressed(vis[y:y+length, x])
                if vis1.size > 0 and (vis1 > threshold).all():
                    self.vis_mask[y:y+length, x] = True

    def execute_threshold(self, factor):
        for length, threshold in zip(self.lengths, self.thresholds):
            self.vertical_var_threshold(length, factor*threshold) # first time
            self.horizontal_var_threshold(length, factor*threshold) # then freq
