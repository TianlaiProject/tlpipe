"""The VarThreshold method.

Inheritance diagram
-------------------

.. inheritance-diagram:: VarThreshold tlpipe.rfi.sum_threshold.SumThreshold
   :parts: 1

"""

import numpy as np
import combinatorial_threshold


class VarThreshold(combinatorial_threshold.CombinatorialThreshold):
    """The VarThreshold method.

    For more details, see Offringa et al., 2000, MNRAS, 405, 155,
    *Post-correlation radio frequency interference classification methods*.

    """

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
                if vis1.size > 0 and (np.abs(vis1) > threshold).all():
                    self.vis_mask[y, x:x+length] = True

    def vertical_var_threshold(self, length, threshold):

        height, width = self.vis.shape

        if length > height:
            return

        vis = np.ma.array(self.vis, mask=self.vis_mask)

        for x in xrange(width):
            for y in xrange(height-length):
                vis1 = np.ma.compressed(vis[y:y+length, x])
                if vis1.size > 0 and (np.abs(vis1) > threshold).all():
                    self.vis_mask[y:y+length, x] = True

    def execute_threshold(self, factor, direction):
        for direct in direction:
            if direct == 'time':
                for length, threshold in zip(self.time_lengths, self.time_thresholds):
                    self.vertical_var_threshold(length, factor*threshold) # first time
            elif direct == 'freq':
                for length, threshold in zip(self.freq_lengths, self.freq_thresholds):
                    self.horizontal_var_threshold(length, factor*threshold) # then freq
            else:
                warnings.warn('Invalid direction: %s, no RFI thresholding will be done' % direct)
