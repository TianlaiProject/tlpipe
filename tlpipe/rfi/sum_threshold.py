"""The SumThreshold method.

Inheritance diagram
-------------------

.. inheritance-diagram:: SumThreshold tlpipe.rfi.var_threshold.VarThreshold
   :parts: 1

"""

import numpy as np
import combinatorial_threshold


class SumThreshold(combinatorial_threshold.CombinatorialThreshold):
    """The SumThreshold method.

    For more details, see Offringa et al., 2000, MNRAS, 405, 155,
    *Post-correlation radio frequency interference classification methods*.

    """

    def __init__(self, time_freq_vis, time_freq_vis_mask=None, first_threshold=6.0, exp_factor=1.5, distribution='Rayleigh', max_threshold_length=1024, min_connected=1):

        super(SumThreshold, self).__init__(time_freq_vis, time_freq_vis_mask, first_threshold, exp_factor, distribution, max_threshold_length)

        self.min_connected = max(1, int(min_connected))


    def horizontal_sum_threshold(self, length, threshold):

        height, width = self.vis.shape

        if length > width:
            return

        if length == 1:
            for y in xrange(height):

                # if all have been masked, continue
                if self.vis_mask[y].all():
                    continue

                for x in xrange(width):
                    if (not self.vis_mask[y, x]) and (np.abs(self.vis[y, x]) > threshold):
                        self.vis_mask[y, x] = True
        elif length > 1:
            tmp_mask = self.vis_mask.copy()
            for y in xrange(height):

                # if all have been masked, continue
                if self.vis_mask[y].all():
                    continue

                sm, cnt, left, right = 0, 0, 0, 0
                for right in xrange(length-1):
                    if not self.vis_mask[y, right]:
                        sm += self.vis[y, right]
                        cnt += 1

                while(right < width):
                    # add the sample at the right
                    if not self.vis_mask[y, right]:
                        sm += self.vis[y, right]
                        cnt += 1
                    # check
                    if (cnt > 0) and (np.abs(sm / cnt) > threshold):
                        tmp_mask[y, left:left+length] = True
                    # subtract the sample at the left
                    if not self.vis_mask[y, left]:
                        sm -= self.vis[y, left]
                        cnt -= 1

                    left += 1
                    right += 1

            # set to the new mask
            self.vis_mask[:] = tmp_mask

    def vertical_sum_threshold(self, length, threshold):

        height, width = self.vis.shape

        if length > height:
            return

        if length == 1:
            for x in xrange(width):

                # if all have been masked, continue
                if self.vis_mask[:, x].all():
                    continue

                for y in xrange(height):
                    if (not self.vis_mask[y, x]) and (np.abs(self.vis[y, x]) > threshold):
                        self.vis_mask[y, x] = True
        elif length > 1:
            tmp_mask = self.vis_mask.copy()
            for x in xrange(width):

                # if all have been masked, continue
                if self.vis_mask[:, x].all():
                    continue

                sm, cnt, top, bottom = 0, 0, 0, 0
                for bottom in xrange(length-1):
                    if not self.vis_mask[bottom, x]:
                        sm += self.vis[bottom, x]
                        cnt += 1

                while(bottom < height):
                    # add the sample at the bottom
                    if not self.vis_mask[bottom, x]:
                        sm += self.vis[bottom, x]
                        cnt += 1
                    # check
                    if (cnt > 0) and (np.abs(sm / cnt) > threshold):
                        tmp_mask[top:top+length, x] = True
                    # subtract the sample at the top
                    if not self.vis_mask[top, x]:
                        sm -= self.vis[top, x]
                        cnt -= 1

                    top += 1
                    bottom += 1

            # set to the new mask
            self.vis_mask[:] = tmp_mask

    def execute_threshold(self, factor):
        for length, threshold in zip(self.lengths, self.thresholds):
            self.vertical_sum_threshold(length, factor*threshold) # first time
        for length, threshold in zip(self.lengths, self.thresholds):
            self.horizontal_sum_threshold(length, factor*threshold) # then freq

    def execute(self, sensitivity=1.0):
        super(SumThreshold, self).execute(sensitivity)

        if self.min_connected > 1:
            # self.filter_connected_samples()
            raise NotImplementedError
