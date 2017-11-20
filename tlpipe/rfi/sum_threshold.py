"""The SumThreshold method.

Inheritance diagram
-------------------

.. inheritance-diagram:: SumThreshold tlpipe.rfi.var_threshold.VarThreshold
   :parts: 1

"""

import warnings
import numpy as np
import combinatorial_threshold
from _sum_threshold import threshold_len1, hthreshold, vthreshold


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
            threshold_len1(self.vis, self.vis_mask, height, width, threshold)
        elif length > 1:
            hthreshold(self.vis, self.vis_mask, height, width, length, threshold)

    def vertical_sum_threshold(self, length, threshold):

        height, width = self.vis.shape

        if length > height:
            return

        if length == 1:
            threshold_len1(self.vis, self.vis_mask, height, width, threshold)
        elif length > 1:
            vthreshold(self.vis, self.vis_mask, height, width, length, threshold)

    def execute_threshold(self, factor, direction):
        for direct in direction:
            if direct == 'time':
                for length, threshold in zip(self.time_lengths, self.time_thresholds):
                    self.vertical_sum_threshold(length, factor*threshold) # first time
            elif direct == 'freq':
                for length, threshold in zip(self.freq_lengths, self.freq_thresholds):
                    self.horizontal_sum_threshold(length, factor*threshold) # then freq
            else:
                warnings.warn('Invalid direction: %s, no RFI thresholding will be done' % direct)

    def execute(self, sensitivity=1.0, direction=('time', 'freq')):
        super(SumThreshold, self).execute(sensitivity, direction)

        if self.min_connected > 1:
            # self.filter_connected_samples()
            raise NotImplementedError
