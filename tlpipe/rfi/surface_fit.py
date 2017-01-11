import abc
import numpy as np


class SurfaceFitMethod(object):
    """Abstract base class for surface fit methods."""

    __metaclass__ = abc.ABCMeta  # Enforce Abstract class


    def __init__(self, time_freq_vis, time_freq_vis_mask=None):

        self.vis = np.abs(time_freq_vis) # fit for only the amplitude

        if time_freq_vis_mask is None:
            self.vis_mask = np.where(np.isfinite(self.vis), False, True)
        elif self.vis.shape == time_freq_vis_mask.shape:
            self.vis_mask = time_freq_vis_mask.astype(np.bool)
        else:
            raise ValueError('Invalid time_freq_vis_mask')

        self._background = np.zeros_like(self.vis)


    @abc.abstractmethod
    def fit(self):
        """Abstract method that needs to be implemented by sub-classes."""
        return self._background