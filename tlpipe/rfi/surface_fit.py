import abc
import numpy as np


class SurfaceFitMethod(object):
    """Abstract base class for surface fitting methods.

    A surface fit to the correlated visibilities :math:`V(\\nu, t)` as a
    function of frequency :math:`\\nu` and time :math:`t` can produce a
    surface :math:`\\hat{V}(\\nu, t)` that represents the astronomical
    information in the signal. Requiring :math:`\\hat{V}(\\nu, t)` to be
    a smooth surface is a good assumption for most astronomical continuum
    sources, as their observed amplitude tend not to change rapidly with
    time and frequency, whereas specific types of RFI can create sharp edges
    in the time-frequency domain. The residuals between the fit and the data
    contain the system noise and the RFI, which can then be thresholded
    without the chance of flagging astronomical sources that have visibilities
    with high amplitude.

    .. note::
        Because of the smoothing in both time and frequency direction, this
        method is not directly usable when observing strong line sources or
        strong pulsars.

    """

    __metaclass__ = abc.ABCMeta  # Enforce Abstract class


    def __init__(self, time_freq_vis, time_freq_vis_mask=None):

        self.vis = time_freq_vis

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