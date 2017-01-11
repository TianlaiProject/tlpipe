import local_fit
import numpy as np
import scipy.ndimage as ndimage


class LocalGaussianFilter(local_fit.LocalFitMethod):

    def __init__(self, time_freq_vis, time_freq_vis_mask=None, time_window_size=20, freq_window_size=40, time_kernal_size=7.5, freq_kernal_size=15.0):

        super(LocalGaussianFilter, self).__init__(time_freq_vis, time_freq_vis_mask, time_window_size, freq_window_size)

        self._hksize = freq_kernal_size
        self._vksize = time_kernal_size


    def fit(self):
        """Fit the background."""

        vis = np.where(self.vis_mask, 0, self.vis) # fill masked vals to 0
        ndimage.gaussian_filter(vis, sigma=(self._vksize, self._hksize), order=0, output=self._background)

        return self._background