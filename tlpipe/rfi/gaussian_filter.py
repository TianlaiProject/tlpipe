import surface_fit
import numpy as np
import scipy.ndimage as ndimage


class GaussianFilter(surface_fit.SurfaceFitMethod):

    def __init__(self, time_freq_vis, time_freq_vis_mask=None, time_kernal_size=7.5, freq_kernal_size=15.0, fill_val=0):

        super(GaussianFilter, self).__init__(time_freq_vis, time_freq_vis_mask)

        self._hksize = freq_kernal_size
        self._vksize = time_kernal_size
        self._fill_val = fill_val


    def fit(self):
        """Fit the background."""

        vis = np.where(self.vis_mask, self._fill_val, self.vis) # fill masked vals
        ndimage.gaussian_filter(vis, sigma=(self._vksize, self._hksize), order=0, output=self._background)

        return self._background