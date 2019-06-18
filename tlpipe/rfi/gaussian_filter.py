"""Gaussian filter method.

Inheritance diagram
-------------------

.. inheritance-diagram:: GaussianFilter
   :parts: 1

"""

import surface_fit
import numpy as np
import scipy.ndimage as ndimage


class GaussianFilter(surface_fit.SurfaceFitMethod):
    """Gaussian filter method.

    In this method, the background is caculated by a Gaussian high pass
    filtering process.

    """

    def __init__(self, time_freq_vis, time_freq_vis_mask=None, time_kernal_size=7.5, freq_kernal_size=15.0, fill_val=0, filter_direction=('time', 'freq')):

        super(GaussianFilter, self).__init__(time_freq_vis, time_freq_vis_mask)

        self._hksize = freq_kernal_size
        self._vksize = time_kernal_size
        self._fill_val = fill_val
        self._direction = filter_direction


    def fit(self):
        """Fit the background."""

        vis = np.where(self.vis_mask, self._fill_val, self.vis) # fill masked vals
        if set(self._direction) == set(('time',)):
            ndimage.gaussian_filter1d(vis, sigma=self._vksize, axis=0, order=0, output=self._background)
        elif set(self._direction) == set(('freq',)):
            ndimage.gaussian_filter1d(vis, sigma=self._hksize, axis=1, order=0, output=self._background)
        else:
            if not set(self._direction) == set(('time', 'freq')):
                warnings.warn('Invalid fileter direction: %s, will filter in both time and freq directions' % direct)
            ndimage.gaussian_filter(vis, sigma=(self._vksize, self._hksize), order=0, output=self._background)

        return self._background