"""Spline interpolation method.

Inheritance diagram
-------------------

.. inheritance-diagram:: Interpolate
   :parts: 1

"""

import surface_fit
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


class Interpolate(surface_fit.SurfaceFitMethod):
    """Spline interpolation method.

    This is not really a surface fit method, but intended to fill masked
    (or invalid) values presented in the data by spline interpolation.

    """

    def __init__(self, time_freq_vis, time_freq_vis_mask=None, direction='vertical', order=3, ext=0, mask_ratio=0.1):

        super(Interpolate, self).__init__(time_freq_vis, time_freq_vis_mask)

        if direction in ('horizontal', 'vertical'):
            self.direction = direction
        else:
            raise ValueError('direction can be either horizontal or vertical')

        order = int(order)
        if order >= 1 and order <= 5:
            self.order = order
        else:
            raise ValueError('Degree of the smoothing spline. Must be 1 <= k <= 5')

        if ext in (0, 1, 2, 3, 'extrapolate', 'zeros', 'raise', 'const'):
            self.ext = ext
        else:
            raise ValueError('Invalid extrapolate mode')

        if mask_ratio >=0.0 and mask_ratio <=1.0:
            self.mask_ratio = mask_ratio
        else:
            raise ValueError('Value of mask_ratio must between 0 and 1')


    def interpolate_horizontally(self):

        height, width = self.vis.shape

        for ri in xrange(height):
            on = np.where(self.vis_mask[ri])[0] # masked inds
            off = np.where(np.logical_not(self.vis_mask[ri]))[0] # un-masked inds
            if len(off) <= max(self.order + 1, self.mask_ratio*width):
                if len(off) == 0:
                    self._background[ri] = 0 # fill 0 if all has been masked
                else:
                    self._background[ri, on] = np.median(self.vis[ri, off])
                    self._background[ri, off] = self.vis[ri, off]
            else:
                itp = InterpolatedUnivariateSpline(off, self.vis[ri, off])
                self._background[ri] = itp(np.arange(width))


    def interpolate_vertically(self):

        height, width = self.vis.shape

        for ci in xrange(width):
            on = np.where(self.vis_mask[:, ci])[0] # masked inds
            off = np.where(np.logical_not(self.vis_mask[:, ci]))[0] # un-masked inds
            if len(off) <= max(self.order + 1, self.mask_ratio*height):
                if len(off) == 0:
                    self._background[:, ci] = 0 # fill 0 if all has been masked
                else:
                    self._background[on, ci] = np.median(self.vis[off, ci])
                    self._background[off, ci] = self.vis[off, ci]
            else:
                itp = InterpolatedUnivariateSpline(off, self.vis[off, ci])
                self._background[:, ci] = itp(np.arange(height))


    def fit(self):
        """Fit the background."""

        if self.direction == 'horizontal':
            self.interpolate_horizontally()
        else:
            self.interpolate_vertically()

        return self._background