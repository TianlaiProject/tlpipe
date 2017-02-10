"""Local fit method.

Inheritance diagram
-------------------

.. inheritance-diagram:: LocalFitMethod
   :parts: 1

"""

import abc
import surface_fit


class LocalFitMethod(surface_fit.SurfaceFitMethod):
    """Abstract base class for local fit method."""

    __metaclass__ = abc.ABCMeta  # Enforce Abstract class


    def __init__(self, time_freq_vis, time_freq_vis_mask=None, time_window_size=20, freq_window_size=40):

        super(LocalFitMethod, self).__init__(time_freq_vis, time_freq_vis_mask)

        height, width = self.vis.shape

        if 2 * freq_window_size > width:
            self._hsize = freq_window_size / 2
        else:
            self._hsize = freq_window_size

        if 2 * time_window_size > height:
            self._vsize = time_window_size / 2
        else:
            self._vsize = time_window_size


    def calculate_background(self, x, y):

        height, width = self.vis.shape

        start_x = max(0, x-self._hsize)
        end_x = min(width, x+self._hsize)

        start_y = max(0, y-self._vsize)
        end_y = min(height, y+self._vsize)

        return self._calculate(x, y, start_x, end_x, start_y, end_y)

    @abc.abstractmethod
    def _calculate(self, x, y, start_x, end_x, start_y, end_y):
        """The actual background calculate method."""
        return

    def fit(self):
        """Fit the background."""

        height, width = self.vis.shape

        for y in xrange(height):
            for x in xrange(width):
                self._background[y, x] = self.calculate_background(x, y)

        return self._background