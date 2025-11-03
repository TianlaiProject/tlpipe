import numpy as np
import h5py

from . import cylinder


class TlUnpolarisedCylinder(cylinder.UnpolarisedCylinderTelescope):
    """A telescope describing the Tianlai non-polarized cylinder array."""

    def __init__(self, latitude=45, longitude=0, freqs=[], band_width=None, tsys_flat=50.0, ndays=1.0, accuracy_boost=1.0, l_boost=1.0, bl_range=[0.0, 1.0e7], auto_correlations=False, local_origin=True, cylinder_width=15.0, feedpos=np.zeros((0, 3)), lmax=None, mmax=None, in_cylinder=True, touching=True, cylspacing=0.0, non_commensurate=False, e_width=0.7, h_width=1.0, use_beam=None, use_beam_freq_offset=0):

        num_feeds = len(feedpos)
        self.feedpos = feedpos[:, :2] # do not care z
        self.use_beam = use_beam
        self.use_beam_freq_offset = use_beam_freq_offset

        cylinder.UnpolarisedCylinderTelescope.__init__(self, latitude, longitude, freqs, band_width, tsys_flat, ndays, accuracy_boost, l_boost, bl_range, auto_correlations, local_origin, lmax, mmax, 3, num_feeds, cylinder_width, 0.4, in_cylinder, touching, cylspacing, non_commensurate, e_width, h_width)


    @property
    def _single_feedpositions(self):
        """The set of feed positions on *all* cylinders.

        Returns
        -------
        feedpositions : np.ndarray
            The positions in the telescope plane of the receivers. Packed as
            [[u1, v1], [u2, v2], ...].
        """

        return self.feedpos


    def beam(self, feed, freq):
        """Beam for a particular feed.

        Parameters
        ----------
        feed : integer
            Index for the feed.
        freq : integer
            Index for the frequency.

        Returns
        -------
        beam : np.ndarray
            A Healpix map (of size self._nside) of the beam. Potentially
            complex.
        """
        if self.use_beam:
            if feed == 0 and freq == 0:
                print(f'Use beam model in {self.use_beam}...', flush=True)
            with h5py.File(self.use_beam, 'r') as f:
                bm2 = f['bm2'][freq + self.use_beam_freq_offset]
            return np.abs(bm2)**0.5
        else:
            return super().beam(feed, freq)
