import numpy as np

from fmmode.core import telescope
from fmmode.telescope import cylinder, cylbeam
from fmmode.util import config
from fmmode.util import typeutil


class RandomCylinder(cylinder.UnpolarisedCylinderTelescope):

    pos_sigma = 0.5

    def feed_positions_cylinder(self, cylinder_index):

        pos = super(RandomCylinder, self).feed_positions_cylinder(cylinder_index)

        rs = np.random.get_state()
        np.random.seed(cylinder_index)

        p1 = np.sort(pos[:, 1] + self.pos_sigma * self.feed_spacing * np.random.standard_normal(pos.shape[0]))

        np.random.set_state(rs)

        pos[:, 1] = p1
        return pos



class GradientCylinder(cylinder.UnpolarisedCylinderTelescope):

    min_spacing = config.Property(proptype=float, default=-1.0)
    max_spacing = config.Property(proptype=float, default=20.0)


    def feed_positions_cylinder(self, cylinder_index):

        if cylinder_index >= self.num_cylinders or cylinder_index < 0:
            raise Exception("Cylinder index is invalid.")

        nf = self.num_feeds

        # Parameters for gradient feedspacing
        a = self.wavelengths[-1] / 2.0 if self.min_spacing < 0.0 else self.min_spacing
        # b = 2 * (sp - a) / nf
        b = 2.0*(self.max_spacing - a * (nf-1)) / (nf-1)**2.0

        pos = np.empty([nf, 2], dtype=np.float64)

        i = np.arange(nf)

        pos[:, 0] = cylinder_index * self.cylinder_spacing
        pos[:, 1] = a*i + 0.5 * b*i**2

        return pos



class CylinderExtra(cylinder.UnpolarisedCylinderTelescope):

    extra_feeds = config.Property(proptype=np.array, default=[])


    def feed_positions_cylinder(self, cylinder_index):

        pos = super(CylinderExtra, self).feed_positions_cylinder(cylinder_index)


        nextra = self.extra_feeds.shape[0]

        pos2 = np.zeros((pos.shape[0] + nextra, 2), dtype=np.float64)

        pos2[nextra:] = pos

        pos2[:nextra, 0] = cylinder_index * self.cylinder_spacing
        pos2[:nextra, 1] = self.extra_feeds

        return pos2



class CylinderPerturbed(cylinder.PolarisedCylinderTelescope):
    """A base for a polarised telescope.

    Again, an abstract class, but the only things that require implementing are
    the `feedpositions`, `_get_unique` and the beam functions `beamx` and `beamy`.

    Abstract Methods
    ----------------
    beamx, beamy : methods
        Routines giving the field pattern for the x and y feeds.
    """

    npert = 2

    @property
    def beamclass(self):
        """Simple beam mode of dual polarisation feeds."""
        nsfeed = self._single_feedpositions.shape[0]

        # Evens are X feed for each pert, Odds are Y feed
        beamclass = [ bc * np.ones(nsfeed) for bc in range(2*self.npert) ]

        return np.concatenate(beamclass).astype(np.int)


    @property
    def feedpositions(self):
        beampos = [ self._single_feedpositions for bc in range(2*self.npert)]
        return np.concatenate(beampos)


    def beamx(self, feed, freq):
        """Beam for the x polarisation feed.

        Parameters
        ----------
        feed : integer
            Index for the feed.
        freq : integer
            Index for the frequency.

        Returns
        -------
        beam : np.ndarray
            Healpix maps (of size [self._nside, 2]) of the field pattern in the
            theta and phi directions.
        """
        beampert = int(self.beamclass[feed] / 2)

        if beampert == 0:

            return cylbeam.beam_x(self._angpos, self.zenith,
                self.cylinder_width / self.wavelengths[freq], self.fwhm_e, self.fwhm_h)

        elif beampert == 1:

            beam0 = cylbeam.beam_x(self._angpos, self.zenith,
                self.cylinder_width / self.wavelengths[freq], self.fwhm_e, self.fwhm_h)

            beam1 = cylbeam.beam_x(self._angpos, self.zenith,
                self.cylinder_width / self.wavelengths[freq], self.fwhm_e * 1.01, self.fwhm_h)

            dbeam = (beam1 - beam0) / (0.01 * self.fwhm_e)

            return dbeam


    def beamy(self, feed, freq):
        """Beam for the x polarisation feed.

        Parameters
        ----------
        feed : integer
            Index for the feed.
        freq : integer
            Index for the frequency.

        Returns
        -------
        beam : np.ndarray
            Healpix maps (of size [self._nside, 2]) of the field pattern in the
            theta and phi directions.
        """

        beampert = int(self.beamclass[feed] / 2)

        if beampert == 0:

            return cylbeam.beam_y(self._angpos, self.zenith,
                self.cylinder_width / self.wavelengths[freq], self.fwhm_e, self.fwhm_h)

        elif beampert == 1:

            beam0 = cylbeam.beam_y(self._angpos, self.zenith,
                self.cylinder_width / self.wavelengths[freq], self.fwhm_e, self.fwhm_h)

            beam1 = cylbeam.beam_y(self._angpos, self.zenith,
                self.cylinder_width / self.wavelengths[freq], self.fwhm_e * 1.01, self.fwhm_h)

            dbeam = (beam1 - beam0) / (0.01 * self.fwhm_e)

            return dbeam



class UnpolarisedCylinderShift(cylinder.UnpolarisedCylinderTelescope):

    shift = config.Property(proptype=typeutil.float_or_non_empty_list, default=0.0)


    def feed_positions_cylinder(self, cylinder_index):

        if cylinder_index >= self.num_cylinders or cylinder_index < 0:
            raise Exception("Cylinder index is invalid.")

        ncyl = self.num_cylinders
        if len(self.shift) == 1:
            self.shift *= ncyl
        if len(self.shift) != ncyl:
            raise Exception('Invalid shifts.')

        pos = super(UnpolarisedCylinderShift, self).feed_positions_cylinder(cylinder_index)
        v_shift = np.ones_like(pos)
        v_shift[:, 0] = 0.0 # u-direction doesn't change
        pos += self.shift[cylinder_index] * v_shift

        return pos


class PolarisedCylinderShift(cylinder.PolarisedCylinderTelescope):

    shift = config.Property(proptype=typeutil.float_or_non_empty_list, default=0.0)


    def feed_positions_cylinder(self, cylinder_index):

        if cylinder_index >= self.num_cylinders or cylinder_index < 0:
            raise Exception("Cylinder index is invalid.")

        ncyl = self.num_cylinders
        if len(self.shift) == 1:
            self.shift *= ncyl
        if len(self.shift) != ncyl:
            raise Exception('Invalid shifts.')

        pos = super(PolarisedCylinderShift, self).feed_positions_cylinder(cylinder_index)
        v_shift = np.ones_like(pos)
        v_shift[:, 0] = 0.0 # u-direction doesn't change
        pos += self.shift[cylinder_index] * v_shift

        return pos


class UnequalFeedsUnpolarisedCylinder(cylinder.UnpolarisedCylinderTelescope):
    """A unpolarized unequal number of feeds cylinder array.

    A class for multiple side by side placed cylinders, each may have different number of equal spacing feeds.
    """

    num_cylinders = config.Property(proptype=typeutil.positive_int, default=3)
    num_feeds = config.Property(proptype=typeutil.positive_int_or_non_empty_list, default=[31, 32, 33])
    feed_spacing = config.Property(proptype=typeutil.positive_float_or_non_empty_list, default=[15.5/30, 0.5, 15.5/32])
    cylinder_width = config.Property(proptype=typeutil.positive_float, default=15.0)


    def feed_positions_cylinder(self, cylinder_index):
        """Get the feed positions on the specified cylinder.

        Parameters
        ----------
        cylinder_index : integer
            The cylinder index, an integer from 0 to self.num_cylinders.

        Returns
        -------
        feed_positions : np.ndarray
            The positions in the telescope plane of the receivers. Packed as
            [[u1, v1], [u2, v2], ...].
        """

        if cylinder_index >= self.num_cylinders or cylinder_index < 0:
            raise Exception("Cylinder index is invalid.")

        ncyl = self.num_cylinders
        if len(self.num_feeds) == 1:
            self.num_feeds *= ncyl
        if len(self.num_feeds) != ncyl:
            raise Exception('Invalid num_feeds.')

        if len(self.feed_spacing) == 1:
            self.feed_spacing *= ncyl
        if len(self.feed_spacing) != ncyl:
            raise Exception('Invalid feed_spacings.')

        nf = self.num_feeds[cylinder_index]
        sp = self.feed_spacing[cylinder_index]
        if self.non_commensurate:
            nf = nf - cylinder_index
            sp = sp / (nf - 1.0) * nf

        pos = np.empty([nf, 2], dtype=np.float64)

        pos[:, 0] = cylinder_index * self.cylinder_spacing
        pos[:, 1] = np.arange(nf) * sp

        return pos


class UnequalFeedsCylinder(cylinder.PolarisedCylinderTelescope):
    """A polarized unequal number of feeds cylinder array.

    A class for multiple side by side placed cylinders, each may have different number of equal spacing feeds.
    """

    num_cylinders = config.Property(proptype=typeutil.positive_int, default=3)
    num_feeds = config.Property(proptype=typeutil.positive_int_or_non_empty_list, default=[31, 32, 33])
    feed_spacing = config.Property(proptype=typeutil.positive_float_or_non_empty_list, default=[15.5/30, 0.5, 15.5/32])
    cylinder_width = config.Property(proptype=typeutil.positive_float, default=15.0)


    def feed_positions_cylinder(self, cylinder_index):
        """Get the feed positions on the specified cylinder.

        Parameters
        ----------
        cylinder_index : integer
            The cylinder index, an integer from 0 to self.num_cylinders.

        Returns
        -------
        feed_positions : np.ndarray
            The positions in the telescope plane of the receivers. Packed as
            [[u1, v1], [u2, v2], ...].
        """

        if cylinder_index >= self.num_cylinders or cylinder_index < 0:
            raise Exception("Cylinder index is invalid.")

        ncyl = self.num_cylinders
        if len(self.num_feeds) == 1:
            self.num_feeds *= ncyl
        if len(self.num_feeds) != ncyl:
            raise Exception('Invalid num_feeds.')

        if len(self.feed_spacing) == 1:
            self.feed_spacing *= ncyl
        if len(self.feed_spacing) != ncyl:
            raise Exception('Invalid feed_spacings.')

        nf = self.num_feeds[cylinder_index]
        sp = self.feed_spacing[cylinder_index]
        if self.non_commensurate:
            nf = nf - cylinder_index
            sp = sp / (nf - 1.0) * nf

        pos = np.empty([nf, 2], dtype=np.float64)

        pos[:, 0] = cylinder_index * self.cylinder_spacing
        pos[:, 1] = np.arange(nf) * sp

        return pos


class ArbitraryPolarisedCylinder(cylinder.PolarisedCylinderTelescope):
    """A polarized cylinder array with arbitrary number of feeeds and feeds position.

    A class for multiple side by side placed cylinders, each may have different number of arbitrary spacing feeds.
    """

    num_cylinders = config.Property(proptype=typeutil.positive_int, default=3)
    num_feeds = config.Property(proptype=typeutil.positive_int_or_non_empty_list, default=[31, 32, 33])
    feed_spacing = config.Property(proptype=typeutil.non_empty_list, default=[((15.5/30) * np.arange(31)).tolist(), (0.5 * np.arange(32)).tolist(), ((15.5/32) * np.arange(33)).tolist()])
    cylinder_width = config.Property(proptype=typeutil.positive_float, default=15.0)


    def feed_positions_cylinder(self, cylinder_index):
        """Get the feed positions on the specified cylinder.

        Parameters
        ----------
        cylinder_index : integer
            The cylinder index, an integer from 0 to self.num_cylinders.

        Returns
        -------
        feed_positions : np.ndarray
            The positions in the telescope plane of the receivers. Packed as
            [[u1, v1], [u2, v2], ...].
        """

        if cylinder_index >= self.num_cylinders or cylinder_index < 0:
            raise Exception("Cylinder index is invalid.")

        ncyl = self.num_cylinders
        if len(self.num_feeds) == 1:
            self.num_feeds *= ncyl
        if len(self.num_feeds) != ncyl:
            raise Exception('Invalid num_feeds.')

        if len(self.feed_spacing) == 1:
            self.feed_spacing *= ncyl
        if len(self.feed_spacing) != ncyl:
            raise Exception('Invalid feed_spacings.')
        for idx in range(ncyl):
            if len(self.feed_spacing[idx]) != self.num_feeds[idx]:
                raise Exception('Invalid feed_spacings.')

        nf = self.num_feeds[cylinder_index]
        sp = self.feed_spacing[cylinder_index]
        if self.non_commensurate:
            nf = nf - cylinder_index
            sp = sp / (nf - 1.0) * nf

        pos = np.empty([nf, 2], dtype=np.float64)

        pos[:, 0] = cylinder_index * self.cylinder_spacing
        pos[:, 1] = sp

        return pos
