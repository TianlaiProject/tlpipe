import abc
import itertools

import numpy as np

from cora.util import hputil, units
from caput import mpiutil

from tlpipe.map.fmmode.core import visibility
from tlpipe.map.fmmode.util import hpproj


def in_range(arr, min, max):
    """Check if array entries are within the given range.

    Parameters
    ----------
    arr : np.ndarray
        Array to check.
    min, max : scalar or np.ndarray
        Minimum and maximum values to test against. Values can be in arrays
        broadcastable against `arr`.

    Returns
    -------
    val : boolean
        True if all entries are within range.
    """
    return (arr >= min).all() and (arr < max).all()


def out_of_range(arr, min, max):
    return not in_range(arr, min, max)


def _merge_keyarray(keys1, keys2, mask1=None, mask2=None):

    tmask1 = mask1 if mask1 is not None else np.ones_like(keys1, dtype=np.bool)
    tmask2 = mask2 if mask2 is not None else np.ones_like(keys2, dtype=np.bool)

    # Merge two groups of feed arrays
    cmask = np.logical_and(tmask1, tmask2)
    ckeys = _remap_keyarray(keys1 + 1.0J * keys2, mask=cmask)

    if mask1 is None and mask2 is None:
        return ckeys
    else:
        return ckeys, cmask


def _remap_keyarray(keyarray, mask=None):
    # Look through an array of keys and attach integer labels to each
    # equivalent classes of keys (also take into account masking).
    if mask is None:
        mask = np.ones(keyarray.shape, np.bool)

    ind = np.where(mask)

    un, inv = np.unique(keyarray[ind], return_inverse=True)

    fmap = -1*np.ones(keyarray.shape, dtype=np.int)

    fmap[ind] = np.arange(un.size)[inv]
    return fmap


def _get_indices(keyarray, mask=None, return_allpairs=False):
    # Return a pair of indices for each group of equivalent feed pairs
    if mask is None:
        mask = np.ones(keyarray.shape, np.bool)

    wm = np.where(mask.ravel())[0]
    keysflat = keyarray.ravel()[wm]

    un, ind = np.unique(keysflat, return_index=True)
    # CHANGE: np (< 1.6) does not support multiple indices in np.unravel_index
    #upairs = np.array(np.unravel_index(wm[ind], keyarray.shape)).T
    upairs = np.array([np.unravel_index(i1, keyarray.shape) for i1 in wm[ind] ])

    # get all feed pairs grouped by redundancy
    if return_allpairs:
        all_inds = []
        for ui in un:
            all_inds += np.where(keysflat == ui)[0].tolist()
        all_pairs = np.array([np.unravel_index(i1, keyarray.shape) for i1 in wm[all_inds] ])

        #return np.sort(upairs, axis=-1) # Sort to ensure we are in upper triangle
        return upairs, all_pairs
    else:
        return upairs


def max_lm(baselines, wavelengths, uwidth, vwidth=0.0):
    """Get the maximum (l,m) that a baseline is sensitive to.

    Parameters
    ----------
    baselines : np.ndarray
        An array of baselines.
    wavelengths : np.ndarray
        An array of wavelengths.
    uwidth : np.ndarray
        Width of the receiver in the u-direction.
    vwidth : np.ndarray
        Width of the receiver in the v-direction.

    Returns
    -------
    lmax, mmax : array_like
    """

    umax = (np.abs(baselines[:, 0]) + uwidth) / wavelengths
    vmax = (np.abs(baselines[:, 1]) + vwidth) / wavelengths

    mmax = np.ceil(2 * np.pi * umax).astype(np.int64)
    lmax = np.ceil((mmax**2 + (2*np.pi*vmax)**2)**0.5).astype(np.int64)

    return lmax, mmax


def latlon_to_sphpol(latlon):

    zenith = np.array([np.pi / 2.0 - np.radians(latlon[0]),
                       np.remainder(np.radians(latlon[1]), 2*np.pi)])

    return zenith



class TransitTelescope(object):
    """Base class for simulating any transit interferometer.

    This is an abstract class, and several methods must be implemented before it
    is usable. These are:

    * `feedpositions` - a property which contains the positions of all the feeds
    * `_get_unique` -  calculates which baselines are identical
    * `_transfer_single` - calculate the beam transfer for a single baseline+freq
    * `_make_matrix_array` - makes an array of the right size to hold the
      transfer functions
    * `_copy_transfer_into_single` - copy a single transfer matrix into a
      collection.

    The last two are required for supporting polarised beam functions.

    Properties
    ----------
    zenith : [latitude, longitude]
        Must be set in degrees (implicit conversion to spherical polars on radians)
    freq_lower, freq_higher : scalar
        The lower / upper bound of the lowest / highest frequency bands.
    num_freq : scalar
        The number of frequency bands (only use for setting up the frequency
        binning). Generally using `nfreq` is preferred.
    tsys_flat : scalar
        The system temperature (in K). Override `tsys` for anything more
        sophisticated.
    positive_m_only: boolean
        Whether to only deal with half the `m` range. In many cases we are
        much less sensitive to negative-m (depending on the hemisphere, and
        baseline alignment). This does not affect the beams calculated, only
        how they're used in further calculation. Default: False
    minlength, maxlength : scalar
        Minimum and maximum baseline lengths to include (in metres).

    """
    __metaclass__ = abc.ABCMeta  # Enforce Abstract class


    def __init__(self, latitude=45, longitude=0, freqs=[], beam_theta_range=[0.0, 180.0], tsys_flat=50.0, ndays=1.0, accuracy_boost=1.0, l_boost=1.0, bl_range=[0.0, 1.0e7], auto_correlations=False):

        self.zenith = latlon_to_sphpol([latitude, longitude])
        self.frequencies = np.array(freqs)
        self.beam_theta_range = beam_theta_range
        self.tsys_flat = tsys_flat
        self.ndays = ndays
        self.accuracy_boost = accuracy_boost
        self.l_boost = l_boost
        self.minlength = bl_range[0]
        self.maxlength = bl_range[1]
        self.auto_correlations = auto_correlations


    _pickle_keys = []

    def __getstate__(self):

        state = self.__dict__.copy()

        for key in self.__dict__:
            if (key not in self._pickle_keys) and (key[0] == "_"):
                del state[key]

        return state



    #========= Properties related to baselines =========

    _baselines = None

    @property
    def baselines(self):
        """The unique baselines in the telescope."""
        if self._baselines is None:
            self.calculate_feedpairs()

        return self._baselines


    _redundancy = None

    @property
    def redundancy(self):
        """The redundancy of each baseline (corresponds to entries in
        cyl.baselines)."""
        if self._redundancy is None:
            self.calculate_feedpairs()

        return self._redundancy

    @property
    def nbase(self):
        """The number of unique baselines."""
        return self.npairs


    @property
    def npairs(self):
        """The number of unique feed pairs."""
        return self.uniquepairs.shape[0]


    _uniquepairs = None

    @property
    def uniquepairs(self):
        """An (npairs, 2) array of the feed pairs corresponding to each baseline."""
        if self._uniquepairs is None:
            self.calculate_feedpairs()
        return self._uniquepairs


    _allpairs = None

    @property
    def allpairs(self):
        """An (nbl, 2) array of the all antenna feed pairs grouped according to redundancies."""
        if self._allpairs is None:
            self.calculate_feedpairs()
        return self._allpairs


    _feedmap = None

    @property
    def feedmap(self):
        """An (nfeed, nfeed) array giving the mapping between feedpairs and
        the calculated baselines. Each entry is an index into the arrays of unique pairs."""

        if self._feedmap is None:
            self.calculate_feedpairs()

        return self._feedmap


    _feedmask = None

    @property
    def feedmask(self):
        """An (nfeed, nfeed) array giving the entries that have been
        calculated. This allows to mask out pairs we want to ignore."""

        if self._feedmask is None:
            self.calculate_feedpairs()

        return self._feedmask

    _feedconj = None

    @property
    def feedconj(self):
        """An (nfeed, nfeed) array giving the feed pairs which must be complex
        conjugated."""

        if self._feedconj is None:
            self.calculate_feedpairs()

        return self._feedconj

    #===================================================



    #======== Properties related to frequencies ========

    @property
    def wavelengths(self):
        """The central wavelength of each frequency band (in metres)."""
        return units.c / (1e6 * self.frequencies)

    @property
    def nfreq(self):
        """The number of frequency bins."""
        return self.frequencies.shape[0]

    #===================================================




    #======== Properties related to the feeds ==========

    @property
    def nfeed(self):
        """The number of feeds."""
        return self.feedpositions.shape[0]

    #===================================================


    #======= Properties related to polarisation ========

    @property
    def num_pol_sky(self):
        """The number of polarisation combinations on the sky that we are
        considering. Should be either 1 (T=I only), 3 (T, Q, U) or 4 (T, Q, U and V).
        """
        return self._npol_sky_

    #===================================================




    #===== Properties related to harmonic spread =======

    _lmax = None

    @property
    def lmax(self):
        """The maximum l the telescope is sensitive to."""
        if self._lmax is None:
            lmax, mmax = max_lm(self.baselines, self.wavelengths.min(), self.u_width, self.v_width)
            self._lmax = int(np.ceil(lmax.max() * self.l_boost))

        return self._lmax

    @property
    def mmax(self):
        """The maximum m the telescope is sensitive to."""

        return self.phi_size / 2


    #===== Properties related to cartesian projection and FFT =======

    _cart_projector = None

    @property
    def cart_projector(self):
        if self._cart_projector is None:
            self._init_trans(hputil.nside_for_lmax(self.lmax, accuracy_boost=self.accuracy_boost))

        return self._cart_projector

    @property
    def theta_values(self):
        """The theta values corresponding to the projected cartesian map."""
        return 90.0 - self.cart_projector.y # in degree, [180, 0]

    @property
    def phi_values(self):
        """The phi values corresponding to the projected cartesian map."""
        x = self.cart_projector.x
        return np.where(x>0.0, 360.0-x, -x) # in degree, [180, 0]+[360, 180]

    @property
    def theta_size(self):
        """Number of theta pixels used in the Fourier transform."""
        return self.theta_values.size

    @property
    def phi_size(self):
        """Number of phi pixels used in the Fourier transform."""

        return self.phi_values.size

    #===================================================



    #== Methods for calculating the unique baselines ===

    def calculate_feedpairs(self):
        """Calculate all the unique feedpairs and their redundancies, and set
        the internal state of the object.
        """

        # Get unique pairs, and create mapping arrays
        self._feedmap, self._feedmask, self._feedconj = self._get_unique()

        # Reorder and conjugate baselines such that the default feedpair
        # points W->E (to ensure we use positive-m)
        self._make_ew()

        # Sort baselines into order
        self._sort_pairs()

        # Create mask of included pairs, that are not conjugated
        tmask = np.logical_and(self._feedmask, np.logical_not(self._feedconj))

        self._uniquepairs, self._allpairs = _get_indices(self._feedmap, mask=tmask, return_allpairs=True)
        self._redundancy = np.bincount(self._feedmap[np.where(tmask)]) # Triangle mask to avoid double counting
        self._baselines = self.feedpositions[self._uniquepairs[:, 0]] - self.feedpositions[self._uniquepairs[:, 1]]

    def _make_ew(self):
        # Reorder baselines pairs, such that the baseline vector always points E (or pure N)

        tmask = np.logical_and(self._feedmask, np.logical_not(self._feedconj))
        uniq = _get_indices(self._feedmap, mask=tmask)

        for i in range(uniq.shape[0]):
            sep = self.feedpositions[uniq[i, 0]] - self.feedpositions[uniq[i, 1]]

            if sep[0] < 0.0 or (sep[0] == 0.0 and sep[1] < 0.0):
                # Reorder feed pairs and conjugate mapping
                # self._uniquepairs[i, 1], self._uniquepairs[i, 0] = self._uniquepairs[i, 0], self._uniquepairs[i, 1]
                self._feedconj = np.where(self._feedmap == i, np.logical_not(self._feedconj), self._feedconj)

    # Tolerance used when comparing baselines. See np.around documentation for details.
    _bl_tol = 6

    def _unique_baselines(self):
        """Map of equivalent baseline lengths, and mask of ones to exclude.
        """
        # Construct array of indices
        fshape = [self.nfeed, self.nfeed]
        f_ind = np.indices(fshape)

        # Construct array of baseline separations in complex representation
        bl1 = (self.feedpositions[f_ind[0]] - self.feedpositions[f_ind[1]])
        bl2 = np.around(bl1[..., 0] + 1.0J * bl1[..., 1], self._bl_tol)

        # Flip sign if required to get common direction to correctly find redundant baselines
        #flip_sign = np.logical_or(bl2.real < 0.0, np.logical_and(bl2.real == 0, bl2.imag < 0))
        #bl2 = np.where(flip_sign, -bl2, bl2)

        # Construct array of baseline lengths
        blen = np.sum(bl1**2, axis=-1)**0.5

        # Create mask of included baselines
        mask = np.logical_and(blen >= self.minlength, blen <= self.maxlength)

        # Remove the auto correlated baselines between all polarisations
        if not self.auto_correlations:
            mask = np.logical_and(blen > 0.0, mask)

        return _remap_keyarray(bl2, mask), mask


    def _unique_beams(self):
        """Map of unique beam pairs, and mask of ones to exclude.
        """
        # Construct array of indices
        fshape = [self.nfeed, self.nfeed]

        bci, bcj = np.broadcast_arrays(self.beamclass[:, np.newaxis], self.beamclass[np.newaxis, :])

        beam_map = _merge_keyarray(bci, bcj)

        if self.auto_correlations:
            beam_mask = np.ones(fshape, dtype=np.bool)
        else:
            beam_mask = np.logical_not(np.identity(self.nfeed, dtype=np.bool))

        return beam_map, beam_mask

    def _get_unique(self):
        """Calculate the unique baseline pairs.

        All feeds are assumed to be identical. Baselines are identified if
        they have the same length, and are selected such that they point East
        (to ensure that the sensitivity ends up in positive-m modes).

        It is also possible to select only baselines within a particular
        length range by setting the `minlength` and `maxlength` properties.

        Parameters
        ----------
        fpairs : np.ndarray
            An array of all the feed pairs, packed as [[i1, i2, ...], [j1, j2, ...] ].

        Returns
        -------
        baselines : np.ndarray
            An array of all the unique pairs. Packed as [ [i1, i2, ...], [j1, j2, ...]].
        redundancy : np.ndarray
            For each unique pair, give the number of equivalent pairs.
        """

        # Fetch and merge map of unique feed pairs
        base_map, base_mask = self._unique_baselines()
        beam_map, beam_mask = self._unique_beams()
        comb_map, comb_mask = _merge_keyarray(base_map, beam_map, mask1=base_mask, mask2=beam_mask)

        # Take into account conjugation by identifying the indices of conjugate pairs
        conj_map = comb_map > comb_map.T
        comb_map = np.dstack((comb_map, comb_map.T)).min(axis=-1)
        comb_map = _remap_keyarray(comb_map, comb_mask)


        return comb_map, comb_mask, conj_map

    def _sort_pairs(self):
        """Re-order keys into a desired sort order.

        By default the order is lexicographic in (baseline u, baselines v,
        beamclass i, beamclass j).
        """

        # Create mask of included pairs, that are not conjugated
        tmask = np.logical_and(self._feedmask, np.logical_not(self._feedconj))
        uniq = _get_indices(self._feedmap, mask=tmask)

        fi, fj = uniq[:, 0], uniq[:, 1]

        # Fetch keys by which to sort (lexicographically)
        bx = self.feedpositions[fi, 0] - self.feedpositions[fj, 0]
        by = self.feedpositions[fi, 1] - self.feedpositions[fj, 1]
        ci = self.beamclass[fi]
        cj = self.beamclass[fj]

        ## Sort by constructing a numpy array with the keys as fields, and use
        ## np.argsort to get the indices

        # Create array of keys to sort
        dt = np.dtype('f8,f8,i4,i4')
        sort_arr = np.zeros(fi.size, dtype=dt)
        sort_arr['f0'] = bx
        sort_arr['f1'] = by
        sort_arr['f2'] = cj
        sort_arr['f3'] = ci

        # Get map which sorts
        sort_ind = np.argsort(sort_arr)

        # Invert mapping
        tmp_sort_ind = sort_ind.copy()
        sort_ind[tmp_sort_ind] = np.arange(sort_ind.size)

        # Remap feedmap entries
        fm_copy = self._feedmap.copy()
        wmask = np.where(self._feedmask)
        fm_copy[wmask] = sort_ind[self._feedmap[wmask]]

        self._feedmap = fm_copy


    #===================================================




    #==== Methods for calculating Transfer matrices ====

    def transfer_matrices(self, bl_indices, f_indices):
        """Calculate the spherical harmonic transfer matrices for baseline and
        frequency combinations.

        Parameters
        ----------
        bl_indices : array_like
            Indices of baselines to calculate.
        f_indices : array_like
            Indices of frequencies to calculate. Must be broadcastable against
            `bl_indices`.

        Returns
        -------
        transfer : np.ndarray, dtype=np.complex128
            An array containing the transfer functions. The shape is somewhat
            complicated, the first indices correspond to the broadcast size of
            `bl_indices` and `f_indices`, then there may be some polarisation
            indices, then finally the (l,m) indices, range (lside, 2*lside-1).
        """

        ## Check indices are all in range
        if out_of_range(bl_indices, 0, self.npairs):
            raise Exception("Baseline indices aren't valid")

        if out_of_range(f_indices, 0, self.nfreq):
            raise Exception("Frequency indices aren't valid")

        # Generate the array for the Transfer functions
        nfreq = f_indices.size
        nbl = bl_indices.size
        nprod = nfreq * nbl
        indices = list(itertools.product(f_indices, bl_indices))
        lind, sind, eind = mpiutil.split_local(nprod)

        # local section of the array
        tshape = (lind, self.num_pol_sky, self.theta_size, self.phi_size)
        tarray = np.zeros(tshape, dtype=np.complex128)

        for ind, (f_ind, bl_ind) in enumerate(indices[sind:eind]):
            tarray[ind] = self._transfer_single(bl_ind, f_ind, self.lmax)

        return tarray


    def transfer_for_frequency(self, freq):
        """Fetch all transfer matrices for a given frequency.

        Parameters
        ----------
        freq : integer
            The frequency index.

        Returns
        -------
        transfer : np.ndarray
            The transfer matrices. Packed as in `TransitTelescope.transfer_matrices`.
        """
        bi = np.arange(self.npairs)
        fi = freq * np.ones_like(bi)

        return self.transfer_matrices(bi, fi)


    def transfer_for_baseline(self, baseline):
        """Fetch all transfer matrices for a given baseline.

        Parameters
        ----------
        baseline : integer
            The baseline index.

        Returns
        -------
        transfer : np.ndarray
            The transfer matrices. Packed as in `TransitTelescope.transfer_matrices`.
        """
        fi = np.arange(self.nfreq)
        bi = baseline * np.ones_like(fi)

        return self.transfer_matrices(bi, fi)

    #===================================================



    #======== Noise properties of the telescope ========


    def tsys(self, f_indices = None):
        """The system temperature.

        Currenty has a flat T_sys across the whole bandwidth. Override for
        anything more complicated.

        Parameters
        ----------
        f_indices : array_like
            Indices of frequencies to get T_sys at.

        Returns
        -------
        tsys : array_like
            System temperature at requested frequencies.
        """
        if f_indices is None:
            freq = self.frequencies
        else:
            freq = self.frequencies[f_indices]
        return np.ones_like(freq) * self.tsys_flat


    def noise_variance_single(self, bl_index, f_index, nt_per_day, ndays=None):
        """Calculate the instrumental noise variance.

        Parameters
        ----------
        bl_index : integer
            Index of baseline to calculate.
        f_index : integer
            Index of frequencies to calculate.
        nt_per_day : integer
            The number of time samples in one sidereal day.
        ndays : integer
            The number of sidereal days observed.

        Returns
        -------
        noise_var : float
            The noise variance.
        """

        ndays = self.ndays if not ndays else ndays # Set to value if not set.
        t_int = ndays * units.t_sidereal / nt_per_day
        # bw = 1.0e6 * (self.freq_upper - self.freq_lower) / self.num_freq
        bw = np.abs(self.frequencies[1] - self.frequencies[0]) * 1e6

        return 2.0*self.tsys(f_index)**2 / (t_int * bw * self.redundancy[bl_index]) # 2.0 for two pol


    def noise_variance(self, bl_indices, f_indices, nt_per_day, ndays=None):
        """Calculate the instrumental noise variance.

        Parameters
        ----------
        bl_indices : array_like
            Indices of baselines to calculate.
        f_indices : array_like
            Indices of frequencies to calculate. Must be broadcastable against
            `bl_indices`.
        nt_per_day : integer
            The number of time samples in one sidereal day.
        ndays : integer
            The number of sidereal days observed.

        Returns
        -------
        noise_var : np.ndarray
            The noise variance.
        """

        ndays = self.ndays if not ndays else ndays # Set to value if not set.
        t_int = ndays * units.t_sidereal / nt_per_day
        # bw = 1.0e6 * (self.freq_upper - self.freq_lower) / self.num_freq
        bw = np.abs(self.frequencies[1] - self.frequencies[0]) * 1e6

        # Broadcast arrays against each other
        bl_indices, f_indices = np.broadcast_arrays(bl_indices, f_indices)

        return 2.0*self.tsys(f_indices)**2 / (t_int * bw * self.redundancy[bl_indices]) # 2.0 for two pol


    def noise_variance_feedpairs(self, fi, fj, f_indices, nt_per_day, ndays=None):
        ndays = self.ndays if not ndays else ndays # Set to value if not set.
        t_int = ndays * units.t_sidereal / nt_per_day
        # bw = 1.0e6 * (self.freq_upper - self.freq_lower) / self.num_freq
        bw = np.abs(self.frequencies[1] - self.frequencies[0]) * 1e6

        return np.ones_like(fi) * np.ones_like(fj) * 2.0*self.tsys(f_indices)**2 / (t_int * bw) # 2.0 for two pol


    def noisepower(self, bl_indices, f_indices, ndays=None):
        """Calculate the instrumental noise power spectrum.

        Assume we are still within the regime where the power spectrum is white
        in `m` modes.

        Parameters
        ----------
        bl_indices : array_like
            Indices of baselines to calculate.
        f_indices : array_like
            Indices of frequencies to calculate. Must be broadcastable against
            `bl_indices`.
        ndays : integer
            The number of sidereal days observed.

        Returns
        -------
        noise_ps : np.ndarray
            The noise power spectrum.
        """

        ndays = self.ndays if not ndays else ndays # Set to value if not set.

        # Broadcast arrays against each other
        bl_indices, f_indices = np.broadcast_arrays(bl_indices, f_indices)

        bw = np.abs(self.frequencies[1] - self.frequencies[0]) * 1e6
        # bw = 1.0e6 * (self.freq_upper - self.freq_lower) / self.num_freq
        delnu = units.t_sidereal * bw / (2*np.pi)
        noisepower = self.tsys(f_indices)**2 / (2 * np.pi * delnu * ndays)
        noisebase = noisepower / self.redundancy[bl_indices]

        return noisebase


    def noisepower_feedpairs(self, fi, fj, f_indices, m, ndays=None):
        ndays = self.ndays if not ndays else ndays

        bw = np.abs(self.frequencies[1] - self.frequencies[0]) * 1e6
        delnu = units.t_sidereal * bw / (2*np.pi)
        noisepower = self.tsys(f_indices)**2 / (2 * np.pi * delnu * ndays)

        return np.ones_like(fi) * np.ones_like(fj) * np.ones_like(m) * noisepower / 2.0 # For unpolarised only at the moment.

    #===================================================


    _nside = None

    def _init_trans(self, nside):
        ## Internal function for generating some common Healpix maps (position,
        ## horizon). These should need to be generated only when nside changes.

        # Angular positions in healpix map of nside
        self._nside = nside
        self._angpos = hputil.ang_positions(nside)

        # The horizon function
        self._horizon = visibility.horizon(self._angpos, self.zenith)

        # size of the cartesian projection
        phi_size = 5 * self._nside + 1 # add 1 to make it odd for nside >= 2
        theta_range = self.beam_theta_range
        theta_size = ((theta_range[1] - theta_range[0]) / 180.0) * (phi_size / 2)
        theta_size = theta_size if (np.mod(theta_size, 2) == 1) else theta_size+1 # make it odd

        latra = [ 90.0-theta_range[1], 90.0-theta_range[0] ]
        self._cart_projector = hpproj.CartesianProj(xsize=phi_size, ysize=theta_size, latra=latra)



    #===================================================
    #================ ABSTRACT METHODS =================
    #===================================================


    # Implement to specify feed positions in the telescope.
    @abc.abstractproperty
    def feedpositions(self):
        """An (nfeed,2) array of the feed positions relative to an arbitary point (in m)"""
        return

    # Implement to specify the beams of the telescope
    @abc.abstractproperty
    def beamclass(self):
        """An nfeed array of the class of each beam (identical labels are
        considered to have identical beams)."""
        return

    # Implement to specify feed positions in the telescope.
    @abc.abstractproperty
    def u_width(self):
        """The approximate physical width (in the u-direction) of the dish/telescope etc, for
        calculating the maximum (l,m)."""
        return

    # Implement to specify feed positions in the telescope.
    @abc.abstractproperty
    def v_width(self):
        """The approximate physical length (in the v-direction) of the dish/telescope etc, for
        calculating the maximum (l,m)."""
        return



    # The work method which does the bulk of calculating all the transfer matrices.
    @abc.abstractmethod
    def _transfer_single(self, bl_index, f_index, lmax):
        """Calculate transfer matrix for a single baseline+frequency.

        **Abstract method** must be implemented.

        Parameters
        ----------
        bl_index : integer
            The index of the baseline to calculate.
        f_index : integer
            The index of the frequency to calculate.
        lmax : integer
            The maximum *l* we are interested in. Determines accuracy of
            spherical harmonic transforms.

        Returns
        -------
        transfer : np.ndarray
            The transfer matrix, an array of shape (pol_indices, lside,
            2*lside-1). Where the `pol_indices` are usually only present if
            considering the polarised case.
        """
        return


    #===================================================
    #============== END ABSTRACT METHODS ===============
    #===================================================





class UnpolarisedTelescope(TransitTelescope):
    """A base for an unpolarised telescope.

    Again, an abstract class, but the only things that require implementing are
    the `feedpositions`, `_get_unique` and the `beam` function.
    """
    __metaclass__ = abc.ABCMeta

    _npol_sky_ = 1

    @abc.abstractmethod
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
        return


    #===== Implementations of abstract functions =======

    def _beam_map_single(self, bl_index, f_index):

        # Get beam maps for each feed.
        feedi, feedj = self.uniquepairs[bl_index]
        beami, beamj = self.beam(feedi, f_index), self.beam(feedj, f_index)

        # Get baseline separation and fringe map.
        uv = self.baselines[bl_index] / self.wavelengths[f_index]
        fringe = visibility.fringe(self._angpos, self.zenith, uv)

        pxarea = (4 * np.pi / beami.shape[0])

        # Beam solid angle (integrate over beam^2 - equal area pixels)
        om_i = np.sum(np.abs(beami)**2 * self._horizon) * pxarea
        om_j = np.sum(np.abs(beamj)**2 * self._horizon) * pxarea

        omega_A = (om_i * om_j)**0.5

        # Calculate the complex visibility transfer function
        cvis = self._horizon * fringe * beami * beamj.conjugate() / omega_A

        return [ cvis ]


    def _transfer_single(self, bl_index, f_index, lmax):

        if self._nside != hputil.nside_for_lmax(lmax, accuracy_boost=self.accuracy_boost):
            self._init_trans(hputil.nside_for_lmax(lmax, accuracy_boost=self.accuracy_boost))

        cvis = self._beam_map_single(bl_index, f_index)
        beam_cart = hpproj.cartesian_proj(cvis[0], self.cart_projector)

        # Perform the inverse Fourier transform along phi direction to get the transfer matrix
        btrans = np.fft.fft(beam_cart, axis=1) / self.phi_size # m = 0 is at left

        return [ btrans ]


    #===================================================

    def noisepower(self, bl_indices, f_indices, ndays = None):
        """Calculate the instrumental noise power spectrum.

        Assume we are still within the regime where the power spectrum is white
        in `m` modes.

        Parameters
        ----------
        bl_indices : array_like
            Indices of baselines to calculate.
        f_indices : array_like
            Indices of frequencies to calculate. Must be broadcastable against
            `bl_indices`.
        ndays : integer
            The number of sidereal days observed.

        Returns
        -------
        noise_ps : np.ndarray
            The noise power spectrum.
        """

        bnoise = TransitTelescope.noisepower(self, bl_indices, f_indices, ndays)

        return bnoise[..., np.newaxis] * 0.5 # Correction for unpolarisedness




class PolarisedTelescope(TransitTelescope):
    """A base for a polarised telescope.

    Again, an abstract class, but the only things that require implementing are
    the `feedpositions`, `_get_unique` and the beam functions `beamx` and `beamy`.

    Abstract Methods
    ----------------
    beamx, beamy : methods
        Routines giving the field pattern for the x and y feeds.
    """
    __metaclass__ = abc.ABCMeta

    _npol_sky_ = 4



    def _beam_map_single(self, bl_index, f_index):

        p_stokes = [ 0.5 * np.array([[1.0,   0.0], [0.0,  1.0]]),
                     0.5 * np.array([[1.0,   0.0], [0.0, -1.0]]),
                     0.5 * np.array([[0.0,   1.0], [1.0,  0.0]]),
                     0.5 * np.array([[0.0, -1.0J], [1.0J, 0.0]]) ]

        # Get beam maps for each feed.
        feedi, feedj = self.uniquepairs[bl_index]
        beami, beamj = self.beam(feedi, f_index), self.beam(feedj, f_index)

        # Get baseline separation and fringe map.
        uv = self.baselines[bl_index] / self.wavelengths[f_index]
        fringe = visibility.fringe(self._angpos, self.zenith, uv)

        pow_stokes = [ np.sum(beami * np.dot(beamj.conjugate(), polproj), axis=1) * self._horizon for polproj in p_stokes]

        # Calculate the solid angle of each beam
        pxarea = (4*np.pi / beami.shape[0])

        om_i = np.sum(np.abs(beami)**2 * self._horizon[:, np.newaxis]) * pxarea
        om_j = np.sum(np.abs(beamj)**2 * self._horizon[:, np.newaxis]) * pxarea

        omega_A = (om_i * om_j)**0.5

        # Calculate the complex visibility transfer function
        cv_stokes = [ p * (2 * fringe / omega_A) for p in pow_stokes ]

        return cv_stokes


    #===== Implementations of abstract functions =======

    def _transfer_single(self, bl_index, f_index, lmax, lside):

        if self._nside != hputil.nside_for_lmax(lmax):
            self._init_trans(hputil.nside_for_lmax(lmax))

        bmap = self._beam_map_single(bl_index, f_index)

        btrans = [ pb.conj() for pb in hputil.sphtrans_complex_pol([bm.conj() for bm in bmap], centered = False, lmax = int(lmax), lside=lside) ]

        return btrans


    #===================================================


class SimpleUnpolarisedTelescope(UnpolarisedTelescope):
    """A base for a unpolarised telescope.

    Again, an abstract class, but the only things that require implementing are
    the `feedpositions`, `_get_unique` and the beam functions `beamx` and `beamy`.

    Abstract Methods
    ----------------
    beam : method
        Routines giving the field pattern for the feeds.
    """

    __metaclass__ = abc.ABCMeta


    @property
    def beamclass(self):
        """Simple beam mode of single polarisation feeds."""
        return np.zeros(self._single_feedpositions.shape[0], dtype=np.int)


    @abc.abstractproperty
    def _single_feedpositions(self):
        """An (nfeed,2) array of the feed positions relative to an arbitrary point (in m)"""
        return

    @property
    def feedpositions(self):
        return self._single_feedpositions





class SimplePolarisedTelescope(PolarisedTelescope):
    """A base for a polarised telescope.

    Again, an abstract class, but the only things that require implementing are
    the `feedpositions`, `_get_unique` and the beam functions `beamx` and `beamy`.

    Abstract Methods
    ----------------
    beamx, beamy : methods
        Routines giving the field pattern for the x and y feeds.
    """

    __metaclass__ = abc.ABCMeta


    @property
    def beamclass(self):
        """Simple beam mode of dual polarisation feeds."""
        nsfeed = self._single_feedpositions.shape[0]
        return np.concatenate((np.zeros(nsfeed), np.ones(nsfeed))).astype(np.int)


    def beam(self, feed, freq):
        if self.beamclass[feed] % 2 == 0:
            return self.beamx(feed, freq)
        else:
            return self.beamy(feed, freq)

    @abc.abstractproperty
    def _single_feedpositions(self):
        """An (nfeed,2) array of the feed positions relative to an arbitrary point (in m)"""
        return

    @property
    def feedpositions(self):
        return np.concatenate((self._single_feedpositions, self._single_feedpositions))


    @abc.abstractmethod
    def beamx(self, feed, freq):
        """Beam for the X polarisation feed.

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

    @abc.abstractmethod
    def beamy(self, feed, freq):
        """Beam for the Y polarisation feed.

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
