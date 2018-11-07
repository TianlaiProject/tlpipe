import numpy as np
import scipy as sp
from scipy import linalg

from tlpipe.map import algebra as al

from constants import T_infinity, T_huge, T_large, T_medium, T_small, T_sys
from constants import f_medium, f_large

class NoiseError(Exception):
    """Exception to raise if the there is something wrong with the noise and
    this peice of data should be ignored.
    """
    pass


class Noise(object):
    """Object that represents the noise matrix for time stream data.
    
    The noise matrix is represented as separate components each with different
    symetries.  This is so the huge matrix does not have to be stored.

    Parameters
    ----------
    time_strem_data : al.vect object
        The data for which we want to represent the noise matrix.  Only meta
        data is used, not the acctual data.
    time : 1D array
        The time axis of the data.
    """

    # Internal nomiclature: The noise matrix is divided into three parts:  The
    # 'diagonal' contains one weight for every data point and represents a
    # fully diagonal matrix.  It has contributions from thermal noise and
    # deweights masked points.
    # The 'frequency_modes' part is the only part that
    # couples frequencies.  There are only a small number of modes along the
    # frequency axes, but each mode has a full time-time covariance matrix.
    # The modes are assumed to be uncorrelated.  The number of frequency modes
    # is named 'm'
    # The 'time_modes' part deweights certain modes along the time axis.  They
    # are uncorrelated between frequencies and the same for each frequency.
    # The number time modes is named 'q'.
    # The 'update' term will generally refer to the second term in the binomial
    # inverse identity, and the 'update_modes' refers to rotation matrices in
    # the identity.

    # ---- Initialization methods. ----

    def __init__(self, time_stream_data, time):
        if len(time_stream_data.shape) == 2:
            self.n_chan = time_stream_data.shape[0]
            self.n_time = time_stream_data.shape[1]
        elif len(time_stream_data.shape) == 1: 
            self.n_chan = 1
            self.n_time = time_stream_data.shape[0]
            time_stream_data = time_stream_data[None, :]
        else:
            raise ValueError("Only 1D/2D data suported (time)/(freq, time).")
        #self.info = dict(time_stream_data.info)
        self._finalized = False
        # Some functions make this assumption.
        if not sp.alltrue(sp.diff(time) > 0):
            raise ValueError("Time must be monotonically increasing.")
        self.time = time

    def _assert_not_finalized(self):
        """Make sure the noise matrix is not finalized and can be modified."""
        if self._finalized:
            raise AssertionError("Noise model closed for modification.")
    
    def _assert_finalized(self):
        """Make sure the noise matrix is finalized and can not be modified."""
        if not self._finalized:
            raise AssertionError("Noise model still being modified.")

    def initialize_diagonal(self):
        """Create the diagonal part of the noise matrix if it doesn't exist."""
        
        self._assert_not_finalized()
        # TODO: Need to copy the axis info from self.info.
        if hasattr(self, "diagonal"):
            return
        diagonal = sp.zeros((self.n_chan, self.n_time), dtype=float)
        diagonal = al.make_mat(diagonal, axis_names=("freq", "time"), 
                               row_axes=(0, 1), col_axes=(0, 1))
        self.diagonal = diagonal

    def add_time_modes(self, n_new_modes=1):
        """Initialize time noise modes.
        """

        self._assert_not_finalized()
        # TODO: Need to copy the axis info from self.info.
        if hasattr(self, "time_modes"):
            current_q = self.time_modes.shape[0]
            new_q = current_q + n_new_modes
            old_time_modes = self.time_modes
            old_time_mode_noise = self.time_mode_noise
            time_modes = sp.zeros((new_q, self.n_time), 
                                    dtype=float)
            time_mode_noise = sp.zeros((new_q, self.n_chan, self.n_chan),
                                       dtype=float)
            time_modes[:current_q,:] = old_time_modes
            time_mode_noise[:current_q,:,:] = old_time_mode_noise
        else :
            current_q = 0
            time_modes = sp.zeros((n_new_modes, self.n_time), 
                                  dtype=float)
            time_mode_noise = sp.zeros((n_new_modes, self.n_chan, self.n_chan),
                                       dtype=float)
        time_modes = al.make_mat(time_modes, axis_names=("time_mode", "time"), 
                                 row_axes=(0,), col_axes=(1,))
        time_mode_noise = al.make_mat(time_mode_noise,
                                      axis_names=("time_mode", "freq", "freq"), 
                                      row_axes=(0,1), col_axes=(0,2))
        self.time_modes = time_modes
        self.time_mode_noise = time_mode_noise
        return current_q
    
    def add_freq_modes(self, n_new_modes=1):
        """Initialize frequency noise modes.
        """
        
        self._assert_not_finalized()
        # TODO: Need to copy the axis info from self.info.
        if hasattr(self, "freq_modes"):
            current_m = self.freq_modes.shape[0]
            new_m = current_m + n_new_modes
            old_freq_modes = self.freq_modes
            old_freq_mode_noise = self.freq_mode_noise
            freq_modes = sp.zeros((new_m, self.n_chan), 
                                    dtype=float)
            freq_mode_noise = sp.zeros((new_m, self.n_time, self.n_time),
                                       dtype=float)
            freq_modes[:current_m,:] = old_freq_modes
            freq_mode_noise[:current_m,:,:] = old_freq_mode_noise
        else :
            current_m = 0
            freq_modes = sp.zeros((n_new_modes, self.n_chan), 
                                  dtype=float)
            freq_mode_noise = sp.zeros((n_new_modes, self.n_time, self.n_time),
                                       dtype=float)
        freq_modes = al.make_mat(freq_modes, axis_names=("freq_mode", "freq"), 
                                 row_axes=(0,), col_axes=(1,))
        freq_mode_noise = al.make_mat(freq_mode_noise,
                                      axis_names=("freq_mode", "time", "time"), 
                                      row_axes=(0,1), col_axes=(0,2))
        self.freq_modes = freq_modes
        self.freq_mode_noise = freq_mode_noise
        return current_m

    # ---- Methods that build up the noise matrix. ----

    def add_thermal(self, thermal_levels):
        """Add a thermal component to the noise.
        
        This modifies the diagonal part of the noise matrix.

        Parameters
        ----------
        thermal_levels : 1D array
            The noise level of each channel (frequency).  This should be the
            thermal variance, in K**2, not K**2/Hz.
        """
        
        self.initialize_diagonal()
        if (not sp.all(sp.isfinite(thermal_levels))
            or sp.any(thermal_levels < T_small**2)):
            raise ValueError("Non finite thermal noise.")
        if isinstance(thermal_levels, sp.ndarray):
            self.diagonal += thermal_levels[:, None]
        else:
            self.diagonal += thermal_levels

    def add_mask(self, mask_inds):
        """Add a mask to the noise.
        
        This modifies the diagonal part of the noise matrix.

        Parameters
        ----------
        mask_inds : 2 element tuple of integer arrays.
            The locations of data points to be masked.
        """
        
        self.initialize_diagonal()
        self.diagonal[mask_inds] += T_infinity**2

    def get_mean_mode(self):
        mode = sp.empty(self.n_time, dtype=float)
        mode[:] = 1.0 / sp.sqrt(self.n_time)
        return mode
    
    def get_slope_mode(self):
        mode = self.time - sp.mean(self.time)
        mode *= 1.0 / sp.sqrt(sp.sum(mode**2)) 
        return mode

    def deweight_time_mean(self, T=T_huge**2):
        """Deweights time mean in each channel.

        This modifies the part of the noise matrix that is the same for each
        channel.
        """
        
        mode = self.get_mean_mode()
        self.deweight_time_mode(mode, T=T)

    def deweight_time_slope(self, T=T_huge**2):
        """Deweights time slope in each channel.
        
        This modifies the part of the noise matrix that is the same for each
        channel.
        """

        mode = self.get_slope_mode()
        self.deweight_time_mode(mode, T=T)

    def deweight_time_mode(self, mode, T=T_huge**2):
        """Dewieghts a given time mode in each channel.
        
        This modifies the part of the noise matrix that is the same for each
        channel.
        """

        start = self.add_time_modes(1)
        self.time_modes[start,:] = mode
        self.time_mode_noise[start,...] = (sp.eye(self.n_chan, dtype=float) 
                                           * T * self.n_time)

    def orthogonalize_mat_mean_slope(self, mat):
        """Removes the time mode and the slope mode from a matrix."""

        if mat.shape != (self.n_time, self.n_time):
            msg = "Expected a time-time matrix."
            raise ValueError(msg)
        modes = [self.get_mean_mode(), self.get_slope_mode()]
        for mode in modes:
            tmp1 = np.sum(mat * mode, 1)
            tmp2 = np.sum(mat * mode[:,None], 0)
            tmp3 = np.sum(tmp2 * mode)
            mat[:,:] -= tmp1[:,None] * mode
            mat[:,:] -= tmp2[None,:] * mode[:,None]
            mat[:,:] += (tmp3 * mode[:,None] * mode[None,:])
        return mat

    def add_correlated_over_f(self, amp, index, f0):
        """Add 1/f noise that is perfectly correlated between frequencies.
        
        # XXX: Out of date.

        This modifies the the correlated mode part of the noise matrix. It adds
        a single mode with equal amplitude in all frequencies.

        Parameters
        ----------
        amp : float
            The amplitude of the noise at `f0`.
        index : float
            The spectral index of the spectrum (normaly near -1).
        f0 : float
            The pivot frequency.
        """
        
        mode = 1.0 / sp.sqrt(self.n_chan)
        thermal = 0.
        # Multiply by number of channels since this is the mean, not the sum
        # (Sept 15, 2011 in Kiyo's notes).
        amp *= self.n_chan
        self.add_over_f_freq_mode(amp, index, f0, thermal, mode)
        return
        
    def add_over_f_freq_mode(self, amp, index, f0, thermal, mode,
                             ortho_mean_slope=False):
        """Add `1/f + const` noise to a given frequency mode."""
        
        # Too steep a spectra will crash finialize.
        if index < -3.5:
            print "Index:", index
            raise NoiseError("Extremely steep index risks singular noise.")
        time = self.time
        # Build the matrix.
        time_deltas = abs(time[:, None] - time)
        # Smallest time step.
        dt = sp.amin(abs(sp.diff(time)))
        # Time step for calculating the correlation function.
        # Over sample for precise interpolation.
        dt_calc = dt / 4
        n_lags = sp.amax(time_deltas) // dt_calc + 5
        # Calculate the correlation function at these lags.
        correlation_function = noise_power.calculate_overf_correlation(amp, 
            index, f0, dt_calc, n_lags)
        # If we are adding too much noise, we risk making the matrix singular.
        if sp.amax(correlation_function) > T_huge**2:
            print "Freq mode max:", sp.amax(correlation_function)
            raise NoiseError("Extremely high 1/f risks singular noise.")
        start_mode = self.add_freq_modes(1)
        self.freq_modes[start_mode,:] = mode
        corr_func_interpolator = \
            interpolate.interp1d(sp.arange(n_lags) * dt_calc,
                                 correlation_function, kind='linear')
        noise_mat = corr_func_interpolator(time_deltas)
        if ortho_mean_slope:
            # This greatly improves numerical stability.
            noise_mat = self.orthogonalize_mat_mean_slope(noise_mat)
        # Add the thermal part to the diagonal.
        BW = 1. / 2. / dt
        thermal_var = thermal * BW * 2
        # Minimum noise level for numerical stability.
        thermal_var = max((thermal_var, T_small**2))
        noise_mat.flat[::self.n_time + 1] += thermal_var
        self.freq_mode_noise[start_mode,...] = noise_mat
        # XXX
        if not hasattr(self, 'debug'):
            self.debug = []
        this_debug = {'amp' : amp, 'index' : index, 'f0' : f0,
                      'thermal' : thermal_var,
                      "max_corr" :  sp.amax(correlation_function)}
        self.debug.append(this_debug)
        
    def deweight_freq_mode(self, mode, T=T_huge**2, ortho_mean_slope=False):
        """Completly deweight a frequency mode."""
        
        n_chan = self.n_chan
        n_time = self.n_time
        start_mode = self.add_freq_modes(1)
        self.freq_modes[start_mode,:] = mode
        noise_mat = (sp.eye(n_time, dtype=float) * T * n_chan)
        if ortho_mean_slope:
            # This greatly improves numerical stability.
            noise_mat = self.orthogonalize_mat_mean_slope(noise_mat)
        # Add a bit to diagonal for conditioning.
        noise_mat.flat[::n_time + 1] += T_small**2
        self.freq_mode_noise[start_mode,...]  = noise_mat

    def add_all_chan_low(self, amps, index, f_0):
        """Deweight frequencies below, and a bit above 'f_0', 
        with 1/f like noise."""
        
        time = self.time
        n_chan = self.n_chan
        f_0 = float(f_0)
        # A bit of extra time range so the map maker doesn't think the modes
        # are periodic.
        extra_time_factor = 1.25
        time_extent = (time[-1] - time[0]) * extra_time_factor
        # Normalize the time to -pi <= t < pi
        time_normalized = time - time[0] - time_extent / 2.
        time_normalized *= 2 * sp.pi / time_extent
        df = 1. / time_extent
        # If there are no frequencies below f_0, do nothing.
        if f_0 * 1.5 <= df :
            return
        # If a channel is already deweighted completly, do nothing to it.
        # XXX BW factors.
        amps = amps.copy()
        amps[amps > T_huge**2] = T_small**2
        # Figure out how many modes to deweight.
        # I could concievably make the cut-off index dependant.
        frequencies = sp.arange(df, f_0 * 2.0, df)
        n_f = len(frequencies)
        n_modes = 2 * n_f
        if n_modes > self.n_time * 0.15:
            print f_0, n_modes, self.n_time
            raise NoiseError("To many time modes to deweight.")
        # Allowcate mememory for the new modes.
        start_mode = self.add_time_modes(n_modes)
        # Loop through and fill the modes.
        for ii, f in enumerate(frequencies):
            this_amp_factor = (f / f_0)**index * df
            this_amps = this_amp_factor * amps
            if sp.any(this_amps > T_large**2):
                print "time mode amplitude:", this_amps
                raise NoiseError("Deweighting amplitude too high.")
            # The cosine (symetric) mode.
            cos_mode = sp.cos((ii+1) * time_normalized)
            norm_cos = sp.sqrt(sp.sum(cos_mode**2))
            cos_mode /= norm_cos
            cos_amp = this_amps * norm_cos**2
            self.time_modes[start_mode + 2*ii,:] = cos_mode
            cos_noise = sp.zeros((n_chan, n_chan), dtype=float)
            cos_noise.flat[::n_chan + 1] = cos_amp
            self.time_mode_noise[start_mode + 2*ii,:,:] = cos_noise
            # The sine (antisymetric) mode.
            sin_mode = sp.sin((ii+1) * time_normalized)
            norm_sin = sp.sqrt(sp.sum(sin_mode**2))
            sin_mode /= norm_sin
            sin_amp = this_amps * norm_sin**2
            self.time_modes[start_mode + 2*ii + 1,:] = sin_mode
            sin_noise = sp.zeros((n_chan, n_chan), dtype=float)
            sin_noise.flat[::n_chan + 1] = sin_amp
            self.time_mode_noise[start_mode + 2*ii + 1,:,:] = sin_noise

    def orthogonalize_modes(self):
        """
        Orthogonalize modes with high overlap, for numerical stability.
        
        XXX: Not used and untested.

        Look for modes with both high noise and high overlap with other modes
        of high noise. Move the noise into one mode or the other to improve
        numerical stability.

        Note that this method does change the noise model.  It decreases the
        overall information and as such is conservative.  In addition it should
        only affect very noisy modes.
        """

        m = self.time_modes.shape[0]
        q = self.freq_modes.shape[0]
        # Find time modes with big noise.
        hi_noise_time_modes = []
        for ii in range(m):
            if np.any(self.time_mode_noise[ii].flat[::self.n_chan + 1]
                      > 0.9 * T_large**2):
                hi_noise_time_modes.append(ii)
        # For each frequency mode, check the overlap with the hi noise time
        # modes.
        print hi_noise_time_modes
        for ii in range(q):
            freq_mode_noise = self.freq_mode_noise[ii]
            freq_mode = self.freq_modes[ii]
            for jj in hi_noise_time_modes:
                time_mode = self.time_modes[jj]
                amp = np.sum(time_mode
                             * np.sum(freq_mode_noise * time_mode, 1))
                print jj, amp
                #if amp > T_large**2: # Very noisy mode.
                if True:
                    # Subtract this mode out of the freq mode noise.
                    tmp = np.sum(freq_mode_noise * time_mode, 1)
                    tmp2 = np.sum(freq_mode_noise * time_mode[:,None], 0)
                    freq_mode_noise[:,:] -= tmp[:,None] * time_mode
                    freq_mode_noise[:,:] -= tmp2[None,:] * time_mode[:,None]
                    freq_mode_noise[:,:] += (amp * time_mode[:,None]
                                             * time_mode[None,:])
                    # Add the subtracted noise into the time mode noise.  Add
                    # it in as diagonal even though it isn't (conservative).
                    self.time_mode_noise[jj,:,:] += np.diag(amp * freq_mode**2)
        # TODO: Similar proceedure could be done with time_modes and freq_modes
        # reversed. Also overlap between time_modes and time_modes as well as
        # freq_modes and freq_modes.

    def finalize(self, frequency_correlations=True, preserve_matrices=True):
        """Tell the class that you are done building the matrix.
        """
        
        # Flag for performing extra checking and debugging.
        CHECKS = False

        self._assert_not_finalized()
        n_time = self.n_time
        n_chan = self.n_chan
        # This diagonal part must be set or the matrix will be singular.
        if not hasattr(self, 'diagonal'):
            raise RuntimeError("Diagonal noise component not set.")
        diagonal_inv = self.diagonal**-1
        self.diagonal_inv = al.as_alg_like(diagonal_inv, self.diagonal)
        if frequency_correlations:
            self._frequency_correlations = True
            if not hasattr(self, "freq_modes"):
                self.add_freq_modes(0)
            if not hasattr(self, "time_modes"):
                self.add_time_modes(0)
            # Calculate the inverses of all matricies.
            freq_mode_inv = al.empty_like(self.freq_mode_noise)
            for ii in xrange(self.freq_mode_noise.shape[0]):
                freq_mode_inv[ii,...] = \
                        scaled_inv(self.freq_mode_noise[ii,...])
                # Check that its positive definiate.
                A = self.freq_mode_noise[ii].view()
                A.shape = (n_time,) * 2
                if get_scaled_cond_h(A) > 1.e11 :
                    e, v = linalg.eigh(A)
                    print "Freq cond:", max(e)/min(e), get_scaled_cond_h(A)
                    print self.debug
                    msg = ("Some freq_mode noise components not positive"
                           " definate.")
                    raise RuntimeError(msg)
                if CHECKS:
                    e, v = linalg.eigh(A)
                    print "freq_mode eigs:", min(e), max(e)
                    print "Initial freq_mode condition number:", max(e)/min(e)
            time_mode_inv = al.empty_like(self.time_mode_noise)
            for ii in xrange(self.time_mode_noise.shape[0]):
                time_mode_inv[ii,...] = scaled_inv(
                    self.time_mode_noise[ii,...])
                A = self.time_mode_noise[ii].view()
                A.shape = (n_chan,) * 2
                if get_scaled_cond_h(A) > 1.e11 :
                    e, v = linalg.eigh(A)
                    print "Time cond:", max(e)/min(e), get_scaled_cond_h(A)
                    msg = ("Some time_mode noise components not positive"
                           " definate.")
                    raise RuntimeError(msg)
                if CHECKS:
                    e, v = linalg.eigh(A)
                    print "time_mode eigs:", min(e), max(e)
                    print "Initial time_mode condition number:", max(e)/min(e)
            # The normal case when we are considering the full noise.
            # Calculate the term in the bracket in the matrix inversion lemma.
            # Get the size of the update term.
            # First, the rank of the correlated frequency part.
            m = self.freq_modes.shape[0]
            n_update =  m * n_time
            # Next, the rank of the all frequencies part.
            q = self.time_modes.shape[0]
            n_update += q * n_chan
            # Build the update matrix in blocks.
            freq_mode_update = sp.zeros((m, n_time, m, n_time), dtype=float)
            freq_mode_update = al.make_mat(freq_mode_update, 
                axis_names=('freq_mode', 'time', 'freq_mode', 'time'),
                row_axes=(0, 1), col_axes=(2, 3))
            cross_update = sp.zeros((m, n_time, q, n_chan), dtype=float)
            cross_update = al.make_mat(cross_update, 
                axis_names=('freq_mode', 'time', 'time_mode', 'freq'),
                row_axes=(0, 1), col_axes=(2, 3))
            time_mode_update = sp.zeros((q, n_chan, q, n_chan), dtype=float)
            time_mode_update = al.make_mat(time_mode_update, 
                axis_names=('time_mode', 'freq', 'time_mode', 'freq'),
                row_axes=(0, 1), col_axes=(2, 3))
            # Build the matrices.
            # Transform the diagonal noise to this funny space and add
            # it to the update term. Do this one pair of modes at a time
            # to make things less complicated.
            for ii in xrange(m):
                for jj in xrange(m):
                    tmp_freq_update = sp.sum(self.freq_modes[ii,:,None]
                                             * self.freq_modes[jj,:,None]
                                             * diagonal_inv[:,:], 0)
                    freq_mode_update[ii,:,jj,:].flat[::n_time + 1] += \
                            tmp_freq_update
            for ii in xrange(m):
                for jj in xrange(q):
                    tmp_cross_update = (self.freq_modes[ii,None,:]
                                        * self.time_modes[jj,:,None]
                                        * diagonal_inv.transpose())
                    cross_update[ii,:,jj,:] += tmp_cross_update
            for ii in xrange(q):
                for jj in xrange(q):
                    tmp_time_update = sp.sum(self.time_modes[ii,None,:]
                                             * self.time_modes[jj,None,:]
                                             * diagonal_inv[:,:], 1)
                    time_mode_update[ii,:,jj,:].flat[::n_chan + 1] += \
                            tmp_time_update
            if CHECKS:
                # Make a copy of these for testing.
                diag_freq_space = freq_mode_update.copy()
                diag_time_space = time_mode_update.copy()
                diag_cross_space = cross_update.copy()
            # Add the update mode noise in thier proper space.
            for ii in range(m):
                freq_mode_update[ii,:,ii,:] += freq_mode_inv[ii,:,:]
            for ii in range(q):
                time_mode_update[ii,:,ii,:] += time_mode_inv[ii,:,:]
            # Put all the update terms in one big matrix and invert it.
            update_matrix = sp.empty((n_update, n_update), dtype=float)
            # Top left.
            update_matrix[:m * n_time,:m * n_time].flat[...] = \
                freq_mode_update.flat
            # Bottom right.
            update_matrix[m * n_time:,m * n_time:].flat[...] = \
                time_mode_update.flat
            # Top right.
            update_matrix[:m * n_time,m * n_time:].flat[...] = \
                cross_update.flat
            # Bottom left.
            tmp_mat = sp.swapaxes(cross_update, 0, 2)
            tmp_mat = sp.swapaxes(tmp_mat, 1, 3)
            update_matrix[m * n_time:,:m * n_time].flat[...] = \
                tmp_mat.flat
            update_matrix_inv = scaled_inv(update_matrix)
            if CHECKS:
                diag_space = sp.empty((n_update, n_update), dtype=float)
                # Top left.
                diag_space[:m * n_time,:m * n_time].flat[...] = \
                    diag_freq_space.flat
                # Bottom right.
                diag_space[m * n_time:,m * n_time:].flat[...] = \
                    diag_time_space.flat
                # Top right.
                diag_space[:m * n_time,m * n_time:].flat[...] = \
                    diag_cross_space.flat
                # Bottom left.
                tmp_mat = sp.swapaxes(diag_cross_space, 0, 2)
                tmp_mat = sp.swapaxes(tmp_mat, 1, 3)
                diag_space[m * n_time:,:m * n_time].flat[...] = \
                    tmp_mat.flat
                e, v = linalg.eig(diag_space)
                print "rotated diagonal eigs:", min(e.real), max(e.real)
                subtraction_term = sp.dot(update_matrix_inv, diag_space)
                e, v = linalg.eig(subtraction_term)
                print "cond:",  1. - max(e.real)
                print "reduced update eigs:", min(e.real), max(e.real)
                if 1. - max(e.real) < 1e-7 or max(e.real) < 0.9:
                    print "Whao!!!"
                    print 1. - max(e)
                    print n_time, n_chan, m, q
                    #time_mod.sleep(300)
                    #raise NoiseError('Negitive eigenvalue detected.')
            # A condition number check on the update matrix.
            if CHECKS:
                e = linalg.eigvalsh(update_matrix)
                print "Update eigs:", min(e), max(e), max(e)/min(e)
            if get_scaled_cond_h(update_matrix) > 1e11:
                msg = "Update term too ill conditioned."
                raise NoiseError(msg)
            # Copy the update terms back to thier own matrices and store them.
            freq_mode_update.flat[...] = \
                    update_matrix_inv[:m * n_time,:m * n_time].flat
            self.freq_mode_update = freq_mode_update
            time_mode_update.flat[...] = \
                    update_matrix_inv[m * n_time:,m * n_time:].flat
            self.time_mode_update = time_mode_update
            cross_update.flat[...] = \
                    update_matrix_inv[:m * n_time,m * n_time:].flat
            self.cross_update = cross_update
            # Set flag so no more modifications to the matricies can occure.
            self._finalized = True
            # Check that the diagonal is positive to catch catastrophic
            # inversion faileurs.
            self.check_inv_pos_diagonal()
        else:
            # Ignore the channel correlations in the noise.  Ignore freq_modes.
            self._frequency_correlations = False
            # TODO: raise a warning (as opposed to printing)?
            if hasattr(self, "freq_modes"):
                print "Warning, frequency mode noise ignored."
            # Calculate a time mode update term for each frequency.
            q = self.time_modes.shape[0]
            # First get the diagonal inverse of the time mode noise.
            time_mode_noise_diag = self.time_mode_noise.view()
            time_mode_noise_diag.shape = (q, self.n_chan**2)
            time_mode_noise_diag = time_mode_noise_diag[:,::self.n_chan + 1]
            time_mode_noise_inv = 1.0/time_mode_noise_diag
            # Allocate memory for the update term.
            time_mode_update = sp.zeros((self.n_chan, q, q), dtype=float)
            time_mode_update = al.make_mat(time_mode_update,
                        axis_names=('freq', 'time_mode', 'time_mode'),
                        row_axes=(0, 1), col_axes=(0,2))
            # Add in the time mode noise.
            for ii in range(q):
                for jj in range(self.n_chan):
                    time_mode_update[jj,ii,ii] += time_mode_noise_inv[ii,jj]
            # Transform the diagonal noise to time_mode space and add it in.
            for ii in range(self.n_chan):
                time_mode_update[ii,...] += sp.sum(self.diagonal_inv[ii,:] 
                                                   * self.time_modes[:,None,:] 
                                                   * self.time_modes[None,:,:],
                                                   -1)
                time_mode_update[ii,...] = scaled_inv(time_mode_update[ii,...])
            self.time_mode_update = time_mode_update
            # Set flag so no more modifications to the matricies can occur.
            self._finalized = True
        # If desired, delete the noise matrices to recover memory.  Doing
        # this means the Noise cannot be 'unfinalized'.
        if not preserve_matrices:
            if hasattr(self, "freq_mode_noise"):
                del self.freq_mode_noise
            del self.time_mode_noise

    def check_inv_pos_diagonal(self, thres=-1./T_huge**2):
        """Checks the diagonal elements of the inverse for positiveness.
        """
        
        noise_inv_diag = self.get_inverse_diagonal()
        if sp.any(noise_inv_diag < thres):
            print (sp.sum(noise_inv_diag < 0), noise_inv_diag.size)
            print (noise_inv_diag[noise_inv_diag < 0], sp.amax(noise_inv_diag))
            time_mod.sleep(300)
            raise NoiseError("Inverted noise has negitive entries on the "
                             "diagonal.")

    # ---- Methods for using the Noise Matrix. ----

    def get_mat(self):
        """Get dense representation of noise matrix.
        
        Not particularly useful except for testing.
        """
 
        n_chan = self.n_chan
        n_time = self.n_time
        n = n_chan * n_time
        freq_modes = self.freq_modes
        time_modes = self.time_modes
        # Allowcate memory.
        out = sp.zeros((n_chan, n_time, n_chan, n_time), dtype=float)
        out = al.make_mat(out, axis_names=('freq','time','freq','time'),
                          row_axes=(0, 1), col_axes=(2, 3))
        # Add the diagonal part.
        if hasattr(self, 'diagonal'):
             out.flat[::n + 1] += self.diagonal.flat
        # Time mode update part.
        if hasattr(self, 'time_mode_noise'):
            m = self.time_modes.shape[0]
            for ii in range(m):
                mode = self.time_modes[ii,:]
                out += (self.time_mode_noise[ii,:,None,:,None]
                        * mode[:,None,None] * mode)
        # Time mode update part.
        if hasattr(self, 'freq_mode_noise'):
            q = self.freq_modes.shape[0]
            for ii in range(q):
                mode = self.freq_modes[ii,:]
                out += (self.freq_mode_noise[ii,None,:,None,:]
                        * mode[:,None,None,None] * mode[:,None])
        return out

    def get_inverse_diagonal(self):
        self._assert_finalized()
        return _mapmaker_c.get_noise_inv_diag(self.diagonal_inv,
                    self.freq_modes, self.time_modes, self.freq_mode_update,
                    self.time_mode_update, self.cross_update)

    def get_inverse(self):
        """Get the full noise inverse.

        This function is more for testing than accually being usefull (since in
        production we will only use part of the inverse at a time).
        """
        
        self._assert_finalized()
        n_chan = self.n_chan
        n_time = self.n_time
        if self._frequency_correlations:
            if hasattr(self, 'flag'):
                e = self.diagonal_inv.flat[:]
                print "Diagonal condition number:", max(e)/min(e)
            freq_modes = self.freq_modes
            time_modes = self.time_modes
            # Get the size of the update term.
            # First, the rank of the correlated frequency part.
            m = self.freq_modes.shape[0]
            n_update =  m * n_time
            # Next, the rank of the all frequencies part.
            q = self.time_modes.shape[0]
            n_update += q * n_chan
            # Allowcate memory.
            out = sp.zeros((n_chan, n_time, n_chan, n_time), dtype=float)
            out = al.make_mat(out, axis_names=('freq','time','freq','time'),
                              row_axes=(0, 1), col_axes=(2, 3))
            # Loop over the frequency indeces to reduce workspace memory and
            # for simplicity.
            for ii in xrange(n_chan):
                this_freq_modes1 = freq_modes.index_axis(1, ii)
                for jj in xrange(n_chan):
                    # Get only the matricies that apply to this slice.
                    this_freq_modes2 = freq_modes.index_axis(1, jj)
                    this_cross1 = self.cross_update.index_axis(3, ii)
                    this_cross2 = self.cross_update.index_axis(3, jj)
                    this_time_update = self.time_mode_update.index_axis(1, ii)
                    this_time_update = this_time_update.index_axis(2, jj)
                    # The freq_mode-freq_mode block of the update term.
                    tmp_mat = al.partial_dot(this_freq_modes1, 
                                             self.freq_mode_update)
                    tmp_mat = al.partial_dot(tmp_mat, this_freq_modes2)
                    out[ii,:,jj,:] -= tmp_mat
                    # The off diagonal blocks.
                    tmp_mat = al.partial_dot(this_cross1, time_modes)
                    tmp_mat = al.partial_dot(this_freq_modes2, tmp_mat)
                    out[ii,:,jj,:] -= tmp_mat.transpose()
                    tmp_mat = al.partial_dot(this_freq_modes1, this_cross2)
                    tmp_mat = al.partial_dot(tmp_mat, time_modes)
                    out[ii,:,jj,:] -= tmp_mat
                    # Finally the time_mode-time_mode part.
                    tmp_mat = al.partial_dot(time_modes.mat_transpose(), 
                                             this_time_update)
                    tmp_mat = al.partial_dot(tmp_mat, time_modes)
                    out[ii,:,jj,:] -= tmp_mat
                    # Multply one side by the diagonal.
                    out[ii,:,jj,:] *= self.diagonal_inv[ii,:,None]
            # Add the identity.
            out.flat[::n_chan * n_time + 1] += 1.0
            # Multiply by the thermal term.
            out[:,:,:,:] *= self.diagonal_inv[:,:]
        else:
            time_modes = self.time_modes
            time_mode_update = self.time_mode_update
            diagonal_inv = self.diagonal_inv
            q = self.time_modes.shape[0]
            # Allocate memory for the output.
            out = sp.zeros((n_chan, n_time, n_time), dtype=float)
            out = al.make_mat(out, axis_names=('freq','time','time'),
                              row_axes=(0, 1), col_axes=(0, 2))
            # First get the update term.
            tmp_update = al.partial_dot(time_modes.mat_transpose(),
                                        time_mode_update)
            out -= al.partial_dot(tmp_update, time_modes)
            # Multiply both sides by the diagonal.
            out *= diagonal_inv[:,:,None]
            out *= diagonal_inv[:,None,:]
            # Finally add the diagonal in.
            out.shape = (n_chan, n_time**2)
            out[:,::n_time + 1] += diagonal_inv
            out.shape = (n_chan, n_time, n_time)
        return out

    def weight_time_stream(self, data):
        """Noise weight a time stream data vector.
        """
        
        self._assert_finalized()
        time_modes = self.time_modes
        time_mode_update = self.time_mode_update
        diagonal_inv = self.diagonal_inv
        # Noise weight by the diagonal part of the noise.
        # These two lines replace an algebra library function.  They are much
        # faster for this case.
        diag_weighted = diagonal_inv * data
        diag_weighted = al.as_alg_like(diag_weighted, data)
        # Calculate the update term carrying the freq modes and the time modes
        # through separately.
        # Transform to the update space.
        tmp_update_term_time = al.partial_dot(time_modes, diag_weighted)
        # Multiply by the update matrix.
        update_term_time = al.partial_dot(time_mode_update,
                                          tmp_update_term_time)
        if self._frequency_correlations:
            # Calculate terms that couple frequencies.
            freq_modes = self.freq_modes
            freq_mode_update = self.freq_mode_update
            cross_update = self.cross_update
            # Transform to the update space.
            tmp_update_term_freq = al.partial_dot(freq_modes, diag_weighted)
            # Multiply by the update matrix.
            update_term_freq = (al.partial_dot(freq_mode_update, 
                                               tmp_update_term_freq)
                                + al.partial_dot(cross_update,
                                                 tmp_update_term_time))
            update_term_time += al.partial_dot(cross_update.mat_transpose(),
                                               tmp_update_term_freq)
            # Transform back.
            update_term_freq = al.partial_dot(freq_modes.mat_transpose(),
                                              update_term_freq)
            update_term_time = al.partial_dot(time_modes.mat_transpose(),
                                              update_term_time)
            # Combine.
            update_term = update_term_freq + update_term_time.transpose()
        else:
            # Transform back.
            update_term_time = al.partial_dot(time_modes.mat_transpose(),
                                              update_term_time)
            update_term = update_term_time
        # Final multiply by the diagonal component.
        update_term = al.partial_dot(diagonal_inv, update_term)
        # Combine the terms.
        out = diag_weighted - update_term
        return out 


def scaled_inv(mat):
    """Performs the matrix inverse by first scaling by the diagonal.

    This performs a scaling operation on a matrix before inverting it and then
    applies the appropriate scaling back to the inverse. This opperation should
    improve the conditioning on symetric positive definate matricies such as
    covariance matricies.
    """

    n = mat.shape[0]
    if mat.shape != (n, n):
        raise ValueError("Matrix must be square.")
    diag = abs(mat.flat[::n + 1])
    scal_inv = sp.sqrt(diag)
    scal = 1. / scal_inv
    out_mat = scal[:,None] * mat * scal[None,:]
    out_mat = linalg.inv(out_mat)
    out_mat = scal[:,None] * out_mat * scal[None,:]
    return out_mat


