import abc
import numpy as np
from tlpipe.utils.robust_stats import winsorized_mean_and_std, winsorized_mode


class CombinatorialThreshold(object):
    """Abstract base class for combinatorial thresholding methods.

    The method will flag a combination of samples when a property of this
    combination exceeds some limit. The more connected samples are combined,
    the lower the sample threshold.

    For more details, see Offringa et al., 2000, MNRAS, 405, 155,
    *Post-correlation radio frequency interference classification methods*.

    For this implementation, the sequence of thresholds are determined by
    the following formula:

    .. math:: \\alpha \\times \\frac{\\rho^{i}}{w} \\times (\\sigma \\times \\beta) + \\eta

    in which :math:`\\alpha` is the first threshold set to 6.0,
    :math:`\\rho = 1.5`, :math:`i` is the current iteration, :math:`w` the
    current window size, :math:`\\sigma` the standard deviation of the values,
    :math:`\\beta` the base sensitivity, set to 1.0, and :math:`\\eta` the
    median.

    """

    def __init__(self, time_freq_vis, time_freq_vis_mask=None, first_threshold=6.0, exp_factor=1.5, distribution='Rayleigh', max_threshold_length=1024):

        self.vis = time_freq_vis
        nt, nf = self.vis.shape

        if time_freq_vis_mask is None:
            self.vis_mask = np.where(np.isfinite(self.vis), False, True)
        elif self.vis.shape == time_freq_vis_mask.shape:
            self.vis_mask = time_freq_vis_mask.astype(np.bool)
        else:
            raise ValueError('Invalid time_freq_vis_mask')

        max_log2_length = np.int(np.ceil(np.log2(max_threshold_length))) + 1
        time_lengths = [ 2**i for i in xrange(max_log2_length) ]
        freq_lengths = [ 2**i for i in xrange(max_log2_length) ]
        # include nt, nf in lengths
        if nt < max_threshold_length:
            time_lengths.append(nt)
        if nf < max_threshold_length:
            freq_lengths.append(nf)
        self.time_lengths = np.unique(sorted(time_lengths))
        self.freq_lengths = np.unique(sorted(freq_lengths))

        if distribution in ('Uniform', 'Gaussian', 'Rayleigh'):
            self.distribution = distribution
        else:
            raise ValueError('Invalid noise distribution %s' % distribution)

        if first_threshold is None:
            self.init_threshold_with_flase_rate(resolution, false_alarm_rate)
        else:
            # self.thresholds = first_threshold / exp_factor**(np.log2(self.lengths))
            # self.thresholds = first_threshold * exp_factor**(np.log2(self.lengths)) / self.lengths # used in aoflagger
            self.time_thresholds = first_threshold * exp_factor**(np.log2(self.time_lengths)) / self.time_lengths # used in aoflagger
            self.freq_thresholds = first_threshold * exp_factor**(np.log2(self.freq_lengths)) / self.freq_lengths # used in aoflagger

    def init_threshold_with_flase_rate(self, resolution, false_alarm_rate):
        raise NotImplementedError('Not implemented yet')


    @abc.abstractmethod
    def execute_threshold(self, factor, direction):
        """Abstract method that needs to be implemented by sub-classes."""
        return

    def execute(self, sensitivity=1.0, direction=('time', 'freq')):
        """Execute the thresholding method."""

        if self.distribution == 'Gaussian':
            mean, std = winsorized_mean_and_std(np.ma.array(self.vis, mask=self.vis_mask))
            factor = sensitivity if std == 0.0 else std * sensitivity
        elif self.distribution == 'Rayleigh':
            mode = winsorized_mode(np.ma.array(self.vis, mask=self.vis_mask))
            factor = sensitivity if mode == 0.0 else mode * sensitivity
        else:
            factor = sensitivity

        self.execute_threshold(factor, direction)