"""Robust statistical utilities.

This implements the Median Absolute Deviation (MAD) and some Winsorized
statistical methods.

A sample :math:`x_1, \\cdots, x_n` is sorted in ascending order. For the
chosen :math:`0 \\le \\gamma \\le 0.5` and :math:`k = [\\gamma n]`
winsorization of the sorted data consists of setting

.. math:: W_i = \\left \\{ \\begin{array}{lll}
         x_{(k+1)}, & \\mbox{ if } & x_{(i)} \\le x_{(k+1)} \\\\
         x_{(i)}, & \\mbox{ if } & x_{(k+1)} \\le x_{(i)} \\le x_{(n-k)} \\\\
         x_{(n-k)}, & \\mbox{ if } & x_{(i)} \\ge x_{(n-k)}.
                \\end{array} \\right.

The winsorized sample mean is :math:`\\hat{\\mu}_w = \\frac{1}{n}
\\sum_{i=1}^{n} W_i` and the winsorized sample variance is
:math:`D_w = \\frac{1}{n-1} \\sum_{i=1}^{n} (W_i - \\hat{\\mu}_w)^2`.

For this implementation, the statistics is computed for winsorized data
with :math:`\\gamma = 0.1`.

"""

import numpy as np
from scipy.stats.mstats import winsorize



def mad(a):
    """Median absolute deviation."""
    return np.median(np.abs(a - np.median(a)))

def MAD(a):
    """Median absolute deviation divides 0.6745."""
    return mad(a) / 0.6745


def _winsorize(a, limits=None, inclusive=(True, True)):
    # drop masked data
    a1 = np.ma.compressed(a)

    # use .data to return an np.ndarray instead of a masked array
    try:
        wa = winsorize(a1, limits=limits, inclusive=inclusive).data
    except IndexError:
        wa = np.zeros(0, dtype=a.dtype)

    return wa

def winsorized_mean_and_std(a):
    a = _winsorize(a, limits=(0.1, 0.1), inclusive=(True, True))

    if a.size == 0:
        return 0, 0

    mean = np.mean(a)
    sqr_sum = np.sum((a - mean)**2)
    # 1.54 from aoflagger thresholdtools.cpp
    std = (1.54 * sqr_sum / a.size)**0.5

    return mean, std

def winsorized_mode(a):
    a = _winsorize(a, limits=(0.1, 0.1), inclusive=(True, True))

    if a.size == 0:
        return 0

    sqr_sum = np.sum(a**2)

    # 1.0541 from aoflagger thresholdtools.cpp
    return 1.0541 * (sqr_sum / (2*a.size))**0.5
