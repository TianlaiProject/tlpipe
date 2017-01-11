import numpy as np
from scipy.stats.mstats import winsorize


def _winsorize(a, limits=None, inclusive=(True, True)):
    # drop masked data
    a1 = np.ma.compressed(a)

    # use .data to return an np.ndarray instead of a masked array
    return winsorize(a1, limits=limits, inclusive=inclusive).data

def winsorized_mean_and_std(a):
    a = _winsorize(a, limits=(0.1, 0.1), inclusive=(True, True))

    if a.size == 0:
        return 0, 0

    mean = np.mean(a)
    sqr = (a - mean)**2
    # 1.54 from aoflagger thresholdtools.cpp
    std = (1.54 * sqr / a.size)**0.5

    return mean, std

def winsorized_mode(a):
    a = _winsorize(a, limits=(0.1, 0.1), inclusive=(True, True))

    if a.size == 0:
        return 0

    sqr = a**2

    # 1.0541 from aoflagger thresholdtools.cpp
    return 1.0541 * (sqr / (2*a.size))
