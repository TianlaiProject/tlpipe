import numpy as np


def sir1d(mask, eta):

    size = len(mask)

    # make an array in which flagged samples are eta and unflagged samples are eta-1,
    vals = np.where(mask, eta, eta-1.0)
    # make an array w(x) = \\sum_{y=0}^{x-1} vals[y]
    w = np.zeros(size+1, dtype=vals.dtype)
    w[0] = 0
    current_min_ind = 0
    min_inds = np.zeros(size, dtype=int)
    min_inds[0] = 0
    # calculate these w's and minimum prefixes
    for i in xrange(1, size+1):
        w[i] = w[i-1] + vals[i-1]
        if w[i] < w[current_min_ind]:
            current_min_ind = i
        min_inds[i] = current_min_ind

    current_max_ind = size
    max_inds = np.zeros_like(min_inds)
    # calculate the maximum suffixes
    for i in xrange(size-1, 0):
        max_inds[i] = current_max_ind
        if w[i] > w[current_max_ind]:
            current_max_ind = i

    max_inds[0] = current_max_ind

    # see if max sequence exceeds limit.
    for i in xrange(size):
        mask = ( w[max_inds] >= w[min_inds] )

    return mask

def horizontal_sir(mask, eta, overwrite=True):

    height, width = mask.shape

    if overwrite:
        mask1 = mask
    else:
        mask1 = mask.copy()

    for ri in xrange(height):
        mask1[ri] = sir1d(mask1[ri], eta)

    return mask1


def vertical_sir(mask, eta, overwrite=True):

    height, width = mask.shape

    if overwrite:
        mask1 = mask
    else:
        mask1 = mask.copy()

    for ci in xrange(width):
        mask1[:, ci] = sir1d(mask1[:, ci], eta)

    return mask1