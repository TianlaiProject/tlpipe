"""This implements the scale-invariant rank (SIR) operator.

The operator considers a sample to be contaminated with RFI when the sample
is in a subsequence of mostly flagged samples. To be more precise, it will
flag a subsequence when more than :math:`(1 - \\eta) N` of its samples are
flagged, with :math:`N` the number of samples in the subsequence and
:math:`\\eta` a constant, :math:`0 \\le \\eta \\le 1`. Using :math:`\\rho`
to denote the operator, the output :math:`\\rho(X)` can be formally defined
as

.. math:: \\rho(X) \\equiv \\bigcup \\left\\{ [Y1, Y2) \\ \\mid \\ \#(X \\cap [Y1, Y2)) \ge (1 - \\eta)(Y2 - Y1) \\right\\},

with :math:`[Y1, Y2)` a half-open interval of a one-dimensional set, and the
hash symbol :math:`\#` denoting the count-operator that returns the number of
elements in the set. In words, the equation defines :math:`\\rho(X)` to consist
of all the samples that are in an interval :math:`[Y1, Y2)`, in which the ratio
of samples in the input :math:`X` is greater or equal than :math:`(1 - \\eta)`.
Parameter :math:`\\eta` represents the aggressiveness of the method: with
:math:`\\eta = 0`, no additional samples are flagged and :math:`\\rho(X) = X`.
On the other hand, :math:`\\eta = 1` implies all samples will be flagged.

For more details, see Offringa et al., 2012, A&A, 539, A95, *A morphological
algorithm for improving radio-frequency interference detection*.

"""

import numpy as np


def sir1d(mask, eta):

    MASKRFI = 2
    if eta <= 0 or mask.all():
        return mask

    if eta >= 1:
        mask |= MASKRFI
        return mask

    size = len(mask)

    # make an array in which flagged samples are eta and unflagged samples are eta-1,
    vals = np.where(mask, eta, eta-1.0)
    # make an array M(x) = \\sum_{y=0}^{x-1} vals[y]
    M = np.zeros(size+1, dtype=vals.dtype)
    M[1:] = np.cumsum(vals)

    # check mask condition
    for i in xrange(0, size):
        if np.max(M[i+1:]) >= np.min(M[:i+1]):
            mask[i] |= MASKRFI
        else:
            mask[i] = np.bitwise_and(mask[i],~MASKRFI)

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


if __name__ == '__main__':
    # test sir 1d
    # mask = np.array([False]*8)
    # mask = np.array([False]*3 + [True] + [False]*7)
    # mask = np.array([True] + [False]*3 + [True] + [False]*7 + [True])
    # mask = np.array([True] + [False]*3 + [True, True] + [False]*7)
    mask = np.array([0, 0, 1, 0, 1, 0, 1, 1, 0, 0], dtype=bool)
    eta = 0.2
    # eta = 0.5
    print mask.astype(int)
    print sir1d(mask, eta).astype(int)

    # test sir 2d
    ra = np.random.rand(8, 10) - 0.5
    mask = np.where(ra>0, True, False)
    print mask.astype(int)
    print horizontal_sir(mask, eta, False).astype(int)
    print vertical_sir(mask, eta, False).astype(int)
