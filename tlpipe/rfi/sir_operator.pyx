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
cimport numpy as np
from cython cimport boundscheck, wraparound
from cpython.mem cimport PyMem_Malloc, PyMem_Free


# @boundscheck(False)
# @wraparound(False)
# cpdef sir1d(np.ndarray[np.uint8_t, cast=True, ndim=1] mask, double eta):
#     ### these are all numpy level operations, could not get the best preformance

#     cdef int size = mask.shape[0]

#     if eta <= 0.0 or mask.all():
#         return mask

#     if eta >= 1.0:
#         return np.ones_like(mask)

#     # make an array in which flagged samples are eta and unflagged samples are eta-1,
#     vals = np.where(mask, eta, eta-1.0)

#     # make an array M(x) = \\sum_{y=0}^{x-1} vals[y]
#     M = np.zeros(size+1, dtype=vals.dtype)
#     M[1:] = np.cumsum(vals)

#     # check mask condition
#     cdef int i
#     for i in range(0, size):
#         if np.max(M[i+1:]) >= np.min(M[:i+1]):
#             mask[i] = True
#         else:
#             mask[i] = False

#     return mask


@boundscheck(False)
@wraparound(False)
cpdef sir1d(np.ndarray[np.uint8_t, cast=True, ndim=1] mask, double eta):
    # use more C level operations to get higher performance

    cdef int size = mask.shape[0]

    if eta <= 0.0 or mask.all():
        return mask

    if eta >= 1.0:
        return np.ones_like(mask)

    cdef double *vals = <double *> PyMem_Malloc(size * sizeof(double))
    # if not vals:
    #     raise MemoryError()
    cdef double *M = <double *> PyMem_Malloc((size + 1) * sizeof(double))
    # if not M:
    #     raise MemoryError()
    cdef double *M1 = <double *> PyMem_Malloc(size * sizeof(double)) # to save min val of M[:-1]
    # if not M1:
    #     raise MemoryError()
    cdef double *M2 = <double *> PyMem_Malloc(size * sizeof(double)) # to save max val of M[1:]
    # if not M1:
    #     raise MemoryError()

    cdef int vi
    for vi in range(size):
        if mask[vi] == True:
           vals[vi] = eta
        else:
           vals[vi] = eta - 1.0

    cdef int mi
    cdef double cumsum = 0.0
    M[0] = 0.0
    for mi in range(1, size+1):
        cumsum += vals[mi-1]
        M[mi] = cumsum

    M1[0] = M[0]
    # save min val of M[:-1] to M1
    for mi in range(1, size):
        if M[mi] < M1[mi-1]:
            M1[mi] = M[mi]
        else:
            M1[mi] = M1[mi-1]

    M2[size-1] = M[size]
    # save max val of M[1:] to M2
    for mi in range(size-2, -1, -1):
        if M[mi+1] > M2[mi+1]:
            M2[mi] = M[mi+1]
        else:
            M2[mi] = M2[mi+1]

    # check mask condition
    cdef int i
    for i in range(size):
        if M1[i] <= M2[i]:
            mask[i] = True
        else:
            mask[i] = False

    PyMem_Free(vals)
    PyMem_Free(M)
    PyMem_Free(M1)
    PyMem_Free(M2)

    return mask


@boundscheck(False)
@wraparound(False)
def horizontal_sir(np.ndarray[np.uint8_t, cast=True, ndim=2] mask, double eta, bint overwrite=True):

    cdef np.ndarray[np.uint8_t, cast=True, ndim=2] mask1
    cdef int height = mask.shape[0]

    if overwrite:
        mask1 = mask
    else:
        mask1 = mask.copy()

    cdef int ri
    for ri in range(height):
        mask1[ri] = sir1d(mask1[ri], eta)

    return mask1


@boundscheck(False)
@wraparound(False)
def vertical_sir(np.ndarray[np.uint8_t, cast=True, ndim=2] mask, double eta, bint overwrite=True):

    cdef np.ndarray[np.uint8_t, cast=True, ndim=2] mask1
    cdef int width = mask.shape[1]

    if overwrite:
        mask1 = mask
    else:
        mask1 = mask.copy()

    cdef int ci
    for ci in range(width):
        mask1[:, ci] = sir1d(mask1[:, ci], eta)

    return mask1
