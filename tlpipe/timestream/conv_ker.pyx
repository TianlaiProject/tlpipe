import numpy as np
cimport numpy as np


FLOAT64 = np.float64
COMPLEX128= np.complex128
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.float64_t FLOAT64_t
ctypedef np.complex128_t COMPLEX128_t

cimport cython
@cython.boundscheck(False) # turn of bounds-checking for entire function
def conv_kernal(np.ndarray[FLOAT64_t, ndim=1] u, np.ndarray[FLOAT64_t, ndim=1] v, FLOAT64_t ui, FLOAT64_t vi, FLOAT64_t sigma, FLOAT64_t l0, FLOAT64_t m0):
    cdef unsigned int nrow = v.shape[0]
    cdef unsigned int ncol = u.shape[0]
    cdef np.ndarray[COMPLEX128_t, ndim=1] rfactor = np.zeros(nrow, dtype=COMPLEX128)
    cdef np.ndarray[COMPLEX128_t, ndim=1] cfactor = np.zeros(ncol, dtype=COMPLEX128)
    cdef np.ndarray[COMPLEX128_t, ndim=2] ker = np.zeros([nrow, ncol], dtype=COMPLEX128)
    cdef FLOAT64_t two_sigma = 2 * sigma
    cdef FLOAT64_t two_pi_sigma = two_sigma * np.pi
    cdef COMPLEX128_t l0_factor = 0.5J * l0 / two_sigma
    cdef COMPLEX128_t m0_factor = 0.5J * m0 / two_sigma

    cdef unsigned int r, c

    for r in range(nrow):
        rfactor[r] = np.exp( -(two_pi_sigma * (v[r] - vi) - m0_factor)**2 )
    for c in range(ncol):
        cfactor[c] = np.exp( -(two_pi_sigma * (u[c] - ui) - l0_factor)**2 )
    for r in range(nrow):
        for c in range(ncol):
            ker[r, c] = rfactor[r] * cfactor[c]

    return ker
