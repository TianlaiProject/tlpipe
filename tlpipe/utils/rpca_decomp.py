import numpy as np
from scipy import linalg as la


def mad(a):
    """Median absolute deviation. Works for both real and complex array."""

    def madr(x):
        """Median absolute deviation of a real array."""
        return np.median(np.abs(x - np.median(x)))

    if np.isrealobj(a):
        return madr(a)
    else:
        return np.sqrt(madr(a.real)**2 + madr(a.imag)**2)

def MAD(a):
    """Median absolute deviation divides 0.6745."""
    return mad(a) / 0.6745



def l0_norm(a):
    """Return the :math:`l_0`-norm (i.e., number of non-zero elements) of an array."""
    return len(np.where(a.flatten() != 0.0)[0])


def l1_norm(a):
    """Return the :math:`l_1`-norm of an array."""
    return np.sum(np.abs(a))


def truncate(a, lmbda):
    """Hard thresholding operator, which works for both real and complex array."""
    return a * (np.abs(a) > lmbda)


def sign(a):
    """Sign of an array, which works for both real and complex array."""
    if np.isrealobj(a):
        return np.sign(a)
    else:
        return np.exp(1.0J * np.angle(a))


def shrink(a, lmbda):
    """Soft thresholding operator, which works for both real and complex array."""
    return sign(a) * np.maximum(np.abs(a) - lmbda, 0.0) # work for both real and complex


def decompose(M, rank=1, S=None, lmbda=None, threshold='hard', max_iter=100, tol=1.0e-8, check_hermitian=False, debug=False):
    """Stable principal component decomposition of an Hermitian matrix."""

    if check_hermitian:
        if not np.allclose(M, M.T.conj()):
            raise ValueError('M must be a Hermitian matrix')

    # if M is zero matrix, L and S should be zero matrix too
    if np.allclose(M, 0.0):
        return np.zeros_like(M), np.zeros_like(M)

    if lmbda is None:
        fixed_lmbda = False
    else:
        fixed_lmbda = True

    if threshold == 'hard':
        hard  = True
    elif threshold == 'soft':
        hard = False
    else:
        raise ValueError('Unknown thresholding method: %s' % threshold)

    if (S is None) or (S.shape != M.shape):
        # initialize S as zero
        S = np.zeros_like(M)
    else:
        S = S.astype(M.dtype)

    S_old = S
    L_old = np.zeros_like(M)

    d = M.shape[0]
    MF= la.norm(M, ord='fro')

    for it in xrange(max_iter):
        # compute only the largest rank eigen values and vectors, which is faster
        s, U = la.eigh(M - S, eigvals=(d-rank, d-1))
        # threshold s to make L Hermitian positive semidefinite
        # L = np.dot(U[:, -rank:]*np.maximum(s[-rank:], 0), U[:, -rank:].T.conj())
        L = np.dot(U*np.maximum(s, 0), U.T.conj())

        res = M - L

        s1 = la.eigh(res, eigvals_only=True, eigvals=(d-1, d-1))
        # L may be under noise
        if s[-1] < 0.2 * s1[-1]:
            res = M

        if not fixed_lmbda:
            # the universal threshold: sigma * (2 * log(d*d))**0.5
            th = (2.0 * np.log10(d * d))**0.5 * MAD(res)
            if hard: # hard-thresholding
                lmbda = 2**0.5 * th
            else: # soft-thresholding
                lmbda = th

            if debug:
                print 'lmbda:', lmbda

        # compute new S
        if hard:
            S = truncate(res, lmbda)
        else:
            S = shrink(res, lmbda)

        tol1 = (la.norm(L - L_old, ord='fro') + la.norm(S - S_old, ord='fro')) / MF
        if tol1 < tol:
            if debug:
                print 'Converge when iteration: %d with tol: %g < %g' % (it, tol1, tol)
            break

        L_old = L
        S_old = S

    else:
        print 'Exit with max_iter: %d, tol: %g >= %g' % (it, tol1, tol)

    return L, S