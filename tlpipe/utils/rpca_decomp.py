import numpy as np
from scipy import linalg as la


def MAD(a):
    return np.median(np.abs(a - np.median(a))) / 0.6745


def l0_norm(a):
    return len(np.where(a.flatten() != 0.0)[0])


def l1_norm(a):
    return np.sum(np.abs(a))


def truncate(a, lmbda):
    return a * (np.abs(a) > lmbda)


def sign(a):
    if np.isrealobj(a):
        return np.sign(a)
    else:
        return np.exp(1.0J * np.angle(a))


def shrink(a, lmbda):
    return sign(a) * np.maximum(np.abs(a) - lmbda, 0.0) # work for both real and complex


def decompose(V, rank=1, lmbda=None, threshold='hard', max_iter=100, tol=1.0e-8, check_hermitian=False, debug=False):

    if check_hermitian:
        if not np.allclose(V, V.T.conj()):
            raise ValueError('V must be a Hermitian matrix')

    d = V.shape[0]
    VF= la.norm(V, ord='fro')

    if lmbda is None:
        # lmbda = MAD(V)
        lmbda = max(1.0, 2 * np.log(d)**0.5) * min(MAD(V), np.std(V))
        fixed_lmbda = False
        if debug:
            print 'lmbda:', lmbda
    else:
        fixed_lmbda = True

    N_old = np.zeros_like(V)
    # initialize S to be the outliers
    if threshold == 'hard':
        S = truncate(V, lmbda)
    elif threshold == 'soft':
        S = shrink(V, lmbda)
    else:
        raise ValueError('Unknown thresholding method: %s' % threshold)

    for it in xrange(max_iter):
        # compute only the largest rank eigen values and vectors, which is faster
        s, U = la.eigh(V - S, eigvals=(d-rank, d-1))
        # threshold s to make V0 Hermitian positive semidefinite
        # V0 = np.dot(U[:, -rank:]*np.maximum(s[-rank:], 0), U[:, -rank:].T.conj())
        V0 = np.dot(U*np.maximum(s, 0), U.T.conj())

        res = V - V0
        N = res - S
        tol1 = la.norm(N_old - N, ord='fro') / VF
        if tol1 < tol:
            if debug:
                print 'Converge when iteration: %d with tol: %g < %g' % (it, tol1, tol)
            break

        if not fixed_lmbda and it >= 1:
            # lmbda = 5.0 * np.std(N)
            # use the universal threshold: sigma * (2 * log(d*d))**0.5
            lmbda = max(1.0, 2 * np.log(d)**0.5) * min(MAD(N), np.std(N))
            if debug:
                print 'lmbda:', lmbda

        # compute new S
        if threshold == 'hard':
            S = truncate(res, lmbda)
        elif threshold == 'soft':
            S = shrink(res, lmbda)

        N_old = N

    else:
        print 'Exit with max_iter: %d, tol: %g >= %g' % (it, tol1, tol)

    return V0, S
