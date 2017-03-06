import numpy as np
from scipy import linalg as la


def MAD(a):
    return np.median(np.abs(a - np.median(a))) / 0.6745


def truncate(a, lmbda):
    return a * (np.abs(a) > lmbda)


def shrink(a, lmbda):
    # return np.sign(a) * np.maximum(np.abs(a) - lmbda, 0.0) # works only for real a
    return a * np.maximum(1.0 - lmbda / np.abs(a), 0.0)


def decompose(V, rank=1, lmbda=None, threshold='hard', max_iter=100, tol=1.0e-8, check_hermitian=False, debug=False):

    if check_hermitian:
        if not np.allclose(V, V.T.conj()):
            raise ValueError('V must be a Hermitian matrix')

    if lmbda is None:
        lmbda = MAD(V)
        fixed_lmbda = False
        if debug:
            print 'lmbda:', lmbda
    else:
        fixed_lmbda = True

    d = V.shape[0]
    S = np.zeros_like(V)
    NF_old = np.Inf
    lmbdas = [ lmbda ]

    for it in xrange(max_iter):
        # compute only the largest rank eigen values and vectors, which is faster
        s, U = la.eigh(V - S, eigvals=(d-rank, d-1))
        # threshold s to make V0 Hermitian positive semidefinite
        # V0 = np.dot(U[:, -rank:]*np.maximum(s[-rank:], 0), U[:, -rank:].T.conj())
        V0 = np.dot(U*np.maximum(s, 0), U.T.conj())
        res = V - V0
        N = res - S
        NF = la.norm(N, ord='fro')
        tol1 = np.abs(NF_old - NF) / NF
        if tol1 < tol:
            if debug:
                print 'Converge when iteration: %d with tol: %g < %g' % (it, tol1, tol)
            break
        NF_old = NF
        if not fixed_lmbda and it >= 1:
            # lmbda = 5.0 * np.std(N)
            # use a threshold of sigma * (2 * log(d*d))**0.5
            lmbda = 2 * np.log(d)**0.5 * np.std(N)

            # avoid cycling between two lambdas
            if len(lmbdas) == 2:
                if lmbdas[0] == lmbda:
                    # lmbda = 0.5 * (lmbdas[1] + lmbda)
                    lmbda = (lmbdas[1] + lmbda)
                lmbdas.pop(0)
            if debug:
                print 'lmbda:', lmbda
            lmbdas.append(lmbda)

        if threshold == 'hard':
            S = truncate(res, lmbda)
        elif threshold == 'soft':
            S = shrink(res, lmbda)
        else:
            raise ValueError('Unknown thresholding method: %s' % threshold)
    else:
        print 'Exit with max_iter: %d, tol: %g >= %g' % (it, tol1, tol)

    return V0, S
