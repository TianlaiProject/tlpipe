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

    if lmbda is None:
        fixed_lmbda = False
    else:
        fixed_lmbda = True

    d = V.shape[0]
    S = np.zeros_like(V)
    opt_old = np.Inf

    min_opt = np.Inf
    min_tol = np.Inf

    for it in xrange(max_iter):
        # compute only the largest rank eigen values and vectors, which is faster
        s, U = la.eigh(V - S, eigvals=(d-rank, d-1))
        # threshold s to make V0 Hermitian positive semidefinite
        # V0 = np.dot(U[:, -rank:]*np.maximum(s[-rank:], 0), U[:, -rank:].T.conj())
        V0 = np.dot(U*np.maximum(s, 0), U.T.conj())
        res = V - V0
        N = res - S

        if not fixed_lmbda:
            # use the universal threshold: sigma * (2 * log(d*d))**0.5
            lmbda = max(1.0, 2 * np.log(d)**0.5) * min(MAD(N), np.std(N))
            if debug:
                print 'lmbda:', lmbda

        if threshold == 'hard':
            opt = 0.5 * la.norm(N, ord='fro')**2 + lmbda * l0_norm(S)
        elif threshold == 'soft':
            opt = 0.5 * la.norm(N, ord='fro')**2 + lmbda * l1_norm(S)
        else:
            raise ValueError('Unknown thresholding method: %s' % threshold)

        tol1 = opt_old - opt
        # if tol1 >= 0.0 and tol1 < tol:
        if abs(tol1) < tol:
            if debug:
                print 'Converge when iteration: %d, opt: %g, with tol: %g < %g' % (it, opt, tol1, tol)
            break

        # record the minimum opt and its corresponding tol, V0, S and lmbda
        if opt < min_opt:
            min_opt = opt
            min_tol = tol1 # tol corresponding to this opt
            min_V0 = V0
            min_S = S

        opt_old = opt

        if threshold == 'hard':
            S = truncate(res, lmbda)
        elif threshold == 'soft':
            S = shrink(res, lmbda)

    else:
        print 'Exit with max_iter: %d, opt: %g, tol: %g >= %g' % (it, min_opt, min_tol, tol)
        return min_V0, min_S

    return V0, S
