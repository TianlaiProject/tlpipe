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

    d = V.shape[0]
    if lmbda is None:
        lmbda = max(1.0, 2 * np.log(d)**0.5) * MAD(V)
        fixed_lmbda = False
        if debug:
            print 'lmbda:', lmbda
    else:
        fixed_lmbda = True

    S = np.zeros_like(V)
    NF_old = np.Inf

    min_tol = np.Inf
    if not fixed_lmbda:
        min_lmbda = lmbda
        lmbdas = [ lmbda ]
        tols = [ np.Inf ]

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

        # record the minimum tol and its corresponding V0, S and lmbda
        if tol1 < min_tol:
            min_tol = tol1
            min_V0 = V0
            min_S = S
            if not fixed_lmbda:
                min_lmbda = lmbda

        NF_old = NF
        if not fixed_lmbda and it >= 1:
            # lmbda = 5.0 * np.std(N)
            # use the universal threshold: sigma * (2 * log(d*d))**0.5
            lmbda = max(1.0, 2 * np.log(d)**0.5) * np.std(N)

            # avoid cycling around
            while (lmbda in lmbdas) and (tol1 in tols):
                while True:
                    factor = np.random.gamma(1.0)
                    if factor != 1.0 and (factor >=0.9 or factor <= 10.0):
                        break
                lmbda = factor * min_lmbda

            if debug:
                print 'lmbda:', lmbda
            lmbdas.append(lmbda)
            tols.append(tol1)

        if threshold == 'hard':
            S = truncate(res, lmbda)
        elif threshold == 'soft':
            S = shrink(res, lmbda)
        else:
            raise ValueError('Unknown thresholding method: %s' % threshold)
    else:
        print 'Exit with max_iter: %d, tol: %g >= %g' % (it, min_tol, tol)
        return min_V0, min_S

    return V0, S
