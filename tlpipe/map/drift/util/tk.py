import numpy as np
from scipy import linalg as la
from scipy.optimize import root


def tk(A, b, th=1.0e-8, x0=1.0, noise=False):
    """Get the inverse of b = Ax + e by Tikhonov regularization method.

    This implements the method presented in Hochstenbach, et.al., Regularization
    parameter determination for discrete ill-posed problem.

    Parameters
    ----------
    A : np.ndarray, (m, n)
        Matrix.
    b : np.ndarray, (m,)
        Right hand side vector.
    th : float, optional
        A threshold that determining the rank of `A`.
    x0 : float, optional
        A initial value for the inner root method.
    noise : boolean, optional
        Wether to also return the estimated noise level. Default False.

    Returns
    -------
    x : np.ndarray, (n,)
        The solution vector.
    noise_level : float
        The estimated noise level. Return this only when `noise` is True.

    """

    U, s, Vh = la.svd(A, full_matrices=False)
    Uh, V = U.T.conj(), Vh.T.conj()
    Uhb = np.dot(Uh, b)
    if (s < th).all(): # rank of A is 0
        if noise:
            return 0, 1.0
        else:
            return 0
    r = np.where(s >= th)[0][-1] + 1 # rank of A
    dt = np.inf # delta
    xk = np.zeros_like(A[0])

    def xm(mu):
        return np.dot(V*(s/(s**2 + mu**2)), Uhb)

    for k in xrange(1, r+1):
        xk = xk + (1.0 / s[k-1]) * Uhb[k-1] * V[:, k-1]
        assert np.allclose(xk, np.dot(np.dot(V[:, :k]*(1.0/s[:k]), Uh[:k, :]), b))
        pk = la.norm(b - np.dot(A, xk))

        def f(mu):
            return pk - la.norm(b - np.dot(A, xm(mu)))

        sol = root(f, x0)
        # print sol.x

        xmk1 = xm(sol.x)
        dt1 = la.norm(xk - xmk1)
        if dt1 > dt or k == r:
            if r == 1:
                xmk, pk1 = xmk1, pk
            if noise:
                return xmk, pk1 / la.norm(b)
            else:
                return xmk
        else:
            xmk = xmk1
            dt = dt1
            if noise:
                pk1 = pk


if __name__ == '__main__':
    m, n = 80, 100
    r = 50 # rank
    th = 1.0
    A = np.random.randn(m, n) + 1.0J * np.random.randn(m, n)
    U, s, Vh = la.svd(A, full_matrices=False)
    A = np.dot(U[:, :r]*s[:r], Vh[:r, :])
    x = th * (np.random.randn(n) + 1.0J * np.random.randn(n))

    # gauss noise
    e = np.random.randn(m) + 1.0J * np.random.randn(m)

    b0 = np.dot(A, x)
    b = b0 + e

    xm = tk(A, b)
    print la.norm(xm - x), la.norm(x)
