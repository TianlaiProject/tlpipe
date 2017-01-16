import numpy as np


def sir1d(mask, eta):

    if eta <= 0 or mask.all():
        return mask

    if eta >= 1:
        return np.ones_like(mask)

    size = len(mask)

    # make an array in which flagged samples are eta and unflagged samples are eta-1,
    vals = np.where(mask, eta, eta-1.0)
    # make an array M(x) = \\sum_{y=0}^{x-1} vals[y]
    M = np.zeros(size+1, dtype=vals.dtype)
    M[1:] = np.cumsum(vals)

    # check mask condition
    for i in xrange(0, size):
        if np.max(M[i+1:]) >= np.min(M[:i+1]):
            mask[i] = True
        else:
            mask[i] = False

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
