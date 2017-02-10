"""This implements the mathematical morphological dilate operation."""

import numpy as np


def dilate1d(mask, size):

    if size == 0:
        return mask

    mask1 = mask.copy()
    size = min(size, mask.size)
    int_size = int(size)

    dist = int_size + 1
    for x in xrange(size):
        if mask[x]:
            dist = -int_size
        dist += 1

    for x in xrange(mask.size - size):
        if mask[x+size]:
            dist = -int_size
        if dist <= int_size:
            mask1[x] = True
            dist += 1
        else:
            mask1[x] = False

    for x in xrange(mask.size-size, mask.size):
        if dist <= int_size:
            mask1[x] = True
            dist += 1
        else:
            mask1[x] = False

    return mask1

def horizontal_dilate(mask, size, overwrite=True):

    height, width = mask.shape

    if overwrite:
        mask1 = mask
    else:
        mask1 = mask.copy()

    for ri in xrange(height):
        mask1[ri] = dilate1d(mask1[ri], size)

    return mask1


def vertical_dilate(mask, eta, overwrite=True):

    height, width = mask.shape

    if overwrite:
        mask1 = mask
    else:
        mask1 = mask.copy()

    for ci in xrange(width):
        mask1[:, ci] = dilate1d(mask1[:, ci], size)

    return mask1


if __name__ == '__main__':
    # test dilate 1d
    # mask = np.array([False]*8)
    # mask = np.array([False]*3 + [True] + [False]*7)
    mask = np.array([False]*3 + [True] + [False]*7 + [True])
    size = 2
    print mask.astype(int)
    print dilate1d(mask, size).astype(int)

    # test dilate 2d
    ra = np.random.rand(8, 10) - 0.5
    mask = np.where(ra>0, True, False)
    print mask.astype(int)
    print horizontal_dilate(mask, size, False).astype(int)
    print vertical_dilate(mask, size, False).astype(int)
