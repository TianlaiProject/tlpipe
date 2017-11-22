import numpy as np
cimport numpy as np
from cython cimport boundscheck, wraparound


@boundscheck(False)
@wraparound(False)
def threshold_len1(float[:, :] vis, np.ndarray[np.uint8_t, cast=True, ndim=2] vis_mask, int height, int width, float threshold):
    cdef int x, y

    for y in range(height):
        for x in range(width):
            if vis_mask[y, x] == False and abs(vis[y, x]) > threshold:
                vis_mask[y, x] = True


@boundscheck(False)
@wraparound(False)
def hthreshold(float[:, :] vis, np.ndarray[np.uint8_t, cast=True, ndim=2] vis_mask, int height, int width, int length, float threshold):
    cdef int x, y
    cdef int cnt, left, right
    cdef float sm
    cdef np.ndarray[np.uint8_t, cast=True, ndim=2] tmp_mask = vis_mask.copy()

    for y in range(height):

        # if all have been masked, continue
        for x in range(width):
            if vis_mask[y, x] == False:
                break
        else:
            continue

        sm = 0.0
        cnt = 0
        left = 0
        right = 0
        for right in range(length-1):
            if vis_mask[y, right] == False:
                sm += vis[y, right]
                cnt += 1

        while(right < width):
            # add the sample at the right
            if vis_mask[y, right] == False:
                sm += vis[y, right]
                cnt += 1
            # check
            if (cnt > 0) and (abs(sm / cnt) > threshold):
                tmp_mask[y, left:left+length] = True
            # subtract the sample at the left
            if vis_mask[y, left] == False:
                sm -= vis[y, left]
                cnt -= 1

            left += 1
            right += 1

    # set to the new mask
    vis_mask[:] = tmp_mask


@boundscheck(False)
@wraparound(False)
def vthreshold(float[:, :] vis, np.ndarray[np.uint8_t, cast=True, ndim=2] vis_mask, int height, int width, int length, float threshold):
    cdef int x, y
    cdef int cnt, top, bottom
    cdef float sm
    cdef np.ndarray[np.uint8_t, cast=True, ndim=2] tmp_mask = vis_mask.copy()

    for x in range(width):

        # if all have been masked, continue
        for y in range(height):
            if vis_mask[y, x] == False:
                break
        else:
            continue

        sm = 0.0
        cnt = 0
        top = 0
        bottom = 0
        for bottom in range(length-1):
            if vis_mask[bottom, x] == False:
                sm += vis[bottom, x]
                cnt += 1

        while(bottom < height):
            # add the sample at the bottom
            if vis_mask[bottom, x] == False:
                sm += vis[bottom, x]
                cnt += 1
            # check
            if (cnt > 0) and (abs(sm / cnt) > threshold):
                tmp_mask[top:top+length, x] = True
            # subtract the sample at the top
            if vis_mask[top, x] == False:
                sm -= vis[top, x]
                cnt -= 1

            top += 1
            bottom += 1

    vis_mask[:] = tmp_mask