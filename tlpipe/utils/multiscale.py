import numpy as np
from scipy.ndimage import convolve1d
from scipy.ndimage import median_filter
from robust_stats import MAD


# default spline wavelet scaling function
_phi = np.array([1.0/16, 1.0/4, 3.0/8, 1.0/4, 1.0/16])


def up_sampling(a):
    """Up-sampling an array by interleaving it with zero values."""
    shp = a.shape
    shp1 = [ 2*i-1 for i in shp ]
    a1 = np.zeros(shp1, dtype=a.dtype)
    a1[[slice(None, None, 2) for i in shp]] = a

    return a1


def convolve(a, phi):
    """Convolve `a` along each axis sequentially by `phi`."""
    for ax in xrange(a.ndim):
        a = convolve1d(a, phi, axis=ax, mode='reflect')

    return a


def starlet_transform(a, level=None, gen2=False, approx_only=False, phi=_phi):
    """Computes the starlet transform (i.e. the undecimated isotropic wavelet
    transform) of an array.

    The output is a python list containing the sub-bands. If the keyword Gen2
    is set, then it is the 2nd generation starlet transform which is computed:
    i.e., g = Id - h*h instead of g = Id - h.

    """

    if level == None:
        level = int(np.ceil(np.log2(np.min(a.shape))))

    if level <= 0:
        return [ a ]

    phi = phi.astype(a.dtype)
    W = []

    for li in xrange(level):
        if li > 0:
            phi = up_sampling(phi)
        approx = convolve(a, phi)
        if not approx_only:
            if gen2:
                # 2nd generation starlet transfrom applies smoothing twice
                W.append(a - convolve(approx, phi))
            else:
                W.append(a - approx)
        a = approx

    W.append(approx)

    return W


def starlet_smooth(a, level=None, phi=_phi):
    """Return the smooth component of the first generation starlet transform."""
    return starlet_transform(a, level=level, gen2=False, approx_only=True, phi=phi)[0]


def starlet_detrend(a, level=None, phi=_phi):
    """Return the detrended component (i.e., smooth component being subtracted) of the first generation starlet transfrom."""
    return a - starlet_smooth(a, level, phi)


def multiscale_median_transform(a, level=None, scale=2, approx_only=False):
    """Multiscale median transform."""

    if level == None:
        level = int(np.ceil(np.log2(np.min(a.shape))))

    if level <= 0:
        return [ a ]

    W = []

    for li in xrange(level):
        if li > 0:
            scale *= 2
        approx = median_filter(a, 2*scale+1)
        if not approx_only:
            W.append(a - approx)
        a = approx

    W.append(approx)

    return W


def multiscale_median_smooth(a, level=None, scale=2):
    """Return the smooth component of the multiscale median transform."""
    return multiscale_median_transform(a, level=level, scale=scale, approx_only=True)[0]


def multiscale_median_detrend(a, level=None, scale=2):
    """Return the detrended component (i.e., smooth component being subtracted) of the multiscale median transfrom."""
    return a - multiscale_median_smooth(a, level, scale)


def median_wavelet_transform(a, level=None, scale=2, tau=5.0, approx_only=False, phi=_phi):
    """Median-wavelet transfrom."""

    if level == None:
        level = int(np.ceil(np.log2(np.min(a.shape))))

    if level <= 0:
        return [ a ]

    phi = phi.astype(a.dtype)
    W = []

    for li in xrange(level):
        if li > 0:
            scale *= 2
        approx = median_filter(a, 2*scale+1)
        w = a - approx
        th = tau * MAD(w)
        # th = tau * MAD(w[w!=0])
        w[np.abs(w) > th] = 0
        approx += w
        approx = starlet_smooth(approx, li+1, phi)

        if not approx_only:
            W.append(a - approx)
        a = approx

    W.append(approx)

    return W


def median_wavelet_smooth(a, level=None, scale=2, tau=5.0, phi=_phi):
    """Return the smooth component of the median-wavelet transform."""
    return median_wavelet_transform(a, level=level, scale=scale, tau=tau, approx_only=True, phi=phi)[0]


def median_wavelet_detrend(a, level=None, scale=2, tau=5.0, phi=_phi):
    """Return the detrended component (i.e., smooth component being subtracted) of the median-wavelet transfrom."""
    return a - median_wavelet_smooth(a, level, scale, tau, phi)


def multiscale_median_flag(a, level=None, scale=2, tau=5.0, return_mask=True):

    if level == None:
        level = int(np.ceil(np.log2(np.min(a.shape))))

    if return_mask:
        mask = np.zeros_like(a, dtype=bool)

    if level <= 0:
        if return_mask:
            return a, mask
        else:
            return a

    for li in xrange(level):
        if li > 0:
            scale *= 2
        approx = median_filter(a, 2*scale+1)
        w = a - approx
        th = tau * MAD(w)
        # th = tau * MAD(w[w!=0])
        inds = np.where(np.abs(w) > th)[0]
        if return_mask:
            mask[inds] = True
        w[inds] = np.sign(w[inds]) * th
        a = approx + w

    if return_mask:
        return a, mask
    else:
        return a
