import numpy as np

def window_nd(shape, name="blackman"):
    """define a window function in n-dim
    `name` options: blackman, bartlett, hamming, hanning, kaiser
    """
    print "using a %s window" % name
    ndim = len(shape)
    window = np.ones(shape)

    for index, l in enumerate(shape):
        sl = ([np.newaxis]*index) + [slice(None)] + \
             ([np.newaxis]) * (ndim - index - 1)
        window_func = getattr(np, name)
        arr = window_func(l)
        window *= arr[sl]

    return window


def rfftfreqn(n, d = None):
    """Generate an array of FFT frequencies for an n-dimensional *real* transform.

    Parameters
    ----------
    n : tuple
        A tuple containing the dimensions of the window in each
        dimension.
    d : float or tuple, optional
        The sample spacing along each dimension

    Returns
    -------
    frequencies : ndarray
        An array of shape (n[0], n[1], ..., n[-1]/2+1, len(n)),
        containing the frequency vector of each sample.
    """

    n = np.array(n)
    if d is None:
        d = n
    else:
        if(len(d) != len(n)):
            raise Exception("Sample spacing array is the wrong length.")
        d *= n

    # Use slice objects and mgrid to produce an array with
    # zero-centered frequency vectors, except the last dimension (as
    # we are doing a real transform).
    mgslices = map(lambda x: slice(-x/2, x/2, 1.0), n[:-1]) + [slice(0, n[-1]/2+1, 1.0)]
    mgarr = np.mgrid[mgslices]

    # Use fftshift to move the zero elements to the start of each
    # dimension (except the last dimension).
    shiftaxes = range(1,len(n))
    shiftarr = np.fft.fftshift(mgarr, axes=shiftaxes)

    # Roll the first axis to the back so the components of the
    # frequency are the final axis.
    rollarr = np.rollaxis(shiftarr, 0, len(n)+1)

    # If we are given the sample spacing, scale each vector component
    # such that it is a proper frequency.
    rollarr[...,:] /= d

    return rollarr

