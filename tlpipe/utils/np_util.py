import numpy as np


def unique(ar, return_index=False, return_inverse=False, return_counts=False):
    """Find the unique elements of an array.

    Returns the sorted unique elements of an array. There are two optional
    outputs in addition to the unique elements: the indices of the input array
    that give the unique values, and the indices of the unique array that
    reconstruct the input array.

    Copied from newer version of numpy, as old version has no `return_counts`
    argument.

    Parameters
    ----------
    ar : array_like
        Input array. This will be flattened if it is not already 1-D.
    return_index : bool, optional
        If True, also return the indices of `ar` that result in the unique
        array.
    return_inverse : bool, optional
        If True, also return the indices of the unique array that can be used
        to reconstruct `ar`.
    return_counts : bool, optional
        .. versionadded:: 1.9.0
        If True, also return the number of times each unique value comes up
        in `ar`.

    Returns
    -------
    unique : ndarray
        The sorted unique values.
    unique_indices : ndarray, optional
        The indices of the first occurrences of the unique values in the
        (flattened) original array. Only provided if `return_index` is True.
    unique_inverse : ndarray, optional
        The indices to reconstruct the (flattened) original array from the
        unique array. Only provided if `return_inverse` is True.
    unique_counts : ndarray, optional
        .. versionadded:: 1.9.0
        The number of times each of the unique values comes up in the
        original array. Only provided if `return_counts` is True.

    See Also
    --------
    numpy.lib.arraysetops : Module with a number of other functions for
                            performing set operations on arrays.

    Examples
    --------
    >>> np.unique([1, 1, 2, 2, 3, 3])
    array([1, 2, 3])
    >>> a = np.array([[1, 1], [2, 3]])
    >>> np.unique(a)
    array([1, 2, 3])

    Return the indices of the original array that give the unique values:

    >>> a = np.array(['a', 'b', 'b', 'c', 'a'])
    >>> u, indices = np.unique(a, return_index=True)
    >>> u
    array(['a', 'b', 'c'],
           dtype='|S1')
    >>> indices
    array([0, 1, 3])
    >>> a[indices]
    array(['a', 'b', 'c'],
           dtype='|S1')

    Reconstruct the input array from the unique values:

    >>> a = np.array([1, 2, 6, 4, 2, 3, 2])
    >>> u, indices = np.unique(a, return_inverse=True)
    >>> u
    array([1, 2, 3, 4, 6])
    >>> indices
    array([0, 1, 4, 3, 1, 2, 1])
    >>> u[indices]
    array([1, 2, 6, 4, 2, 3, 2])

    """
    ar = np.asanyarray(ar).flatten()

    optional_indices = return_index or return_inverse
    optional_returns = optional_indices or return_counts

    if ar.size == 0:
        if not optional_returns:
            ret = ar
        else:
            ret = (ar,)
            if return_index:
                ret += (np.empty(0, np.bool),)
            if return_inverse:
                ret += (np.empty(0, np.bool),)
            if return_counts:
                ret += (np.empty(0, np.intp),)
        return ret

    if optional_indices:
        perm = ar.argsort(kind='mergesort' if return_index else 'quicksort')
        aux = ar[perm]
    else:
        ar.sort()
        aux = ar
    flag = np.concatenate(([True], aux[1:] != aux[:-1]))

    if not optional_returns:
        ret = aux[flag]
    else:
        ret = (aux[flag],)
        if return_index:
            ret += (perm[flag],)
        if return_inverse:
            iflag = np.cumsum(flag) - 1
            iperm = perm.argsort()
            ret += (np.take(iflag, iperm),)
        if return_counts:
            idx = np.concatenate(np.nonzero(flag) + ([ar.size],))
            ret += (np.diff(idx),)
    return ret


def average(a, axis=None, weights=None, returned=False):
    """
    Return the weighted average of array over the given axis.

    Copied from newer version of numpy, as old version raise "ComplexWarning:
    Casting complex values to real discards the imaginary part".

    Parameters
    ----------
    a : array_like
        Data to be averaged.
        Masked entries are not taken into account in the computation.
    axis : int, optional
        Axis along which the average is computed. The default is to compute
        the average of the flattened array.
    weights : array_like, optional
        The importance that each element has in the computation of the average.
        The weights array can either be 1-D (in which case its length must be
        the size of `a` along the given axis) or of the same shape as `a`.
        If ``weights=None``, then all data in `a` are assumed to have a
        weight equal to one.   If `weights` is complex, the imaginary parts
        are ignored.
    returned : bool, optional
        Flag indicating whether a tuple ``(result, sum of weights)``
        should be returned as output (True), or just the result (False).
        Default is False.

    Returns
    -------
    average, [sum_of_weights] : (tuple of) scalar or MaskedArray
        The average along the specified axis. When returned is `True`,
        return a tuple with the average as the first element and the sum
        of the weights as the second element. The return type is `np.float64`
        if `a` is of integer type and floats smaller than `float64`, or the
        input data-type, otherwise. If returned, `sum_of_weights` is always
        `float64`.

    Examples
    --------
    >>> a = np.ma.array([1., 2., 3., 4.], mask=[False, False, True, True])
    >>> np.ma.average(a, weights=[3, 1, 0, 0])
    1.25

    >>> x = np.ma.arange(6.).reshape(3, 2)
    >>> print x
    [[ 0.  1.]
     [ 2.  3.]
     [ 4.  5.]]
    >>> avg, sumweights = np.ma.average(x, axis=0, weights=[1, 2, 3],
    ...                                 returned=True)
    >>> print avg
    [2.66666666667 3.66666666667]

    """
    a = np.ma.asarray(a)
    mask = a.mask
    ash = a.shape
    if ash == ():
        ash = (1,)
    if axis is None:
        if mask is np.ma.nomask:
            if weights is None:
                n = a.sum(axis=None)
                d = float(a.size)
            else:
                w = np.ma.filled(weights, 0.0).ravel()
                n = umath.add.reduce(a._data.ravel() * w)
                d = umath.add.reduce(w)
                del w
        else:
            if weights is None:
                n = a.filled(0).sum(axis=None)
                d = float(umath.add.reduce((~mask).ravel()))
            else:
                w = np.ma.array(np.ma.filled(weights, 0.0), float, mask=mask).ravel()
                n = np.ma.add.reduce(a.ravel() * w)
                d = np.ma.add.reduce(w)
                del w
    else:
        if mask is np.ma.nomask:
            if weights is None:
                d = ash[axis] * 1.0
                n = np.ma.add.reduce(a._data, axis)
            else:
                w = np.ma.filled(weights, 0.0)
                wsh = w.shape
                if wsh == ():
                    wsh = (1,)
                if wsh == ash:
                    w = np.array(w, float, copy=0)
                    n = np.ma.add.reduce(a * w, axis)
                    d = np.ma.add.reduce(w, axis)
                    del w
                elif wsh == (ash[axis],):
                    r = [None] * len(ash)
                    r[axis] = slice(None, None, 1)
                    w = eval("w[" + repr(tuple(r)) + "] * np.ma.ones(ash, float)")
                    n = np.ma.add.reduce(a * w, axis)
                    d = np.ma.add.reduce(w, axis, dtype=float)
                    del w, r
                else:
                    raise ValueError('average: weights wrong shape.')
        else:
            if weights is None:
                n = np.ma.add.reduce(a, axis)
                d = umath.add.reduce((~mask), axis=axis, dtype=float)
            else:
                w = np.ma.filled(weights, 0.0)
                w = np.ma.filled(weights, 0.0)
                wsh = w.shape
                if wsh == ():
                    wsh = (1,)
                if wsh == ash:
                    w = array(w, dtype=float, mask=mask, copy=0)
                    n = np.ma.add.reduce(a * w, axis)
                    d = np.ma.add.reduce(w, axis, dtype=float)
                elif wsh == (ash[axis],):
                    r = [None] * len(ash)
                    r[axis] = slice(None, None, 1)
                    w = eval("w[" + repr(tuple(r)) +
                             "] * np.ma.masked_array(np.ma.ones(ash, float), mask)")
                    n = np.ma.add.reduce(a * w, axis)
                    d = np.ma.add.reduce(w, axis, dtype=float)
                else:
                    raise ValueError('average: weights wrong shape.')
                del w
    if n is np.ma.masked or d is np.ma.masked:
        return np.ma.masked
    result = n / d
    del n

    if isinstance(result, np.ma.MaskedArray):
        if ((axis is None) or (axis == 0 and a.ndim == 1)) and \
           (result.mask is np.ma.nomask):
            result = result._data
        if returned:
            if not isinstance(d, np.ma.MaskedArray):
                d = np.ma.masked_array(d)
            if isinstance(d, ndarray) and (not d.shape == result.shape):
                d = np.ma.ones(result.shape, dtype=float) * d
    if returned:
        return result, d
    else:
        return result