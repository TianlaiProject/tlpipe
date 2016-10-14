"""Path utils."""

import os


def iter_path(path, iteration):
    """Insert current iteration flag to the file path.

    Parameters
    ----------
    path : string
        The output path.
    iteration : integer
        The iteration number.

    Returns
    -------
    new_path : string
        The generated new path which has the iteration been inserted.

    """
    head, tail = os.path.split(path)
    return os.path.join(head, '%d' % iteration, tail)


def _single_input_path(path):
    # Normalize the given input `path`.
    # This function supposes that `path` is a absolute path if it starts with /,
    # else it is relative to os.environ['TL_OUTPUT'].

    if not path.startswith('/'):
        path = os.environ['TL_OUTPUT'] + path

    return os.path.abspath(os.path.normpath(os.path.expanduser(path)))


def input_path(path, iteration=None):
    """Normalize the given input `path`.

    This function supposes that `path` is a absolute path if it starts with /,
    else it is relative to os.environ['TL_OUTPUT'].

    Parameters
    ----------
    path : string or list of strings
        The input path.
    iteration : None or integer, optional
        The iteration number. If it is not None, it will be inserted into the
        path before the path's basename. Default is None.

    Returns
    -------
    norm_path : string or list of strings
        The normalized absolute input path.

    """
    func = iter_path if iteration is not None else lambda x, y: x

    if type(path) is str:
        return _single_input_path(func(path, iteration))
    elif type(path) is list:
        return [ _single_input_path(func(p, iteration)) for p in path ]
    else:
        raise ValueError('Invalid input path %s' % path)


def _single_output_path(path, relative, mkdir):
    # Normalize the given output `path`.
    # This function supposes that `path` is relative to os.environ['TL_OUTPUT'] if `relative` is True.

    if relative:
        path = os.environ['TL_OUTPUT'] + path
    abs_path = os.path.abspath(os.path.normpath(os.path.expanduser(path)))

    if mkdir:
        path_dir = os.path.dirname(abs_path)
        try:
            os.makedirs(path_dir)
        except OSError:
            pass

    return abs_path


def output_path(path, relative=True, mkdir=True, iteration=None):
    """Normalize the given output `path`.

    This function supposes that `path` is relative to os.environ['TL_OUTPUT'] if
    `relative` is True.

    Parameters
    ----------
    path : string or list of strings
        The output path.
    relative : bool, optional
        The `path` is relative to os.environ['TL_OUTPUT'] if True. Default is True.
    mkdir : bool, optional
        Make the path directory if True. Default is True.
    iteration : None or integer, optional
        The iteration number. If it is not None, it will be inserted into the
        path before the path's basename. Default is None.

    Returns
    -------
    norm_path : string or list of strings
        The normalized absolute output path.

    """
    func = iter_path if iteration is not None else lambda x, y: x

    if type(path) is str:
        return _single_output_path(func(path, iteration), relative, mkdir)
    elif type(path) is list:
        return [ _single_output_path(func(p, iteration), relative, mkdir) for p in path ]
    else:
        raise ValueError('Invalid output path %s' % path)
