"""Path utils."""

import os


def _single_input_path(path):
    # Normalize the given input `path`.
    # This function supposes that `path` is a absolute path if it starts with /,
    # else it is relative to os.environ['TL_OUTPUT'].

    if not path.startswith('/'):
        path = os.environ['TL_OUTPUT'] + path

    return os.path.abspath(os.path.normpath(os.path.expanduser(path)))


def input_path(path):
    """Normalize the given input `path`.

    This function supposes that `path` is a absolute path if it starts with /,
    else it is relative to os.environ['TL_OUTPUT'].

    Parameters
    ----------
    path : string or list of strings
        The input path.

    Returns
    -------
    norm_path : string or list of strings
        The normalized absolute input path.

    """
    if type(path) is str:
        return _single_input_path(path)
    elif type(path) is list:
        return [ _single_input_path(p) for p in path ]
    else:
        raise ValueError('Invalid input path %s' % path)


def _single_output_path(path, mkdir):
    # Normalize the given output `path`.
    # This function supposes that `path` is relative to os.environ['TL_OUTPUT'].

    path = os.environ['TL_OUTPUT'] + path
    abs_path = os.path.abspath(os.path.normpath(os.path.expanduser(path)))

    if mkdir:
        path_dir = os.path.dirname(abs_path)
        try:
            os.makedirs(path_dir)
        except OSError:
            pass

    return abs_path


def output_path(path, mkdir=True):
    """Normalize the given output `path`.

    This function supposes that `path` is relative to os.environ['TL_OUTPUT'].

    Parameters
    ----------
    path : string or list of strings
        The output path.
    mkdir : bool, optional
        Make the path dircetory if True. Default is True.

    Returns
    -------
    norm_path : string or list of strings
        The normalized absolute output path.

    """
    if type(path) is str:
        return _single_output_path(path, mkdir)
    elif type(path) is list:
        return [ _single_output_path(p, mkdir) for p in path ]
    else:
        raise ValueError('Invalid output path %s' % path)
