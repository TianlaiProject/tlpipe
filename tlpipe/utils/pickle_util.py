"""Some functions related to pickle."""

import pickle


def get_value(val):
    """Return unpickled value of `val` if it is a pickled object, else just return itself."""
    try:
        val = pickle.loads(val)
    except:
        pass

    return val
