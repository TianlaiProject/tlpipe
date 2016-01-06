"""Date and time utils."""

import ephem


def get_ephdate(local_time, tzone='UTC+08'):
    """Convert `local_time` to ephem utc date.

    Parameters
    ----------
    local_time : string
        Local time in any string format that can be recognized by ephem.
        Refer to http://rhodesmill.org/pyephem/date.html
    tzone : string, optional
        Time zone in format 'UTC[+/-]xx'. Defaut: UTC+08.

    Returns
    -------
    utc_time : float
        A float number representation of a UTC PyEphem date.

    See Also
    --------
    `get_juldate`
    """
    local_time = ephem.Date(local_time)
    tz = int(tzone[3:])
    utc_time = local_time - tz * ephem.hour

    return utc_time


def get_juldate(local_time, tzone='UTC+08'):
    """Convert `local_time` to Julian date.

    Parameters
    ----------
    local_time : string
        Local time in any string format that can be recognized by ephem.
        Refer to http://rhodesmill.org/pyephem/date.html
    tzone : string, optional
        Time zone in format 'UTC[+/-]xx'. Defaut: UTC+08.

    Returns
    -------
    julian_date : float
        A float number representation of a Julian date.

    See Also
    --------
    `get_ephdate`
    """

    return ephem.julian_date(get_ephdate(local_time, tzone))
