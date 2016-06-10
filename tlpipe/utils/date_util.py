"""Date and time utils."""

import re
import ephem


def get_ephdate(local_time, tzone='UTC+08h'):
    """Convert `local_time` to ephem utc date.

    Parameters
    ----------
    local_time : string, python date or datetime object, etc.
        Local time that can be passed to ephem.Date function.
        Refer to http://rhodesmill.org/pyephem/date.html
    tzone : string, optional
        Time zone in format 'UTC[+/-]xxh'. Defaut: UTC+08h.

    Returns
    -------
    utc_time : float
        A float number representation of a UTC PyEphem date.

    See Also
    --------
    `get_juldate`
    """
    local_time = ephem.Date(local_time)
    pattern = '[-+]?\d+'
    tz = re.search(pattern, tzone).group()
    tz = int(tz)
    utc_time = local_time - tz * ephem.hour

    return utc_time


def get_juldate(local_time, tzone='UTC+08h'):
    """Convert `local_time` to Julian date.

    Parameters
    ----------
    local_time : string, python date or datetime object, etc.
        Local time that can be passed to ephem.Date function.
        Refer to http://rhodesmill.org/pyephem/date.html
    tzone : string, optional
        Time zone in format 'UTC[+/-]xxh'. Defaut: UTC+08h.

    Returns
    -------
    julian_date : float
        A float number representation of a Julian date.

    See Also
    --------
    `get_ephdate`
    """

    return ephem.julian_date(get_ephdate(local_time, tzone))
