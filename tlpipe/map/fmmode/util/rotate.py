import numpy as np
import healpy as hp


def rotate_map(hpmap, nest=False, rot=None, coord=None, inv=False, deg=True, eulertype='ZYX'):
    """Rotate a input helapix map `hpmap`, including astronomical coordinate systems transform.

    Parameters
    ----------
    hpmap : float, array-like
        An array containing the map, supports masked maps.
    nest : bool, optional
        If True, ordering scheme is NESTED. Default: False (RING)
    rot : scalar or sequence, optional
        Describe the rotation to apply. In the form (lon, lat, psi) (unit: degrees):
        the point at longitude *lon* and latitude *lat* will be at the center.
        An additional rotation of angle *psi* around this direction is applied.
    coord : None or sequence of str, optional
        Either one of 'G', 'E' or 'C' to describe the coordinate system of
        the map, or a sequence of 2 of these to rotate the map from the first
        to the second coordinate system. If rot is also given, the coordinate
        transform is applied first, and then the rotation.
    inv : bool
        If True, the inverse rotation is defined. (Default: False)
    deg : bool
        If True, angles are assumed to be in degree. (Default: True)
    eulertype : str
        The Euler angle convention used. See euler_matrix_new().

"""

    npix = hpmap.shape[-1]
    nside = hp.npix2nside(npix)
    theta, phi = hp.pix2ang(nside, np.arange(npix))

    # coord or rotation
    r = hp.Rotator(rot=rot, coord=coord, inv=inv, deg=deg, eulertype=eulertype)
    theta, phi = r(theta, phi)

    pix = hp.ang2pix(nside, theta, phi, nest=nest)

    return hpmap[..., pix]
