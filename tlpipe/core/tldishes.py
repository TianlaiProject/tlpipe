import os
import numpy as np
import aipy as ap


def xyz2XYZ_m(lat):
    """
    Matrix of coordinates conversion through xyz to XYZ.
    xyz coord: z toward zenith, x toward East, y toward North, xy in the horizon plane;
    XYZ coord: Z toward north pole, X in the local meridian plane, Y toward East, XY plane parallel to equatorial plane.
    Arguments:
    - `lat`: latitude of the observing position, in unit radian.
    """
    sin_a, cos_a = np.sin(lat), np.cos(lat)
    zero = np.zeros_like(lat)
    one = np.ones_like(lat)
    mat =  np.array([[  zero,   -sin_a,   cos_a  ],
                     [   one,     zero,    zero  ],
                     [  zero,    cos_a,   sin_a  ]])
    if len(mat.shape) == 3: mat = mat.transpose([2, 0, 1])
    return mat

def latlong_conv(lat):
    """
    Covert the string represent latitude/longitude to radian.
    Arguments:
    - `lat`: string represent latitude
    """
    str_lat = lat.split(":")
    lat = 0.0
    for n in range(len(str_lat)):
        lat += float(str_lat[n])/(60.0**n)
    return lat * np.pi / 180.0

# location of the antenna array
# lat = '44:9:11.00'
# lon = '91:48:23.00'
lat = '44:9:8.439'
lon = '91:48:20.177'
#lat = '64:54:16.7' # test for Jsim, move tl to lat=58, make CasA cross Z
#lon = '91:48:20.177'
elev = 1504.3 # m

# (+E, +N) coordinates of each antenna in unit m
dishes_coord = np.loadtxt(os.path.dirname(__file__) + '/16dishes_coord.txt')
center_coord = dishes_coord[15] # central antenna coordinate
dishes_coord -= center_coord
# antenna positions
ant_pos_m = np.zeros((dishes_coord.shape[0], 3), dtype=dishes_coord.dtype)
ant_pos_m[:, :2] = dishes_coord
nants = ant_pos_m.shape[0]
m2ns = 100.0 / ap.const.c * 1.0e9 # c in unit cm
ant_pos_ns = m2ns * ant_pos_m
ant_pos_ns = np.dot(xyz2XYZ_m(latlong_conv(lat)), ant_pos_ns.T).T


prms = {
    'loc': (lat, lon, elev),
    'antpos': ant_pos_ns,
    'delays': [0.] * nants, # zero delays for all antennas
    'offsets': [0.] * nants, # zero offsets for all antennas
    'amps': [1.] * nants,
    'bp_r': [np.array([1.])] * nants,
    'bp_i': [np.array([0.])] * nants,
    'beam': ap.fit.Beam2DGaussian,
    'bm_xwidth': np.radians(4.0),
    'bm_ywidth': np.radians(4.0),
    'pointing': (0.0, latlong_conv(lat), 0.0), # pointing to the North Pole, az (clockwise around z = up, 0 at x axis = north), alt (from horizon), also see coord.py
    # 'pointing': (0.0, np.pi/2, 0.0), # zenith
}

def get_aa(freqs):
    '''Return the AntennaArray to be used for simulation.'''
    beam = prms['beam'](freqs)
    try: beam.set_params(prms)
    except(AttributeError): pass
    # location = prms['loc']
    antennas = []
    pointing = prms['pointing']
    assert(len(prms['delays']) == nants and len(prms['offsets']) == nants and len(prms['bp_r']) == nants and len(prms['bp_i']) == nants and len(prms['amps']) == nants)
    for pos, dly, off, bp_r, bp_i, amp in zip(prms['antpos'], prms['delays'], prms['offsets'], prms['bp_r'], prms['bp_i'], prms['amps']):
        # antennas.append(ap.pol.Antenna(pos[0],pos[1],pos[2], beam, phsoff=[dly, off], bp_r=bp_r, bp_i=bp_i, amp=amp, pointing=pointing))
        antennas.append(ap.pol.Antenna(pos[0],pos[1],pos[2], beam, phsoff={'x':[dly, off], 'y':[dly, off]}, bp_r={'x':bp_r, 'y':bp_r}, bp_i={'x':bp_i, 'y':bp_i}, amp={'x':amp, 'y':amp}, pointing=pointing))
    aa = ap.pol.AntennaArray(prms['loc'], antennas)
    return aa
