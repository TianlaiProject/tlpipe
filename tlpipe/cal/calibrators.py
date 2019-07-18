import numpy as np
import ephem
import aipy as a


# 20 sources from Perley and Butler, 2017, An accurate flux density scale from 50 MHz to 50 GHz
src_data = {
    # key       name               RA            DEC          a0       a1       a2       a3      a4       a5          angle_size
    'j0133': ('J0133-3629',   'xx:xx:xxx',   'xx:xx:xxx',   1.0440, -0.6619, -0.2252,  0.0,     0.0,     0.0,    np.radians(14.0 / 60)),
    '3c48' : ('3C48',         'xx:xx:xxx',   'xx:xx:xxx',   1.3253, -0.7553, -0.1914,  0.0498,  0.0,     0.0,    np.radians(1.2 / 3600)),
    'for'  : ('Fornax A',     '03:22:41.7',  '-37:12:30',   2.2175, -0.6606,  0.0,     0.0,     0.0,     0.0,    np.radians(55.0 / 60)),
    '3c123': ('3C123',        'xx:xx:xxx',   'xx:xx:xxx',   1.8017, -0.7884, -0.1035, -0.0248,  0.0090,  0.0,    np.radians(44.0 / 3600)),
    'j0444': ('J0444-2809',   'xx:xx:xxx',   'xx:xx:xxx',   0.9710, -0.8938, -0.1176,  0.0,     0.0,     0.0,    np.radians(2.0 / 60)),
    '3c138': ('3C138',        'xx:xx:xxx',   'xx:xx:xxx',   1.0088, -0.4981, -0.1552, -0.0102,  0.0223,  0.0,    np.radians(0.7 / 3600)),
    'pic'  : ('Pictor A',     '05:19:49.7',  '-45:46:45',   1.9380, -0.7470, -0.0739,  0.0,     0.0,     0.0,    np.radians(8.3 / 60)),
    'crab' : ('Taurus A',     '05:34:32.0',  '+22:00:52',   2.9516, -0.2173, -0.0473, -0.0674,  0.0,     0.0,    np.radians(0.0)),
    '3c147': ('3C147',        'xx:xx:xxx',   'xx:xx:xxx',   1.4516, -0.6961, -0.2007,  0.0640, -0.0464,  0.0289, np.radians(0.9 / 3600)),
    '3c196': ('3C196',        'xx:xx:xxx',   'xx:xx:xxx',   1.2872, -0.8530, -0.1534, -0.0200,  0.0201,  0.0,    np.radians(7.0 / 3600)),
    'hyd'  : ('Hydra A',      '09:18:05.7',  '-12:05:44',   1.7795, -0.9176, -0.0843, -0.0139,  0.0295,  0.0,    np.radians(8.0 / 60)),
    'vir'  : ('Virgo A',      '12:30:49.4',  '+12:23:28',   2.4466, -0.8116, -0.0483,  0.0,     0.0,     0.0,    np.radians(14.0 / 60)),
    '3c286': ('3C286',        'xx:xx:xxx',   'xx:xx:xxx',   1.2481, -0.4507, -0.1798,  0.0357,  0.0,     0.0,    np.radians(3.0 / 3600)),
    '3c295': ('3C295',        'xx:xx:xxx',   'xx:xx:xxx',   1.4701, -0.7658, -0.2780, -0.0347,  0.0399,  0.0,    np.radians(5.0 / 3600)),
    'her'  : ('Hercules A',   '16:51:08.15', '4:59:33.3',   1.8298, -0.1247, -0.0951,  0.0,     0.0,     0.0,    np.radians(3.1 / 60)),
    '3c353': ('3C353',        'xx:xx:xxx',   'xx:xx:xxx',   1.8627, -0.6938, -0.0998, -0.0732,  0.0,     0.0,    np.radians(5.3)),
    '3c380': ('3C380',        'xx:xx:xxx',   'xx:xx:xxx',   1.2320, -0.7909,  0.0947,  0.0976, -0.1794, -0.1566, np.radians(0.0)),
    'cyg'  : ('Cygnus A',     '19:59:28.3',  '+40:44:02',   3.3498, -1.0022, -0.2246,  0.0227,  0.0425,  0.0,    np.radians(2.0 / 60)),
    '3c444': ('3C444',        'xx:xx:xxx',   'xx:xx:xxx',   1.1064, -1.0052, -0.0750, -0.0767,  0.0,     0.0,    np.radians(2.0 / 60)),
    'cas'  : ('Cassiopeia A', '23:23:27.94', '+58:48:42.4', 3.3584, -0.7518, -0.0347, -0.0705,  0.0,     0.0,    np.radians(0.0)),
}


class RadioBody(object):
    """A celestial source."""
    def __init__(self, name, poly_coeffs, ionref, srcshape):
        self.src_name = name
        self.poly_coeffs = poly_coeffs
        self.ionref = list(ionref)
        self.srcshape = list(srcshape)

    def __str__(self):
        return "%s" % self.src_name

    def compute(self, observer):
        """Update coordinates relative to the provided `observer`.

        Must be called at each time step before accessing information.
        """
        # Generate a map for projecting baselines to uvw coordinates
        self.map = a.coord.eq2top_m(observer.sidereal_time() - self.ra, self.dec)

    def get_crds(self, crdsys, ncrd=3):
        """Return the coordinates of this location in the desired coordinate
        system ('eq','top') in the current epoch.

        If ncrd=2, angular coordinates (ra/dec or az/alt) are returned,
        and if ncrd=3, xyz coordinates are returned.
        """
        assert(crdsys in ('eq','top'))
        assert(ncrd in (2,3))
        if crdsys == 'eq':
            if ncrd == 2:
                return (self.ra, self.dec)
            return a.coord.radec2eq((self.ra, self.dec))
        else:
            if ncrd == 2:
                return (self.az, self.alt)
            return a.coord.azalt2top((self.az, self.alt))

    def get_jys(self, afreq):
        """Update fluxes at the given frequencies.

        Parameters
        ----------
        afreq : float or float array
            frequency in GHz.
        """
        log_nv = np.log10(afreq)
        a0, a1, a2, a3, a4, a5 = self.poly_coeffs
        log_S = a0 + a1 * log_nv + a2 * log_nv**2 + a3 * log_nv**3 + a4 * log_nv**4 + a5 * log_nv**5
        return 10**log_S


class RadioFixedBody(ephem.FixedBody, RadioBody):
    """A source at fixed RA,DEC.  Combines ephem.FixedBody with RadioBody."""
    def __init__(self, ra, dec, poly_coeffs, name='', epoch=ephem.J2000, ionref=(0.0, 0.0), srcshape=(0.0, 0.0, 0.0), **kwargs):
        RadioBody.__init__(self, name, poly_coeffs, ionref, srcshape)
        ephem.FixedBody.__init__(self)
        self._ra, self._dec = ra, dec
        self._epoch = epoch

    def __str__(self):
        if self._dec<0:
            return RadioBody.__str__(self) + '\t' + str(self._ra) +'\t'+ str(self._dec)
        else:
            return RadioBody.__str__(self) + '\t' + str(self._ra) +'\t'+'+' + str(self._dec)

    def compute(self, observer):
        ephem.FixedBody.compute(self, observer)
        RadioBody.compute(self, observer)



_cat = None

def get_src(name):
    """Return a source in `src_data` withe key == `name`."""
    name, ra, dec, a0, a1, a2, a3, a4, a5, srcshape = src_data[name]
    try:
        len(srcshape)
    except(TypeError):
        srcshape = (srcshape, srcshape, 0.)
    return RadioFixedBody(ra, dec, poly_coeffs=(a0, a1, a2, a3, a4, a5), name=name, srcshape=srcshape)

def get_srcs(srcs=None, cutoff=None):
    global _cat
    if _cat is None:
        _cat = a.fit.SrcCatalog()
        srclist = []
        for s in src_data:
            srclist.append(get_src(s))
        _cat.add_srcs(srclist)
    if srcs is None:
        if cutoff is None:
            srcs = _cat.keys()
        else:
            cut, fq = cutoff
            fq = n.array([fq])
            for s in _misccat.keys():
                _cat[s].update_jys(fq)
            srcs = [ s for s in _misccat.keys() if _cat[s].jys[0] > cut ]

    srclist = []
    for s in srcs:
        try:
            srclist.append(_cat[s])
        except(KeyError):
            pass

    return srclist


if __name__ == '__main__':
    name = 'cyg'
    s = get_src(name)
    print s

    freq = 0.75 # MHz
    print s.get_jys(freq)

    freqs = np.logspace(-1.0, 1.0)
    jys = s.get_jys(freqs)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.figure()
    plt.loglog(freqs, jys)
    plt.savefig('cyg_flux.png')
    plt.close()

    srclist, cutoff, catalogs = a.scripting.parse_srcs(name, 'misc')
    cat = a.src.get_catalog(srclist, cutoff, catalogs)
    s1 = cat.values()[0]
    s1.update_jys(freq)
    print s1.get_jys()