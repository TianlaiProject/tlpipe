"""Phase the source-phased visibility data to the zenith.

Inheritance diagram
-------------------

.. inheritance-diagram:: Phs2zen
   :parts: 2

"""

import numpy as np
import ephem
import aipy as a
import timestream_task
from tlpipe.utils.date_util import get_ephdate
from caput import mpiutil


class Phs2zen(timestream_task.TimestreamTask):
    """Phase the source-phased visibility data to the zenith

    The phasing is done by multiply the visibility by the reference phase of
    the given *source*, as

    .. math:: V_{ij}^{\\text{phs}} = V_{ij} \\cdot e^{2 \\pi i \\, \\boldsymbol{s}_0 \\cdot \\boldsymbol{u}_{ij}}

    This is just the inverse operation of task
    :class:`~tlpipe.timestream.phs2src.Phs2src`.

    """

    params_init = {
                    'source': 'cyg', # <src_name> or <ra XX[:XX:xx]>_<dec XX[:XX:xx]> or <time y/m/d h:m:s> (array pointing of this local time)
                    'catalog': 'misc', # or helm,nvss
                  }

    prefix = 'p2z_'

    def process(self, ts):

        source = self.params['source']
        catalog = self.params['catalog']

        if 'Dish' in ts.attrs['telescope']:
            ant_type = 'dish'
        elif 'Cylinder' in ts.attrs['telescope']:
            ant_type = 'cylinder'
        else:
            raise RuntimeError('Unknown antenna type %s' % ts.attrs['telescope'])

        feedno = ts['feedno'][:].tolist()

        ts.redistribute(0) # make time the dist axis

        # array
        aa = ts.array

        try:
            # convert an observing time to the ra_dec of the array pointing of that time
            src_time = get_ephdate(source, tzone=ts.attrs['timezone']) # utc time
            aa.date = str(ephem.Date(src_time)) # utc time
            # print 'date:', aa.date
            antpointing = np.radians(ts['antpointing'][-1, :, :]) # radians
            azs = antpointing[:, 0]
            alts = antpointing[:, 1]
            if np.allclose(azs, azs[0]) and np.allclose(alts, alts[0]):
                az = azs[0]
                alt = alts[0]
            else:
                raise ValueError('Antennas do not have a common pointing')
            az, alt = ephem.degrees(az), ephem.degrees(alt)
            src_ra, src_dec = aa.radec_of(az, alt)
            source = '%s_%s' % (src_ra, src_dec)
            # print 'source:', source
        except ValueError:
            pass

        # source
        srclist, cutoff, catalogs = a.scripting.parse_srcs(source, catalog)
        cat = a.src.get_catalog(srclist, cutoff, catalogs)
        assert(len(cat) == 1), 'Allow only one source'
        s = cat.values()[0]
        if mpiutil.rank0:
            print 'Undo the source-phase %s to phase to the zenith.' % source


        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']

        ts.time_and_bl_data_operate(self.phs, show_progress=show_progress, progress_step=progress_step, aa=aa, s=s)

        return super(Phs2zen, self).process(ts)

    def phs(self, vis, vis_mask, li, gi, tbl, ts, **kwargs):
        """Function that does the actual phs."""

        t = tbl[0]
        ai, aj = tbl[1]
        aa = kwargs.get('aa')
        s = kwargs.get('s')

        feedno = ts['feedno'][:].tolist()
        i = feedno.index(ai)
        j = feedno.index(aj)

        aa.set_jultime(t)
        s.compute(aa)
        # get fluxes vs. freq of the calibrator
        # Sc = s.get_jys()
        # get the topocentric coordinate of the calibrator at the current time
        s_top = s.get_crds('top', ncrd=3)
        # aa.sim_cache(cat.get_crds('eq', ncrd=3)) # for compute bm_response and sim
        uij = aa.gen_uvw(i, j, src='z').squeeze() # (rj - ri)/lambda

        vis[:] = vis * np.exp(-2.0J * np.pi * np.dot(s_top, uij))[:, np.newaxis]
