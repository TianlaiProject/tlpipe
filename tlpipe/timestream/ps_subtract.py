"""Subtract simulated strong point sources signal from the visibilities.

Inheritance diagram
-------------------

.. inheritance-diagram:: PsSub
   :parts: 2

"""

import numpy as np
import aipy as a
import timestream_task
from tlpipe.core import constants as const
from caput import mpiutil


class PsSub(timestream_task.TimestreamTask):
    """Subtract simulated strong point sources signal from the visibilities.

    .. note::

        This must be done after the data has been calibrated.

    """

    params_init = {
                    'ps': 'cas,cyg', # may also 'hyd', 'her', 'crab', 'vir'
                    'catalog': 'misc', # or helm,nvss
                    'span': 3600.0, # second
                  }

    prefix = 'ps_'

    def process(self, ts):

        ps = self.params['ps']
        catalog = self.params['catalog']
        span = self.params['span']

        ts.redistribute('baseline')

        int_time = ts.attrs['inttime']
        num_span = np.int(span / int_time)
        num_int = np.int(np.ceil(1.0 * const.sday / int_time)) # of one sidereal day
        nt = len(ts.local_time)
        if nt > num_int:
            raise RuntimeError('Now can only process data less than one sidereal day')

        feedno = ts['feedno'][:].tolist()
        nfreq = len(ts['freq'][:])
        pol = ts['pol'][:].tolist()
        bl = ts.local_bl[:] # local bls
        bls = [ tuple(b) for b in bl ]

        # point sources
        srclist, cutoff, catalogs = a.scripting.parse_srcs(ps, catalog)
        cat = a.src.get_catalog(srclist, cutoff, catalogs)
        # nps = len(cat) # number of point sources
        if mpiutil.rank0:
            print 'Subtracting point sources %s...' % ps

        # get transit time of the point sources
        # array
        aa = ts.array
        # aa.set_jultime(ts['jul_date'][0]) # the first obs time point

        for s in cat.values():
            # reset time of the array here
            aa.set_jultime(ts['jul_date'][0]) # the first obs time point
            next_transit = aa.next_transit(s)
            transit_time = a.phs.ephem2juldate(next_transit) # Julian date
            # if tranisit time is in the duration of the data
            if transit_time <= ts['jul_date'][-1]:
                transit_ind = np.searchsorted(ts['jul_date'][:], transit_time)
            else:
                transit_ind = nt + int((transit_time - ts['jul_date'][-1]) / int_time)
            pre_transit_ind = transit_ind - num_int # previous transit time
            inds = range(pre_transit_ind-num_span, pre_transit_ind+num_span) + range(transit_ind-num_span, transit_ind+num_span)
            inds = np.intersect1d(inds, np.arange(nt))

            for ti in inds:
                aa.set_jultime(ts['jul_date'][ti])
                s.compute(aa)
                # get fluxes vs. freq of the point source
                Sc = s.get_jys()
                # get the topocentric coordinate of the point source at the current time
                s_top = s.get_crds('top', ncrd=3)
                aa.sim_cache(s.get_crds('eq', ncrd=3)[:, np.newaxis]) # for compute bm_response and sim
                # for pi in range(len(pol)):
                for pi in xrange(2): # only cal for xx, yy
                    aa.set_active_pol(pol[pi])
                    for bi, (i, j) in enumerate(bls):
                        ai = feedno.index(i)
                        aj = feedno.index(j)
                        uij = aa.gen_uvw(ai, aj, src='z')[:, 0, :] # (rj - ri)/lambda
                        bmij = aa.bm_response(ai, aj).reshape(-1)
                        vis_sim = Sc * bmij * np.exp(-2.0J * np.pi * np.dot(s_top, uij))
                        ts.local_vis[ti, :, pi, bi] -= vis_sim # subtract this ps


        return super(PsSub, self).process(ts)
