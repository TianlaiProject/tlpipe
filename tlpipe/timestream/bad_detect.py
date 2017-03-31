"""Bad/Exceptional visibility values detect.

Inheritance diagram
-------------------

.. inheritance-diagram:: Detect
   :parts: 2

"""

import itertools
import numpy as np
import tod_task
from raw_timestream import RawTimestream
from timestream import Timestream
from caput import mpiutil


class Detect(tod_task.TaskTimestream):
    """Bad/Exceptional visibility values detect.

    This task does a simple bad/exceptional values detection by mask those
    values that are not finite, and those have non-zero imaginary part of an
    auto-correlation.

    """

    params_init = {}

    prefix = 'bd_'

    def process(self, ts):

        ts.redistribute('baseline')

        vis = ts.local_vis
        vis_mask = ts.local_vis_mask

        # mask non-finite vis values
        vis_mask[:] = np.where(np.isfinite(vis), vis_mask, True)

        # mask where the imaginary part of an auto-correlation is non-zero
        if isinstance(ts, RawTimestream):
            for bi, (fi, fj) in enumerate(ts.local_bl):
                if fi == fj:
                    vis_mask[..., bi] = np.where(vis[..., bi].imag == 0.0, vis_mask[..., bi], True)
        elif isinstance(ts, Timestream):
            for bi, (fi, fj) in enumerate(ts.local_bl):
                for pi, pol in enumerate(ts.local_pol):
                    pol = ts.pol_dict[pol]
                    if fi == fj and pol in ['xx', 'yy']:
                        vis_mask[..., pi, bi] = np.where(vis[..., pi, bi].imag == 0.0, vis_mask[..., pi, bi], True)

        # mask bl that have no signal
        problematic_bls = []
        bad_bls = []
        for bi in xrange(len(ts.local_bl)):
            if isinstance(ts, RawTimestream):
                # create a copy of vis for this bi, and fill 0 in masked positions
                vis1 = np.where(ts.local_vis_mask[..., bi], 0, ts.local_vis[..., bi])
                if np.sum(ts.local_vis_mask[..., bi]) >= 0.5 * np.prod(ts.local_vis[..., bi].shape):
                    bl = tuple(ts.local_bl[bi])
                    problematic_bls.append(bl)
                    # print 'Problematic baseline: (%d, %d)' % bl
                if np.allclose(vis1, 0): # all zeros
                    ts.local_vis_mask[..., bi] = True # mask all
                    bl = tuple(ts.local_bl[bi])
                    bad_bls.append(bl)
                    # print 'Bad baseline: (%d, %d)' % bl
            elif isinstance(ts, Timestream):
                for pi, pol in enumerate(ts.local_pol):
                    pol = ts.pol_dict[pol]
                    # create a copy of vis for this bi, and fill 0 in masked positions
                    vis1 = np.where(ts.local_vis_mask[..., pi, bi], 0, ts.local_vis[..., pi, bi])
                    if np.sum(ts.local_vis_mask[..., pi, bi]) >= 0.5 * np.prod(ts.local_vis[..., pi, bi].shape):
                        bl = tuple(ts.local_bl[bi])
                        problematic_bls.append((bl, pol))
                        # print 'Problematic baseline: (%d, %d)' % bl
                    if np.allclose(vis1, 0): # all zeros
                        ts.local_vis_mask[..., pi, bi] = True # mask all
                        bl = tuple(ts.local_bl[bi])
                        bad_bls.append((bl, pol))
                        # print 'Bad baseline: (%d, %d)' % bl


        # gather list
        comm = mpiutil.world
        if comm is not None:
            problematic_bls = list(itertools.chain(*comm.allgather(problematic_bls)))
            bad_bls = list(itertools.chain(*comm.allgather(bad_bls)))

        if mpiutil.rank0:
            print 'Problematic baseline: ', problematic_bls
            print 'Bad baseline: ', bad_bls

        return super(Detect, self).process(ts)
