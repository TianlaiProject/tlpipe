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
from caput import mpiutil


class Detect(tod_task.TaskTimestream):
    """Bad/Exceptional visibility values detect.

    This task does a simple bad/exceptional values detection by mask those
    values that are not finite, and those have non-zero imaginary part of an
    auto-correlation.

    """

    params_init = {
                    'num_auto': 6, # least number of feeds to check auto-correlation
                    'threshold': 2.24, # threshold of the MAD-median rule
                  }

    prefix = 'bd_'

    def process(self, rt):

        assert isinstance(rt, RawTimestream), '%s only works for RawTimestream object currently' % self.__class__.__name__

        num_auto = self.params['num_auto']
        threshold = self.params['threshold']

        rt.redistribute('time')

        vis = rt.local_vis
        vis_mask = rt.local_vis_mask

        # mask non-finite vis values
        vis_mask[:] = np.where(np.isfinite(vis), vis_mask, True)

        xx_auto = []
        yy_auto = []
        # mask where the imaginary part of an auto-correlation is non-zero
        for bi, (fi, fj) in enumerate(rt.local_bl):
            if fi == fj:
                vis_mask[..., bi] = np.where(vis[..., bi].imag == 0.0, vis_mask[..., bi], True)
                if fi % 2 == 1:
                    xx_auto.append(bi)
                else:
                    yy_auto.append(bi)

        # mask values exceed given threshold by using the MAD-median rule
        # see Wilcox, 2014, Modern robust statistical methods can provide substantially higher power and a deeper understanding of data
        for auto in [xx_auto, yy_auto]:
            if len(auto) >= num_auto:
                vis1 = np.ma.array(vis[..., auto].real, mask=vis_mask[..., auto])
                median = np.ma.median(vis1, axis=2)
                abs_diff = np.abs(vis1-median[:, :, np.newaxis])
                mad = np.ma.median(abs_diff, axis=2) / 0.6745
                cnt = vis1.count(axis=2)
                mask1 = vis_mask[..., auto].copy()
                # avoid statistical error for small cnt
                cond = np.logical_and(abs_diff>threshold*mad[:, :, np.newaxis], np.tile(cnt[:, :, np.newaxis], (1, 1, len(auto)))>=num_auto)
                mask1 = np.where(cond, True, mask1)

                vis_mask[..., auto] = mask1 # replace with the new mask

        rt.redistribute('baseline')

        # mask bl that have no signal
        problematic_bls = []
        bad_bls = []
        for bi in xrange(len(rt.local_bl)):
            # create a copy of vis for this bi, and fill 0 in masked positions
            vis1 = np.where(rt.local_vis_mask[..., bi], 0, rt.local_vis[..., bi])
            if np.sum(rt.local_vis_mask) >= 0.5 * np.prod(rt.local_vis.shape):
                bl = tuple(rt.local_bl[bi])
                problematic_bls.append(bl)
                # print 'Problematic baseline: (%d, %d)' % bl
            if np.allclose(vis1, 0): # all zeros
                rt.local_vis_mask[..., bi] = True
                bl = tuple(rt.local_bl[bi])
                bad_bls.append(bl)
                # print 'Bad baseline: (%d, %d)' % bl

        # gather list
        comm = mpiutil.world
        if comm is not None:
            problematic_bls = list(itertools.chain(*comm.allgather(problematic_bls)))
            bad_bls = list(itertools.chain(*comm.allgather(bad_bls)))

        if mpiutil.rank0:
            print 'Problematic baseline: ', problematic_bls
            print 'Bad baseline: ', bad_bls

        rt.add_history(self.history)

        # rt.info()

        return rt
