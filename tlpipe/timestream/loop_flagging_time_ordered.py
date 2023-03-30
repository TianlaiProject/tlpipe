"""RFI flagging by using Local Outlier Probabilities (LoOP).

Inheritance diagram
-------------------

.. inheritance-diagram:: Flag
   :parts: 2

"""

import numpy as np
from caput import mpiutil
from . import timestream_task
from tlpipe.container.timestream import Timestream
from tlpipe.rfi import loop
from tlpipe.utils import progress


class Flag(timestream_task.TimestreamTask):
    """RFI flagging by using Local Outlier Probabilities (LoOP).

    LoOP is a local density based outlier detection method which provides
    outlier scores in the range of [0, 1] that are directly interpretable
    as the probability of a sample being an outlier.

    """

    params_init = {
                    'n_neighbors': 20,
                    'probability_threshold': 0.95, # considered to be outlier when higher than this val
                  }

    prefix = 'lf_'

    def process(self, ts):
        via_memmap = self.params['via_memmap']
        n_neighbors = self.params['n_neighbors']
        probability_threshold = self.params['probability_threshold']

        assert isinstance(ts, Timestream), '%s only works for Timestream object' % self.__class__.__name__

        ts.redistribute('time', via_memmap=via_memmap)

        clf = loop.LocalOutlierProbability(n_neighbors=n_neighbors)

        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']

        if show_progress and mpiutil.rank0:
            pg = progress.Progress(n, step=progress_step)
        nt, nf, _, nbl = ts.local_vis.shape
        for ti in range(nt):
            if show_progress and mpiutil.rank0:
                pg.show(i)
            for fi in range(nf):
                for pol in ['xx', 'yy']:
                    pi = ts.pol_dict[pol]
                    inds = np.where(np.logical_and(ts.local_vis_mask[ti, fi, pi]==False, ts.local_vis[ti, fi, pi].imag!=0))[0]
                    if len(inds) < 0.3 * nbl:
                        # too less valid data
                        ts.local_vis_mask[ti, fi, pi, :] = True
                    else:
                        vis = ts.local_vis[ti, fi, pi, inds]
                        X = np.vstack([vis.real, vis.imag]).T
                        clf.fit(X)
                        p = clf.local_outlier_probabilities
                        ts.local_vis_mask[ti, fi, pi, inds[p>probability_threshold]] = True


        return super(Flag, self).process(ts)
