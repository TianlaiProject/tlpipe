"""RFI flagging by using Local Outlier Probabilities (LoOP).

Inheritance diagram
-------------------

.. inheritance-diagram:: Flag
   :parts: 2

"""

import numpy as np
import timestream_task
from tlpipe.container.raw_timestream import RawTimestream
from tlpipe.container.timestream import Timestream
from tlpipe.rfi import loop


class Flag(timestream_task.TimestreamTask):
    """RFI flagging by using Local Outlier Probabilities (LoOP).

    LoOP is a local density based outlier detection method which provides
    outlier scores in the range of [0, 1] that are directly interpretable
    as the probability of a sample being an outlier.

    """

    params_init = {
                    'n_neighbors': 20,
                    'time_window': 75,
                    'freq_window': 5,
                    'probability_threshold': 0.95, # considered to be outlier when higher than this val
                  }

    prefix = 'lf_'

    def process(self, ts):

        ts.redistribute('baseline')

        if isinstance(ts, RawTimestream):
            func = ts.bl_data_operate
        elif isinstance(ts, Timestream):
            func = ts.pol_and_bl_data_operate

        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']

        func(self.flag, full_data=True, show_progress=show_progress, progress_step=progress_step, keep_dist_axis=False)

        return super(Flag, self).process(ts)

    def flag(self, vis, vis_mask, li, gi, tf, ts, **kwargs):
        """Function that does the actual flag."""

        # if all have been masked, no need to flag again
        if vis_mask.all():
            return

        n_neighbors = self.params['n_neighbors']
        time_window = self.params['time_window']
        freq_window = self.params['freq_window']
        probability_threshold = self.params['probability_threshold']

        clf = loop.LocalOutlierProbability(n_neighbors=n_neighbors)

        nt, nf = vis.shape
        tisf = range(nt) # forward time inds
        tisb = tisf[::-1] # backward time inds
        for fi in range(nf):
            # to make larger overlap area for incremental fit
            if fi % 2 == 0:
                tis = tisf
            else:
                tis = tisb
            for ti in tis:
                if vis_mask[ti, fi]:
                    continue

                lti = max(0, ti - time_window/2)
                hti = min(nt, ti + time_window/2 + 1)
                lfi = max(0, fi - freq_window/2)
                hfi = min(nf, fi + freq_window/2 + 1)

                sec = vis[lti:hti, lfi:hfi].flatten()
                sec_mask = vis_mask[lti:hti, lfi:hfi].flatten()

                vi = np.where(sec_mask==False)[0] # valid inds
                if len(vi) < 10:
                    # less than 10 valid value in the neighbour of this point,
                    # so it is likely to be an outlier
                    vis_mask[ti, fi] = True
                    continue

                ### for check
                # X1 = np.vstack([sec[vi].real, sec[vi].imag]).T
                # clf.fast_fit(X1)
                # p1 = clf.local_outlier_probabilities

                if not clf._fit:
                    X = np.vstack([sec[vi].real, sec[vi].imag]).T
                    clf.fit(X)
                else:
                    iis_old = np.array([ ti_*nf + fi_ for ti_ in range(lti_old, hti_old) for fi_ in range(lfi_old, hfi_old) ])[vi_old]
                    iis = np.array([ ti_*nf + fi_ for ti_ in range(lti, hti) for fi_ in range(lfi, hfi) ])[vi]
                    pop_inds = np.setdiff1d(iis_old, iis) - (lti_old*nf + lfi_old)
                    new_inds = np.setdiff1d(iis, iis_old) - (lti*nf + lfi)
                    pop_inds = [ np.where(vi_old == pi)[0][0] for pi in pop_inds if pi in vi_old ]

                    if len(new_inds) == 0:
                        new_X = np.zeros((0, 2), dtype=sec.real.dtype)
                    else:
                        new_X = np.vstack([sec[new_inds].real, sec[new_inds].imag]).T
                    clf.refit(new_X, pop_inds)

                lti_old = lti
                hti_old = hti
                lfi_old = lfi
                hfi_old = hfi
                vi_old = vi

                p = clf.local_outlier_probabilities

                ### for check
                # if not np.allclose(p, p1):
                #     import pdb; pdb.set_trace()

                ci = np.where(vi == (ti - lti)*nf + (fi - lfi))[0]
                if p[ci] > probability_threshold:
                    # is an outlier
                    vis_mask[ti, fi] = True


                    # # plot
                    # import tlpipe.plot
                    # import matplotlib.pyplot as plt

                    # plt.figure()
                    # plt.scatter(sec[vi].real, sec[vi].imag, c='k')
                    # plt.scatter([sec[vi][ci].real], [sec[vi][ci].imag], s=100*p[ci], c='r', facecolor="none", edgecolor='r', alpha=1)
                    # plt.savefig('vis_scatter_%d.png' % ti)
                    # plt.close()

                    # plt.figure()
                    # plt.plot(sec[vi].real, 'bo')
                    # plt.plot(sec[vi].imag, 'go')
                    # plt.plot(np.abs(sec[vi]), 'ro')
                    # plt.axvline(ci, color='k')
                    # plt.savefig('vis_%d.png' % ti)
                    # plt.close()

                    # print ti, p[ci]

                    # # import pdb; pdb.set_trace()