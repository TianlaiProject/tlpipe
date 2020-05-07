"""RFI flagging by using Local Outlier Factor (LOF).

Inheritance diagram
-------------------

.. inheritance-diagram:: Flag
   :parts: 2

"""

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import timestream_task
from tlpipe.container.raw_timestream import RawTimestream
from tlpipe.container.timestream import Timestream


class Flag(timestream_task.TimestreamTask):
    """RFI flagging by using Local Outlier Factor (LOF).

    LOF is an unsupervised outlier detection algorithm.

    The anomaly score of each sample is called Local Outlier Factor. It
    measures the local deviation of density of a given sample with respect
    to its neighbors. It is local in that the anomaly score depends on how
    isolated the object is with respect to the surrounding neighborhood.
    More precisely, locality is given by k-nearest neighbors, whose distance
    is used to estimate the local density. By comparing the local density of
    a sample to the local densities of its neighbors, one can identify
    samples that have a substantially lower density than their neighbors.
    These are considered outliers.

    We use the algorithm implemented in sklearn.neighbors.LocalOutlierFactor.

    """

    params_init = {
                    'n_neighbors': 20,
                    'contamination': 'auto', # 'auto' or float, the proportion of outliers in the data set
                    'time_window': 75,
                    'freq_window': 5,
                    'score_threshold': 2.0, # considered to be outlier when higher than this val
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
        contamination = self.params['contamination']
        time_window = self.params['time_window']
        freq_window = self.params['freq_window']
        score_threshold = self.params['score_threshold']

        # clf = LocalOutlierFactor(n_neighbors=n_neighbors, metric='l2', contamination=contamination)
        # not support contamination = 'auto' for lower version of sklearn
        clf = LocalOutlierFactor(n_neighbors=n_neighbors, metric='l2')

        nt, nf = vis.shape
        for ti in range(nt):
            for fi in range(nf):
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
                X = np.vstack([sec[vi].real, sec[vi].imag]).T
                clf.fit_predict(X)
                X_scores = np.abs(clf.negative_outlier_factor_)
                if X_scores[np.where(vi == (ti - lti)*nf + (fi - lfi))[0]] > score_threshold:
                    # is an outlier
                    vis_mask[ti, fi] = True
