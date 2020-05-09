# -*- coding: utf-8 -*-

import numpy as np
from scipy import special
import warnings


class LocalOutlierProbability(object):
    """Local Outlier Probabilities (LoOP).

    Based on the work of Kriegel, Kröger, Schubert and Zimek (2009),
    LoOP: Local Outlier Probabilities.
    ----------

    References
    ----------
    .. [1] Breunig M., Kriegel H.-P., Ng R., Sander, J. LOF: Identifying
           Density-based Local Outliers. ACM SIGMOD
           International Conference on Management of Data (2000).
    .. [2] Kriegel H.-P., Kröger P., Schubert E., Zimek A. LoOP: Local Outlier
           Probabilities. 18th ACM conference on
           Information and knowledge management, CIKM (2009).
    .. [3] Goldstein M., Uchida S. A Comparative Evaluation of Unsupervised
           Anomaly Detection Algorithms for Multivariate Data. PLoS ONE 11(4):
           e0152173 (2016).
    .. [4] Hamlet C., Straub J., Russell M., Kerlin S. An incremental and
           approximate local outlier probability algorithm for intrusion
           detection and its evaluation. Journal of Cyber Security Technology
           (2016).
    """

    def __init__(self, n_neighbors=20, lmbda=3):
        self.n_neighbors = n_neighbors
        self.lmbda = lmbda
        self._fit = False

    def _compute_loop(self):
        d2_matrix = self._d2_matrix[np.ix_(self._data_inds, self._data_inds)]
        idx_matrix = np.argsort(d2_matrix, axis=1) # ascending order
        nbr_d2_matrix = np.zeros((self._n_data, self._n_neighbors), dtype=self._d2_matrix.dtype)
        for i in range(self._n_data):
            nbr_d2_matrix[i] = d2_matrix[i, idx_matrix[i, 1:self._n_neighbors+1]]
        sigma = np.sqrt(np.sum(nbr_d2_matrix, axis=1) / self._n_neighbors)
        pdist = self.lmbda * sigma
        plof = np.zeros_like(pdist)
        for i in range(self._n_data):
            plof[i] = pdist[i] / np.mean(pdist[idx_matrix[i, 1:self._n_neighbors+1]]) - 1.0
        nplof = self.lmbda * np.sqrt(np.mean(plof**2))
        p = special.erf(plof / (np.sqrt(2.0) * nplof))
        p[p<0] = 0.0

        return p


    @property
    def data(self):
        return self._data[self._data_inds]


    def fast_fit(self, data):
        n_data = data.shape[0]

        if n_data - 1 < self.n_neighbors:
            warnings.warn('Number of data neighbors (%d) is less than n_neighbors (%d), n_neighbors will be set to %d for estimation' % (n_data - 1, self.n_neighbors, n_data - 1))
            self._n_neighbors = n_data - 1
        else:
            self._n_neighbors = self.n_neighbors

        diff_matrix = data[np.newaxis, :, :] - data[:, np.newaxis, :]
        d2_matrix = diff_matrix[:, :, 0]**2 + diff_matrix[:, :, 1]**2
        idx_matrix = np.argsort(d2_matrix, axis=1) # ascending order
        nbr_d2_matrix = np.zeros((n_data, self._n_neighbors), dtype=d2_matrix.dtype)
        for i in range(n_data):
            nbr_d2_matrix[i] = d2_matrix[i, idx_matrix[i, 1:self._n_neighbors+1]]
        sigma = np.sqrt(np.sum(nbr_d2_matrix, axis=1) / self._n_neighbors)
        pdist = self.lmbda * sigma
        plof = np.zeros_like(pdist)
        for i in range(n_data):
            plof[i] = pdist[i] / np.mean(pdist[idx_matrix[i, 1:self._n_neighbors+1]]) - 1.0
        nplof = self.lmbda * np.sqrt(np.mean(plof**2))
        p = special.erf(plof / (np.sqrt(2.0) * nplof))
        p[p<0] = 0.0

        self.local_outlier_probabilities = p

        return self

    def fit(self, data):
        n_data = data.shape[0]

        if n_data - 1 < self.n_neighbors:
            warnings.warn('Number of data neighbors (%d) is less than n_neighbors (%d), n_neighbors will be set to %d for estimation' % (n_data - 1, self.n_neighbors, n_data - 1))
            self._n_neighbors = n_data - 1
        else:
            self._n_neighbors = self.n_neighbors

        diff_matrix = data[np.newaxis, :, :] - data[:, np.newaxis, :]
        d2_matrix = diff_matrix[:, :, 0]**2 + diff_matrix[:, :, 1]**2

        self._data = data.copy() # does not change the input data
        self._data_inds = range(n_data)
        self._slots = np.zeros(n_data, dtype=np.int) # 0 for taken, -1 for empty
        self._n_data = n_data
        self._d2_matrix = d2_matrix

        self.local_outlier_probabilities = self._compute_loop()
        self._fit = True

        return self

    def refit(self, new_data, pop_inds=[]):
        if not self._fit:
            raise RuntimeError('Must call fit() before refit()')

        pis = [ self._data_inds.pop(pi) for pi in sorted(pop_inds, reverse=True) ]
        pis.sort()

        n_new_data = new_data.shape[0]
        i = -1 # for pop_inds = []
        for i, pi in enumerate(pis):

            if i >= n_new_data:
                self._slots[pi] = -1
                self._n_data -= 1
                continue

            # insert new data to the pop_ind position
            if self._slots[pi] == -1:
                raise RuntimeError('Something wrong happend')
            self._data[pi] = new_data[i]
            diff = self._data[self._data_inds] - new_data[i][np.newaxis, :]
            d2 = diff[:, 0]**2 + diff[:, 1]**2
            self._d2_matrix[pi, self._data_inds] = d2
            self._d2_matrix[self._data_inds, pi] = d2
            # self._slots[pi] = 0 # should be 0 in pi
            self._data_inds.append(pi)
        else:
            for j in range(i+1, n_new_data):
                self._n_data += 1

                if self._n_data > self._data.shape[0]:
                    # double the size of array to hold more data
                    tmp_data = np.zeros((2*self._n_data, self._data.shape[1]), dtype=self._data.dtype)
                    tmp_data[:self._data.shape[0]] = self._data
                    self._data = tmp_data
                    tmp_d2_matrix = np.zeros((2*self._n_data, 2*self._n_data), dtype=self._d2_matrix.dtype)
                    tmp_d2_matrix[:self._d2_matrix.shape[0], :self._d2_matrix.shape[1]] = self._d2_matrix
                    self._d2_matrix = tmp_d2_matrix
                    tmp_slots = np.full((2*self._n_data,), -1, dtype=np.int)
                    tmp_slots[:self._slots.shape[0]] = self._slots
                    self._slots = tmp_slots

                # find the first empty position
                ei = np.where(self._slots == -1)[0][0]
                # insert new data in this empty position
                self._data[ei] = new_data[j]

                self._slots[ei] = 0
                self._data_inds.append(ei)

                diff = self._data[self._data_inds] - new_data[j][np.newaxis, :]
                d2 = diff[:, 0]**2 + diff[:, 1]**2
                self._d2_matrix[ei, self._data_inds] = d2
                self._d2_matrix[self._data_inds, ei] = d2

        if self._n_data - 1 < self.n_neighbors:
            warnings.warn('Number of data neighbors (%d) is less than n_neighbors (%d), n_neighbors will be set to %d for estimation' % (self._n_data - 1, self.n_neighbors, self._n_data - 1))
            self._n_neighbors = self._n_data - 1
        else:
            self._n_neighbors = self.n_neighbors

        self.local_outlier_probabilities = self._compute_loop()

        return self