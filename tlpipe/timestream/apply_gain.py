"""Calibrate the visibility by divide the gain.

Inheritance diagram
-------------------

.. inheritance-diagram:: Apply
   :parts: 2

"""

import numpy as np
import h5py
import timestream_task
from tlpipe.container.timestream import Timestream
from tlpipe.utils.path_util import input_path


class Apply(timestream_task.TimestreamTask):
    """Calibrate the visibility by divide the gain.

    .. math:: V_{ij}^{\\text{cal}} = V_{ij} / (g_i g_j^*).

    """


    params_init = {
                    'gain_file': 'gain.hdf5',
                  }

    prefix = 'ag_'

    def process(self, ts):

        assert isinstance(ts, Timestream), '%s only works for Timestream object' % self.__class__.__name__

        gain_file = self.params['gain_file']
        tag_input_iter = self.params['tag_input_iter']
        if tag_input_iter:
            gain_file = input_path(gain_file, self.iteration)

        # read gain from file
        with h5py.File(gain_file, 'r') as f:
            gain = f['gain'][:]
            gain_src = f['gain'].attrs['calibrator']
            gain_freq = f['gain'].attrs['freq']
            gain_pol = f['gain'].attrs['pol']
            gain_feed = f['gain'].attrs['feed']

        ts.redistribute('baseline')

        feedno = ts['feedno'][:].tolist()
        pol = [ ts.pol_dict[p] for p in ts['pol'][:] ] # as string
        gain_pd = {'xx': 0, 'yy': 1,    0: 'xx', 1: 'yy'} # for gain related op
        freq = ts.freq[:]
        nf = len(freq)

        # shold check freq, pol and feed here, omit it now...

        for fi in range(nf):
            for pi in [pol.index('xx'), pol.index('yy')]:
                pi_ = gain_pd[pol[pi]]
                for bi, (fd1, fd2) in enumerate(ts['blorder'].local_data):
                    g1 = gain[fi, pi_, feedno.index(fd1)]
                    g2 = gain[fi, pi_, feedno.index(fd2)]
                    if np.isfinite(g1) and np.isfinite(g2):
                        ts.local_vis[:, fi, pi, bi] /= (g1 * np.conj(g2))
                    else:
                        # mask the un-calibrated vis
                        ts.local_vis_mask[:, fi, pi, bi] = True

        return super(Apply, self).process(ts)