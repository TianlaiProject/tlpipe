"""Map-making.

Inheritance diagram
-------------------

.. inheritance-diagram:: MapMaking
   :parts: 2

"""

from . import timestream_task
from tlpipe.container.timestream import Timestream

from caput import mpiutil
from cora.util import hputil
from tlpipe.utils.path_util import output_path
from tlpipe.map.drift.pipeline import timestream


class MapMaking(timestream_task.TimestreamTask):
    """Map-making.

    This task calls the submodule :mod:`~tlpipe.map.drift` which uses the m-mode
    formalism method to do the map-making.

    """

    params_init = {
                    'ts_dir': 'map/ts',
                    'ts_name': 'ts',
                    'simulate': False,
                    'input_maps': [],
                    'prior_map': None, # or 'prior.hdf5'
                    'add_noise': True,
                    'dirty_map': False,
                    'nbin': None, # use this if multi-freq synthesize
                    'method': 'svd', # or tk
                    'normalize': True, # only used for dirty map-making
                    'threshold': 1.0e3, # only used for dirty map-making
                    'epsilon': 0.0001, # regularization parameter for tk
                    'correct_order': 1, # tk deconv correction order
                    'save_alm': True, # save also alm
                    'tk_deconv': False, # apply tk deconvolution
                    'loop_factor': 0.1, # loop factor
                    'n_iter': 100, # number of iteration
                  }

    prefix = 'mm_'

    def process(self, tstream):

        simulate = self.params['simulate']
        input_maps = self.params['input_maps']
        prior_map = self.params['prior_map']
        add_noise = self.params['add_noise']
        dirty_map = self.params['dirty_map']
        nbin = self.params['nbin']
        method = self.params['method']
        normalize = self.params['normalize']
        threshold = self.params['threshold']
        eps = self.params['epsilon']
        correct_order = self.params['correct_order']
        save_alm = self.params['save_alm']
        tk_deconv = self.params['tk_deconv']
        loop_factor = self.params['loop_factor']
        n_iter = self.params['n_iter']

        bt = tstream.beamtransfer

        tel = bt.telescope
        tel._lmax = None
        tel._mmax = None
        nside = hputil.nside_for_lmax(tel.lmax, accuracy_boost=tel.accuracy_boost)
        tel._init_trans(nside)

        bt.generate()

        if dirty_map:
            tstream.mapmake_full(nside, 'map_full_dirty.hdf5', nbin, dirty=True, method=method, normalize=normalize, threshold=threshold)
        else:
            tstream.mapmake_full(nside, 'map_full.hdf5', nbin, dirty=False, method=method, normalize=normalize, threshold=threshold, eps=eps, correct_order=correct_order, prior_map_file=prior_map, save_alm=save_alm, tk_deconv=tk_deconv, loop_factor=loop_factor, n_iter=n_iter)

        return tstream


    def read_process_write(self, tstream):
        """Overwrite the method of superclass."""

        if isinstance(tstream, timestream.Timestream):
            return self.process(tstream)
        else:
            ts_dir = output_path(self.params['ts_dir'])
            ts_name = self.params['ts_name']
            if mpiutil.rank0:
                print('Try to load tstream from %s/%s' % (ts_dir, ts_name))
            tstream = timestream.Timestream.load(ts_dir, ts_name)
            return self.process(tstream)