"""Foreground clean by KL transform.

Inheritance diagram
-------------------

.. inheritance-diagram:: KL
   :parts: 2

"""

from . import timestream_task
from tlpipe.container.timestream import Timestream

from caput import mpiutil
from cora.util import hputil
from tlpipe.utils.path_util import output_path
from tlpipe.map.drift.pipeline import timestream
from tlpipe.map.drift.core import kltransform, doublekl


class KL(timestream_task.TimestreamTask):
    """Foreground clean by KL transform."""

    params_init = {
                    'ts_dir': 'map/ts',
                    'ts_name': 'ts',
                    'kl_name': 'kl', # or 'double_kl'
                    'subset': True, # If True, throw away modes below a S/N `threshold`
                    'inverse': False, # If True construct and cache inverse transformation
                    'threshold': 0.1, # S/N threshold to cut modes at
                    'foreground_threshold': 100.0, # Ratio of S/F power below which we throw away modes as being foreground contaminated
                    'foreground_regulariser': 1e-14, # The regularisation constant for the foregrounds. Adds in a diagonal of size reg * cf.max()
                    'use_thermal': True, # Whether to use instrumental noise
                    'use_foregrounds': True, # Whether to use foregrounds
                    'use_polarised': False, # Whether to use polarization
                    'pol_length': 1, # number of polarization to use
                  }

    prefix = 'kl_'

    def process(self, tstream):

        kl_name = self.params['kl_name']
        subset = self.params['subset']
        inverse = self.params['inverse']
        threshold = self.params['threshold']
        foreground_threshold = self.params['foreground_threshold']
        foreground_regulariser = self.params['foreground_regulariser']
        use_thermal = self.params['use_thermal']
        use_foregrounds = self.params['use_foregrounds']
        use_polarised = self.params['use_polarised']
        pol_length = self.params['pol_length']

        bt = tstream.beamtransfer
        tel = bt.telescope
        tel._lmax = None
        tel._mmax = None

        if kl_name == 'kl':
            kl = kltransform.KLTransform(bt, subdir=None, subset=subset, inverse=inverse, threshold=threshold, foreground_regulariser=foreground_regulariser, use_thermal=use_thermal, use_foregrounds=use_foregrounds, use_polarised=use_polarised, pol_length=pol_length)
        elif kl_name == 'double_kl':
            kl = doublekl.DoubleKL(bt, subdir=None, subset=subset, inverse=inverse, threshold=threshold, foreground_regulariser=foreground_regulariser, use_thermal=use_thermal, use_foregrounds=use_foregrounds, use_polarised=use_polarised, pol_length=pol_length, foreground_threshold=foreground_threshold)
        else:
            raise ValueError(f'Unknown kl_name: {kl_name}')

        tstream.beamtransfer.skip_svd = False
        tstream.beamtransfer.generate() # ensure to generate svd files before kl
        tstream.generate_mmodes_svd()

        tstream.set_kltransform(kl_name, kl)
        tstream.generate_mmodes_kl()

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