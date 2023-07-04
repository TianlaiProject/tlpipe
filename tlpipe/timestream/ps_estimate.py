"""Power spectrum estimation.

Inheritance diagram
-------------------

.. inheritance-diagram:: Ps
   :parts: 2

"""

from . import timestream_task
from tlpipe.container.timestream import Timestream

from caput import mpiutil
from cora.util import hputil
from tlpipe.utils.path_util import output_path
from tlpipe.map.drift.pipeline import timestream
from tlpipe.map.drift.core import psestimation, psmc, crosspower


class Ps(timestream_task.TimestreamTask):
    """Power spectrum estimation."""

    params_init = {
                    'ts_dir': 'map/ts',
                    'ts_name': 'ts',
                    'ps_name': 'full', # or 'mc', 'mc_alt', 'cross'
                    'subdir': 'ps', # Subdir to save results in
                    'bandtype': 'polar', # or 'cartesian', Which types of bands to use
                    'k_bands': [ {'spacing' : 'linear', 'start' : 0.0, 'stop' : 0.4, 'num' : 20 }], # Array of band boundaries, polar only
                    'num_theta': 1, # Number of theta bands to use (polar only)
                    'kpar_bands': [ {'spacing' : 'linear', 'start' : 0.0, 'stop' : 0.4, 'num' : 20 }], # Array of band boundaries, cartesian only
                    'kperp_bands': [ {'spacing' : 'linear', 'start' : 0.0, 'stop' : 0.4, 'num' : 20 }], # Array of band boundaries, cartesian only
                    'threshold': 0.0, # Threshold for including eigenmodes
                    'unit_bands': True, # If True, bands are sections of the exact powerspectrum (such that the fiducial bin amplitude is 1)
                    'zero_mean': True, # If True (default), then the fiducial parameters have zero mean
                    'nsamples': 500, # The number of samples to draw from each band
                    'nswitch': 0, # The threshold number of eigenmodes above which we switch to Monte-Carlo estimation
                  }

    prefix = 'pe_'

    def process(self, tstream):

        ps_name = self.params['ps_name']
        subdir = self.params['subdir']
        bandtype = self.params['bandtype']
        k_bands = psestimation.range_config(self.params['k_bands'])
        num_theta = self.params['num_theta']
        kpar_bands = psestimation.range_config(self.params['kpar_bands'])
        kperp_bands = psestimation.range_config(self.params['kperp_bands'])
        threshold = self.params['threshold']
        unit_bands = self.params['unit_bands']
        zero_mean = self.params['zero_mean']
        nsamples = self.params['nsamples']
        nswitch = self.params['nswitch']

        kl = tstream.kl

        if ps_name == 'full':
            ps = psestimation.PSExact(kl, subdir, bandtype, k_bands, num_theta, kpar_bands, kperp_bands, threshold, unit_bands, zero_mean)
        elif kl_name == 'mc':
            ps = psmc.PSMonteCarlo(kl, subdir, bandtype, k_bands, num_theta, kpar_bands, kperp_bands, threshold, unit_bands, zero_mean, nsamples)
        elif kl_name == 'mc_alt':
            ps = psmc.PSMonteCarloAlt(kl, subdir, bandtype, k_bands, num_theta, kpar_bands, kperp_bands, threshold, unit_bands, zero_mean, nsamples, nswitch)
        elif ps_name == 'cross':
            ps = crosspower.CrossPower(kl, subdir, bandtype, k_bands, num_theta, kpar_bands, kperp_bands, threshold, unit_bands, zero_mean)
        else:
            raise ValueError(f'Unknown ps_name: {ps_name}')

        # Calculate the total Fisher matrix and bias and save to a file
        ps.generate()

        tstream.set_psestimator(ps_name, ps)
        tstream.powerspectrum()

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