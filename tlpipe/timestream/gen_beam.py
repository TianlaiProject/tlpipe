"""Generate beam transfer matrices.

Inheritance diagram
-------------------

.. inheritance-diagram:: GenBeam
   :parts: 2

"""

import numpy as np
import timestream_task
from tlpipe.container.timestream import Timestream

from caput import mpiutil
from tlpipe.utils.path_util import output_path
from tlpipe.map.drift.core import beamtransfer


class GenBeam(timestream_task.TimestreamTask):
    """Generate beam transfer matrices.

    Beam transfer matrices can be pre-generated for accelerate map-making.

    """

    params_init = {
                    'tsys': 50.0,
                    'accuracy_boost': 1.0,
                    'l_boost': 1.0,
                    'bl_range': [0.0, 1.0e7],
                    'auto_correlations': False,
                    'beam_dir': 'map/bt',
                    'noise_weight': True,
                  }

    prefix = 'gb_'

    def process(self, ts):

        tsys = self.params['tsys']
        accuracy_boost = self.params['accuracy_boost']
        l_boost = self.params['l_boost']
        bl_range = self.params['bl_range']
        auto_correlations = self.params['auto_correlations']
        beam_dir = output_path(self.params['beam_dir'])
        noise_weight = self.params['noise_weight']


        assert isinstance(ts, Timestream), '%s only works for Timestream object' % self.__class__.__name__

        ts.redistribute('baseline')

        lat = ts.attrs['sitelat']
        # lon = ts.attrs['sitelon']
        lon = 0.0
        # lon = np.degrees(ts['ra_dec'][0, 0]) # the first ra
        local_origin = False
        freqs = ts.freq[:] # MHz
        nfreq = freqs.shape[0]
        band_width = ts.attrs['freqstep'] # MHz
        try:
            ndays = ts.attrs['ndays']
        except KeyError:
            ndays = 1
        feeds = ts['feedno'][:]
        bl_order = mpiutil.gather_array(ts.local_bl, axis=0, root=None, comm=ts.comm)
        bls = [ tuple(bl) for bl in bl_order ]
        az, alt = ts['az_alt'][0]
        az = np.degrees(az)
        alt = np.degrees(alt)
        pointing = [az, alt, 0.0]
        feedpos = ts['feedpos'][:]

        if ts.is_dish:
            from tlpipe.map.drift.telescope import tl_dish

            dish_width = ts.attrs['dishdiam']
            tel = tl_dish.TlUnpolarisedDishArray(lat, lon, freqs, band_width, tsys, ndays, accuracy_boost, l_boost, bl_range, auto_correlations, local_origin, dish_width, feedpos, pointing)
        elif ts.is_cylinder:
            from tlpipe.map.drift.telescope import tl_cylinder

            # factor = 1.2 # suppose an illumination efficiency, keep same with that in timestream_common
            factor = 0.79 # for xx
            # factor = 0.88 # for yy
            cyl_width = factor * ts.attrs['cywid']
            tel = tl_cylinder.TlUnpolarisedCylinder(lat, lon, freqs, band_width, tsys, ndays, accuracy_boost, l_boost, bl_range, auto_correlations, local_origin, cyl_width, feedpos)
        else:
            raise RuntimeError('Unknown array type %s' % ts.attrs['telescope'])

        # beamtransfer
        bt = beamtransfer.BeamTransfer(beam_dir, tel, noise_weight, True)
        bt.generate()


        return super(GenBeam, self).process(ts)