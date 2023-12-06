"""Write the time stream to a CASA MeasurementSet (MS) file.

Inheritance diagram
-------------------

.. inheritance-diagram:: WriteMS
   :parts: 2

"""

import numpy as np
from pyuvdata import UVData
from pyuvdata.utils import ECEF_from_ENU
from astropy.coordinates import EarthLocation
# import ephem
# import h5py
# import aipy as a
from . import timestream_task
from tlpipe.container.timestream import Timestream

from caput import mpiutil
# # from caput import mpiarray
from tlpipe.utils.path_util import output_path
# from tlpipe.utils import progress
# from tlpipe.utils import rpca_decomp
# from tlpipe.cal import calibrators
# import tlpipe.plot
# import matplotlib.pyplot as plt


class WriteMS(timestream_task.TimestreamTask):
    """Write the time stream to a CASA MeasurementSet (MS) file."""


    params_init = {
                    'ms_name': 'NP.ms',
                  }

    prefix = 'wm_'

    def process(self, ts):

        assert isinstance(ts, Timestream), '%s only works for Timestream object' % self.__class__.__name__
        assert mpiutil.size == 1, 'Task %s only works for single process run'  % self.__class__.__name__

        tag_output_iter = self.params['tag_output_iter']
        ms_name = self.params['ms_name']

        ntime, nfreq, npol, nbl = ts.local_vis.shape

        # print(ts.vis.shape)
        # return super(WriteMS, self).process(ts)

        # if 'unit' in ts.vis.attrs:
        #     vis_units = ts.vis.attrs['unit']
        # else:
        #     vis_units = 'uncalib'
        vis_units = 'uncalib'

        telescope_location = EarthLocation.from_geodetic(ts.attrs['sitelon'], ts.attrs['sitelat'], ts.attrs['siteelev'])

        uvd = UVData.new(
            freq_array = 1.0e6 * ts.local_freq, # Hz
            polarization_array = ["xx", "yy", "xy", "yx"],
            antenna_positions = ECEF_from_ENU(ts['feedpos'][:], np.radians(ts.attrs['sitelat']), np.radians(ts.attrs['sitelon']), ts.attrs['siteelev']) - np.array(list(telescope_location.value)),
            telescope_location = telescope_location,
            telescope_name = ts.attrs['telescope'],
            times = np.array(ts.local_time),
            antpairs = None,
            do_blt_outer = True,
            integration_time = ts.attrs['inttime'],
            channel_width = 1.0e6 * ts.attrs['freqstep'], # Hz
            antenna_names = None,
            antenna_numbers = ts['feedno'][:],
            blts_are_rectangular = True,
            data_array = np.array(ts.local_vis).transpose(0, 3, 1, 2).reshape(-1, nfreq, npol),
            # data_array = np.array(ts.local_vis).conj().transpose(0, 3, 1, 2).reshape(-1, nfreq, npol),
            flag_array = np.array(ts.local_vis_mask).transpose(0, 3, 1, 2).reshape(-1, nfreq, npol),
            nsample_array = np.ones((ntime*nbl, nfreq, npol)),
            flex_spw_id_array = None,
            history = '',
            instrument = '',
            vis_units = vis_units,
            antname_format = '{0:03d}',
            empty = False,
            time_axis_faster_than_bls = False,
            # phase_center_catalog = {0: {'cat_name': 'NP', 'cat_type': 'sidereal', 'cat_lon': 0.0, "cat_lat": np.pi/2, 'cat_frame': 'icrs'}},
            phase_center_catalog = {0: {'cat_name': 'NP', 'cat_type': 'sidereal', 'cat_lon': 0.0, "cat_lat": np.pi/2, 'cat_frame': 'fk5', 'cat_epoch': 'J2000.0'}},
            phase_center_id_array = None,
            x_orientation = 'east',
            astrometry_library = 'erfa',
        )

        if tag_output_iter:
            ms_name = output_path(ms_name, iteration=self.iteration)
        else:
            ms_name = output_path(ms_name)
        uvd.write_ms(ms_name)


        return super(WriteMS, self).process(ts)