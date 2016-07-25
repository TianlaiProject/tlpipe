"""Generate telescope running information."""

import numpy as np
import ephem
from datetime import datetime
import tod_task

from caput import mpiarray
from tlpipe.utils import date_util


class RunInfo(tod_task.SingleRawTimestream):
    """Generate telescope running information."""


    prefix = 'ri_'

    def process(self, rt):

        # rt.redistribute('baseline')

        az_alt = np.zeros((rt['sec1970'].local_data.shape[0], 2), dtype=np.float32) # radians

        if rt.is_dish:
            # antpointing = rt['antpointing'][-1, :, :] # degree
            # pointingtime = rt['pointingtime'][-1, :, :] # degree
            az_alt[:, 0] = 0.0 # az
            az_alt[:, 1] = np.pi/2 # alt
        elif rt.is_cylinder:
            az_alt[:, 0] = np.pi/2 # az
            az_alt[:, 1] = np.pi/2 # alt
        else:
            raise RuntimeError('Unknown antenna type %s' % ts.attrs['telescope'])

        # array
        aa = rt.array

        # aa.date = # the date for which the position should be computed (current time)
        # aa.epoch = ephem.J2000 # epoch for which coordinates should be generated (year 2000)
        # aa.temp = 15.0 # temperature in degrees centigrade
        # aa.pressure = 1010.0 # mB, atmospheric pressure in milibars (1010 mB)
        # aa.horizon = 0.0 # degree, at what angle you consider an object to be rising or setting

        ra_dec = np.zeros_like(az_alt) # radians
        for ti in range(az_alt.shape[0]):
            az, alt = az_alt[ti]
            az, alt = ephem.degrees(az), ephem.degrees(alt)
            aa.set_jultime(rt['jul_date'].local_data[ti])
            ra_dec[ti] = aa.radec_of(az, alt) # in radians, a point in the sky above the observer

        if rt.main_data_dist_axis == 0:
            az_alt = mpiarray.MPIArray.wrap(az_alt, axis=0)
            ra_dec = mpiarray.MPIArray.wrap(ra_dec, axis=0)
        # if time is just the distributed axis, create distributed datasets
        rt.create_main_time_ordered_dataset('az_alt', data=az_alt)
        rt['az_alt'].attrs['unit'] = 'radian'
        rt.create_main_time_ordered_dataset('ra_dec', data=ra_dec)
        rt['ra_dec'].attrs['unit'] = 'radian'


        # ra_dec = ra_dec.to_numpy_array(root=None)
        # print np.degrees(ra_dec[0, 1])

        # import matplotlib
        # matplotlib.use('Agg')
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(ra_dec)
        # plt.savefig('ra_dec.png')


        rt.add_history(self.history)

        # rt.info()

        return rt
