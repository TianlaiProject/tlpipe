"""Night time mean subtract for the visibilities.

Inheritance diagram
-------------------

.. inheritance-diagram:: Subtract
   :parts: 2

"""

import numpy as np
import h5py
from . import timestream_task
from tlpipe.container.raw_timestream import RawTimestream
from tlpipe.container.timestream import Timestream
from tlpipe.utils.path_util import output_path
from caput import mpiutil
from caput import mpiarray


class Subtract(timestream_task.TimestreamTask):
    """Night time mean subtract for the visibilities.

    """

    params_init = {
                    'time_range': [21.5, 5.5], # [t1, t2], local hour, use the mean of t1 < t < t2 if t1 < t2 or {t1 < t < 24.0 and 0.0 < t < t2} if t1 > t2
                    'save_night_mean': False,
                    'night_mean_file': 'night_mean/mean.hdf5'
    }

    prefix = 'su_'

    def process(self, ts):

        save_night_mean = self.params['save_night_mean']
        night_mean_file = self.params['night_mean_file']
        tag_output_iter = self.params['tag_output_iter']

        ts.redistribute('baseline')

        if isinstance(ts, RawTimestream):
            func = ts.bl_data_operate

            night_mean = mpiarray.MPIArray.wrap(np.zeros_like(ts.local_vis[0]), axis=1, comm=ts.comm)
            ts.create_freq_and_bl_ordered_dataset('night_mean', night_mean, axis_order=(1, 2))
            # ts.create_freq_and_bl_ordered_dataset('night_mean', night_mean, axis_order=None)
        elif isinstance(ts, Timestream):
            func = ts.pol_and_bl_data_operate

            night_mean = mpiarray.MPIArray.wrap(np.zeros_like(ts.local_vis[0]), axis=2, comm=ts.comm)
            ts.create_freq_pol_and_bl_ordered_dataset('night_mean', night_mean, axis_order=(1, 2, 3))
            # ts.create_freq_pol_and_bl_ordered_dataset('night_mean', night_mean, axis_order=None)

        func(self.operate, full_data=True)

        if save_night_mean:
            # gather bl_order to rank0
            bl_order = mpiutil.gather_array(ts['blorder'].local_data, axis=0, root=0, comm=ts.comm)

            if isinstance(ts, RawTimestream):
                night_mean = mpiutil.gather_array(ts['night_mean'].local_data, axis=1, root=0, comm=ts.comm)
            elif isinstance(ts, Timestream):
                night_mean = mpiutil.gather_array(ts['night_mean'].local_data, axis=2, root=0, comm=ts.comm)

            if tag_output_iter:
                night_mean_file = output_path(night_mean_file, iteration=self.iteration)
            else:
                night_mean_file = output_path(night_mean_file)
            if mpiutil.rank0:
                with h5py.File(night_mean_file, 'w') as f:
                    f.create_dataset('night_mean', data=night_mean)
                    if isinstance(ts, RawTimestream):
                        f['night_mean'].attrs['dims'] = '(freq, bl)'
                    elif isinstance(ts, Timestream):
                        f['night_mean'].attrs['dims'] = '(freq, pol, bl)'
                    f['night_mean'].attrs['freq'] = ts.freq
                    if isinstance(ts, Timestream):
                        f['night_mean'].attrs['pol'] = ts.pol
                    f['night_mean'].attrs['bl_order'] = '/bl_order'
                    f.create_dataset('bl_order', data=bl_order)

        return super(Subtract, self).process(ts)

    def operate(self, vis, vis_mask, li, gi, bl, ts, **kwargs):
        """Function that does the actual operation."""

        t1, t2 = self.params['time_range']

        local_hour = ts['local_hour'].local_data

        if t1 <= t2:
            tis = np.where(np.logical_and(local_hour>=t1, local_hour<=t2))[0]
        else:
            tis1 = np.where(np.logical_and(local_hour>=t1, local_hour<=24.0))[0]
            tis2 = np.where(np.logical_and(local_hour>=0.0, local_hour<=t2))[0]
            tis = np.concatenate([tis1, tis2])

        mean = np.ma.array(vis[tis], mask=vis_mask[tis]).mean(axis=0)

        if isinstance(ts, RawTimestream):
            lbi = li # local bl index
            ts['night_mean'].local_data[:, lbi] = mean
        elif isinstance(ts, Timestream):
            lpi, lbi = li # local pol and bl index
            ts['night_mean'].local_data[:, lpi, lbi] = mean

        vis -= mean[np.newaxis, :]
