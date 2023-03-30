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

        via_memmap = self.params['via_memmap']
        save_night_mean = self.params['save_night_mean']
        night_mean_file = self.params['night_mean_file']
        tag_output_iter = self.params['tag_output_iter']

        ts.redistribute('time', via_memmap=via_memmap)

        t1, t2 = self.params['time_range']

        local_hour = ts['local_hour'].local_data
        if 'ns_on' in ts.keys():
            ns_on = ts['ns_on'].local_data
        else:
            ns_on = np.zeros_like(local_hour, dtype=bool)

        if t1 <= t2:
            tis = np.where(np.logical_and(local_hour>=t1, local_hour<=t2, ns_on==False))[0]
        else:
            tis1 = np.where(np.logical_and(local_hour>=t1, local_hour<=24.0, ns_on==False))[0]
            tis2 = np.where(np.logical_and(local_hour>=0.0, local_hour<=t2, ns_on==False))[0]
            tis = np.concatenate([tis1, tis2])

        # may use too much memory
        # vis_sum = np.ma.sum(np.ma.array(ts.local_vis[tis], mask=ts.local_vis_mask[tis]), axis=0, keepdims=True).filled(0)
        # vis_cnt = np.logical_not(ts.local_vis_mask[tis]).astype(int).sum(axis=0, keepdims=True)

        # use the following iteration to save memmory usage
        vis_sum = np.zeros((1,)+ts.local_vis.shape[1:], dtype=ts.local_vis.dtype)
        vis_cnt = np.zeros_like(vis_sum, dtype='i4')
        for ti in tis:
            vis_ti = np.ma.array(ts.local_vis[ti], mask=ts.local_vis_mask[ti])
            vis_sum += vis_ti.filled(0)
            vis_cnt += (~vis_ti.mask).astype('i4')

        vis_sum = mpiutil.gather_array(vis_sum, axis=0, root=0, comm=ts.comm)
        vis_cnt = mpiutil.gather_array(vis_cnt, axis=0, root=0, comm=ts.comm)

        if mpiutil.rank0:
            vis_sum = vis_sum.sum(axis=0, keepdims=False)
            vis_cnt = vis_cnt.sum(axis=0, keepdims=False)
            night_mean = np.where(vis_cnt==0, 0, vis_sum/vis_cnt)

            if save_night_mean:
                if tag_output_iter:
                    night_mean_file = output_path(night_mean_file, iteration=self.iteration)
                else:
                    night_mean_file = output_path(night_mean_file)
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
                    f.create_dataset('bl_order', data=ts.bl)
        else:
            night_mean = None

        night_mean = mpiutil.bcast(night_mean, comm=ts.comm)

        ts.local_vis[:] -= night_mean[np.newaxis, ...]

        return super(Subtract, self).process(ts)
