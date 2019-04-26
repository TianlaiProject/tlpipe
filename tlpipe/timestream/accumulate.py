"""Accumulate data.

Inheritance diagram
-------------------

.. inheritance-diagram:: Accum
   :parts: 2

"""

import os
import numpy as np
import h5py
import timestream_task
from tlpipe.container.timestream import Timestream
from tlpipe.utils.path_util import input_path, output_path
from caput import mpiutil
from caput import mpiarray


class Accum(timestream_task.TimestreamTask):
    """Accumulate data.

    This task accumulates data abserved in different (sidereal) days
    and records the weight (i.e., the number of valid or un-masked data)
    of each data point.

    .. note::
        This should be done after the data has been calibrated and re-ordered
        to a same LST (or RA).

    """

    params_init = {
                    'check': True, # check data alignment before accumulate
                    'load_data': None, # load data from disk first and accumulate to it if not None but a list of files
                    'cache_to_file': False, # cache the data to file instead of in memory
                    'cache_file_name': 'cache/accumulate.hdf5', # name of the cache file
                  }

    prefix = 'ac_'

    def setup(self):
        self.data = None

    def process(self, ts):

        assert isinstance(ts, Timestream), '%s only works for Timestream object' % self.__class__.__name__

        check = self.params['check']
        load_data = self.params['load_data']

        # cache to file
        if self.params['cache_to_file']:
            cache_file_name = input_path(self.params['cache_file_name'])
            if not os.path.isfile(cache_file_name):
                # write ts to cache_file_name
                cache_file_name = output_path(self.params['cache_file_name'], mkdir=True)
                ts.apply_mask(fill_val=0) # apply mask, fill 0 to masked values
                # create weight dataset
                weight = np.logical_not(ts.local_vis_mask).astype(np.int16) # use int16 to save memory
                weight = mpiarray.MPIArray.wrap(weight, axis=ts.main_data_dist_axis)
                axis_order = ts.main_axes_ordered_datasets[ts.main_data_name]
                ts.create_main_axis_ordered_dataset(axis_order, 'weight', weight, axis_order)
                ts.attrs['ndays'] = 1 # record number of days accumulated

                ts.to_files(cache_file_name)

                # empty ts to release memory
                ts.empty()
            else:
                # accumulate ts to cache_file_name
                ts.apply_mask(fill_val=0) # apply mask, fill 0 to masked values
                for rk in range(ts.nproc):
                    if ts.rank == rk:
                        with h5py.File(cache_file_name, 'r+') as f:
                            # may need some check in future...
                            if np.prod(ts['vis'].local_shape) > 0: # no need to write if no local data
                                slc = []
                                for st, ln in zip(ts['vis'].local_offset, ts['vis'].local_shape):
                                    slc.append(slice(st, st+ln))
                                f['vis'][tuple(slc)] += ts.local_vis # accumulate vis
                                if 'weight' in ts.iterkeys():
                                    f['weight'][tuple(slc)] += ts['weight'].local_data
                                else:
                                    f['weight'][tuple(slc)] += np.logical_not(ts.local_vis_mask).astype(np.int16) # update weight
                                f['vis_mask'][tuple(slc)] = np.where(f['weight'][tuple(slc)] != 0, False, True) # update mask
                            if rk == 0:
                                f.attrs['ndays'] += 1
                    mpiutil.barrier(ts.comm)

                # empty ts to release memory
                ts.empty()

            return cache_file_name

        # accumulate in memory
        # ts.redistribute('baseline')

        # load data from disk if load_data is set
        if (self.data is None) and (not load_data is None):
            self.data = Timestream(load_data, mode='r', start=0, stop=None, dist_axis=ts.main_data_dist_axis, use_hints=True, comm=ts.comm)
            self.data.load_all()

        if self.data is None:
            self.data = ts
            # self.data = ts.copy()
            self.data.apply_mask(fill_val=0) # apply mask, fill 0 to masked values
            # create weight dataset
            weight = np.logical_not(self.data.local_vis_mask).astype(np.int16) # use int16 to save memory
            weight = mpiarray.MPIArray.wrap(weight, axis=self.data.main_data_dist_axis)
            axis_order = self.data.main_axes_ordered_datasets[self.data.main_data_name]
            self.data.create_main_axis_ordered_dataset(axis_order, 'weight', weight, axis_order)
            self.data.attrs['ndays'] = 1 # record number of days accumulated
        else:
            # make they are distributed along the same axis
            ts.redistribute(self.data.main_data_dist_axis)
            # check for ra, dec
            ra_self = self.data['ra_dec'].local_data[:, 0]
            if mpiutil.rank0 and ra_self[0] > ra_self[1]:
                ra_self[0] -= 2*np.pi
            ra_ts = ts['ra_dec'].local_data[:, 0]
            if mpiutil.rank0 and ra_ts[0] > ra_ts[1]:
                ra_ts[0] -= 2*np.pi

            delta = 2*np.pi/self.data['ra_dec'].shape[0]
            if not np.allclose(ra_self, ra_ts, rtol=0, atol=delta):
                print 'RA not align within %f for rank %d, max gap %f' % (delta, mpiutil.rank, np.abs(ra_ts - ra_self).max())
            # assert np.allclose(self.data['ra_dec'].local_data[:, 0], ts['ra_dec'].local_data[:, 0], rtol=0, atol=2*np.pi/self.data['ra_dec'].shape[0]), 'Can not accumulate data, RA not align.'
            # assert np.allclose(self.data['ra_dec'].local_data[:, 1], ts['ra_dec'].local_data[:, 1]), 'Can not accumulate data with different DEC.'
            # other checks if required
            if check:
                assert self.data.attrs['telescope'] == ts.attrs['telescope'], 'Data are observed by different telescopes %s and %s' % (self.data.attrs['telescope'], ts.attrs['telescope'])
                assert np.allclose(self.data.local_freq, ts.local_freq), 'freq not align'
                assert len(self.data.local_pol) == len(ts.local_pol) and (self.data.local_pol == ts.local_pol).all(), 'pol not align'
                assert np.allclose(self.data.local_bl, ts.local_bl), 'bl not align'

            ts.apply_mask(fill_val=0) # apply mask, fill 0 to masked values
            self.data.local_vis[:] += ts.local_vis # accumulate vis
            if 'weight' in ts.iterkeys():
                self.data['weight'].local_data[:] += ts['weight'].local_data
            else:
                self.data['weight'].local_data[:] += np.logical_not(ts.local_vis_mask).astype(np.int16) # update weight
            self.data.local_vis_mask[:] = np.where(self.data['weight'].local_data != 0, False, True) # update mask
            self.data.attrs['ndays'] += 1


        return super(Accum, self).process(self.data)
