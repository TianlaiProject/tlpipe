"""Accumulate data.

Inheritance diagram
-------------------

.. inheritance-diagram:: Accum
   :parts: 2

"""

import numpy as np
import tod_task
from timestream import Timestream
from caput import mpiutil
from caput import mpiarray


class Accum(tod_task.TaskTimestream):
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
                  }

    prefix = 'ac_'

    def setup(self):
        self.data = None

    def process(self, ts):

        assert isinstance(ts, Timestream), '%s only works for Timestream object' % self.__class__.__name__

        check = self.params['check']

        # ts.redistribute('baseline')

        if self.data is None:
            self.data = ts
            # self.data = ts.copy()
            self.data.apply_mask(fill_val=0) # apply mask, fill 0 to masked values
            # create weight dataset
            weight = np.logical_not(self.data.local_vis_mask).astype(np.int16) # use int16 to save memory
            weight = mpiarray.MPIArray.wrap(weight, axis=self.data.main_data_dist_axis)
            axis_order = self.data.main_axes_ordered_datasets[self.data.main_data_name]
            self.data.create_main_axis_ordered_dataset(axis_order, 'weight', weight, axis_order)
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
            self.data['weight'].local_data[:] += np.logical_not(ts.local_vis_mask).astype(np.int16) # update weight
            self.data.local_vis_mask[:] = np.where(self.data['weight'].local_data != 0, False, True) # update mask


        return super(Accum, self).process(self.data)
