"""Re-order data to have longitude from 0 to 2pi."""

import numpy as np
import tod_task


class ReOrder(tod_task.SingleTimestream):
    """Re-order data to have longitude from 0 to 2pi."""

    prefix = 'ro_'

    def process(self, ts):

        ts.redistribute('baseline')

        # re-order data to have longitude from 0 to 2pi
        phi = ts['ra_dec'][:, 0]

        # find phi = 0 ind
        ind0 = np.where(np.diff([phi[-1]] + phi.tolist()) < -1.9 * np.pi)[0][0]

        # re-order all main_axes_ordered_datasets
        for name in ts.main_time_ordered_datasets.keys():
            if name in ts.iterkeys():
                dset = ts[name]
                time_axis = ts.main_time_ordered_datasets[name].index(0)
                sel1 = [slice(0, None)] * (time_axis + 1)
                sel2 = sel1[:]
                sel1[time_axis] = slice(ind0, None)
                sel2[time_axis] = slice(0, ind0)
                dset.local_data[:] = np.concatenate([ dset.local_data[sel1], dset.local_data[sel2] ], axis=time_axis)

        ts.add_history(self.history)

        return ts
