"""Detect noise source signal."""

import numpy as np
import tod_task

from caput import mpiutil
from caput import mpiarray


class Detect(tod_task.SingleRawTimestream):
    """Detect noise source signal."""

    params_init = {
                    'bl': (1, 2), # use this bl
                    'threshold': 1.0, # or yy
                  }

    prefix = 'dt_'

    def process(self, rt):

        bl = self.params['bl']
        threshold = self.params['threshold']

        rt.redistribute(0) # make time the dist axis

        bls = [ set(b) for b in rt.bl ]
        bl_ind = bls.index(set(bl))

        t_mean = np.mean(np.abs(rt.main_data.local_data[:, :, bl_ind]), axis=-1)
        tt_mean = mpiutil.gather_array(t_mean, root=None)

        ns_on = np.where(tt_mean>threshold*tt_mean.mean(), 1, 0) # 1 for noise on
        diff_ns = np.diff(ns_on)
        on_inds = np.where(diff_ns==1)[0]
        first_on = on_inds[0]
        second_on = on_inds[1]
        off_inds = np.where(diff_ns==-1)[0]
        if off_inds[0] > first_on and off_inds[0] < second_on:
            first_off = off_inds[0]
        elif off_inds[1] > first_on and off_inds[1] < second_on:
            first_off = off_inds[1]
        else:
            raise RuntimeError('Some thing wrong happend, could not determine first on/off ind')

        cycle = [True] * (first_off - first_on) + [False] * (second_on - first_off)
        cycle = np.roll(np.array(cycle), first_on+1)
        num_cycle = int(np.ceil(float(len(ns_on)) / len(cycle)))
        ns_on = np.array(cycle.tolist() * num_cycle)[:len(ns_on)]
        ns_on = mpiarray.MPIArray.from_numpy_array(ns_on)

        rt.create_main_time_ordered_dataset('ns_on', ns_on)

        rt.add_history(self.history)

        # rt.info()
        # print rt.main_time_ordered_datasert
        # print rt.time_ordered_datasert

        return rt
