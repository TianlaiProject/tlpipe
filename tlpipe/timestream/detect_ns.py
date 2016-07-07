"""Detect noise source signal."""

import numpy as np
import tod_task

from caput import mpiutil
from caput import mpiarray


class Detect(tod_task.SingleRawTimestream):
    """Detect noise source signal."""

    params_init = {
                    'feed': 1, # use this feed
                    'sigma': 3.0,
                  }

    prefix = 'dt_'

    def process(self, rt):

        feed = self.params['feed']
        sigma = self.params['sigma']

        rt.redistribute(0) # make time the dist axis

        bls = [ set(b) for b in rt.bl ]
        bl_ind = bls.index({feed})

        tt_mean = mpiutil.gather_array(np.mean(rt.main_data.local_data[:, :, bl_ind].real, axis=-1), root=None)
        df =  np.diff(tt_mean, axis=-1)
        pdf = np.where(df>0, df, 0)
        pinds = np.where(pdf>pdf.mean() + sigma*pdf.std())[0]
        pinds = pinds + 1
        ndf = np.where(df<0, df, 0)
        ninds = np.where(ndf<ndf.mean() - sigma*ndf.std())[0]
        ninds = ninds + 1
        if ninds[0] < pinds[0]:
            pinds = np.insert(pinds, 0, 0)
        if pinds[-1] > ninds[-1]:
            ninds = np.insert(ninds, len(ninds), len(ninds))
        if len(pinds) != len(ninds):
            raise RuntimeError('Something wrong happened when detect the noise signal')

        period = np.sort(np.concatenate([np.diff(pinds), np.diff(ninds)]))[-1]
        on_time = np.sort(ninds-pinds)[-1]
        off_time = np.sort(pinds[1:] - ninds[:-1])[-1]
        if period != on_time + off_time:
            raise RuntimeError('period %d != on_time %d + off_time %d' % (period, on_time, off_time))
        num_period = np.int(np.ceil(len(tt_mean) / np.float(period)))
        tmp_ns_on = np.array(([True] * on_time + [False] * off_time) * num_period)[:len(tt_mean)]
        for si, ei in zip(pinds[:-1], pinds[1:]):
            if ei - si == period:
                ns_on = np.roll(tmp_ns_on, si)

        # import matplotlib
        # matplotlib.use('Agg')
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(np.where(ns_on, np.nan, tt_mean))
        # # plt.plot(pinds, tt_mean[pinds], 'ro')
        # # plt.plot(ninds, tt_mean[ninds], 'go')
        # plt.savefig('df.png')
        # err

        ns_on = mpiarray.MPIArray.from_numpy_array(ns_on)

        rt.create_main_time_ordered_dataset('ns_on', ns_on)

        rt.add_history(self.history)

        # rt.info()
        # print rt.main_time_ordered_datasert
        # print rt.time_ordered_datasert

        return rt
