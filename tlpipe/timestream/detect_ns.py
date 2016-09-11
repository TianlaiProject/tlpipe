"""Detect noise source signal."""

from collections import Counter
import numpy as np
import tod_task

from caput import mpiutil
from caput import mpiarray


class Detect(tod_task.IterRawTimestream):
    """Detect noise source signal."""

    params_init = {
                    'feed': None, # use this feed
                    'sigma': 3.0,
                  }

    prefix = 'dt_'

    def process(self, rt):

        feed = self.params['feed']
        sigma = self.params['sigma']

        rt.redistribute(0) # make time the dist axis

        bls = [ set(b) for b in rt.bl ]
        if feed is not None:
            bl_ind = bls.index({feed})
        else:
            bl_ind = np.where(rt.bl[:, 0]==rt.bl[:, 1])[0][0]

        tt_mean = mpiutil.gather_array(np.mean(rt.local_vis[:, :, bl_ind].real, axis=-1), root=None)
        df =  np.diff(tt_mean, axis=-1)
        pdf = np.where(df>0, df, 0)
        pinds = np.where(pdf>pdf.mean() + sigma*pdf.std())[0]
        pinds = pinds + 1
        pT = Counter(np.diff(pinds)).most_common(1)[0][0] # period of pinds
        ndf = np.where(df<0, df, 0)
        ninds = np.where(ndf<ndf.mean() - sigma*ndf.std())[0]
        ninds = ninds + 1
        nT = Counter(np.diff(ninds)).most_common(1)[0][0] # period of pinds
        if pT != nT:
            raise RuntimeError('Period of pinds %d != period of ninds %d' % (pT, nT))
        else:
            period = pT

        ninds = ninds.reshape(-1, 1)
        dinds = (ninds - pinds).flatten()
        on_time = Counter(dinds[dinds>0] % period).most_common(1)[0][0]
        off_time = Counter(-dinds[dinds<0] % period).most_common(1)[0][0]

        if period != on_time + off_time:
            raise RuntimeError('period %d != on_time %d + off_time %d' % (period, on_time, off_time))
        else:
            if mpiutil.rank0:
                print 'Detected noise source: period = %d, on_time = %d, off_time = %d' % (period, on_time, off_time)
        num_period = np.int(np.ceil(len(tt_mean) / np.float(period)))
        tmp_ns_on = np.array(([True] * on_time + [False] * off_time) * num_period)[:len(tt_mean)]
        on_start = Counter(pinds % period).most_common(1)[0][0]
        ns_on = np.roll(tmp_ns_on, on_start)

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
        rt['ns_on'].attrs['period'] = period
        rt['ns_on'].attrs['on_time'] = on_time
        rt['ns_on'].attrs['off_time'] = off_time

        # set vis_mask corresponding to ns_on
        on_inds = np.where(rt['ns_on'].local_data[:])[0]
        rt.local_vis_mask[on_inds] = True

        rt.add_history(self.history)

        # rt.info()

        return rt
