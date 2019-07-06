"""Detect noise source signal by using an appropriate threshold.

Inheritance diagram
-------------------

.. inheritance-diagram:: Detect
   :parts: 2

"""

import warnings
from collections import Counter
import numpy as np
import timestream_task
from tlpipe.container.raw_timestream import RawTimestream
from caput import mpiutil
from caput import mpiarray


class Detect(timestream_task.TimestreamTask):
    """Detect noise source signal by using an appropriate threshold.

    This task automatically finds out the time points that the noise source
    is **on**, and creates a new bool dataset "ns_on" with elements *True*
    corresponding to time points when the noise source is **on**.

    """

    params_init = {
                    'channel': None, # use auto-correlation of this channel
                    'mask_near': 1, # how many extra near ns_on int_time to be masked
                  }

    prefix = 'dt_'

    def process(self, rt):

        assert isinstance(rt, RawTimestream), '%s only works for RawTimestream object currently' % self.__class__.__name__

        channel = self.params['channel']
        mask_near = max(0, int(self.params['mask_near']))

        rt.redistribute(0) # make time the dist axis

        auto_inds = np.where(rt.bl[:, 0]==rt.bl[:, 1])[0].tolist() # inds for auto-correlations
        channels = [ rt.bl[ai, 0] for ai in auto_inds ] # all chosen channels
        if channel is not None:
            if channel in channels:
                bl_ind = auto_inds[channels.index(channel)]
            else:
                bl_ind = auto_inds[0]
                if mpiutil.rank0:
                    print 'Warning: Required channel %d doen not in the data, use channel %d instead' % (channel, rt.bl[bl_ind, 0])
        else:
            bl_ind = auto_inds[0]
        # move the chosen channel to the first
        auto_inds.remove(bl_ind)
        auto_inds = [bl_ind] + auto_inds

        for bl_ind in auto_inds:
            this_chan = rt.bl[bl_ind, 0] # channel of this bl_ind
            vis = np.ma.array(rt.local_vis[:, :, bl_ind].real, mask=rt.local_vis_mask[:, :, bl_ind])
            cnt = vis.count() # number of not masked vals
            total_cnt = mpiutil.allreduce(cnt)
            vis_shp = rt.vis.shape
            ratio = float(total_cnt) / np.prod((vis_shp[0], vis_shp[1])) # ratio of un-masked vals
            if ratio < 0.5: # too many masked vals
                if mpiutil.rank0:
                    warnings.warn('Too many masked values for auto-correlation of Channel: %d, does not use it' % this_chan)
                continue

            tt_mean = mpiutil.gather_array(np.ma.mean(vis, axis=-1).filled(0), root=None)
            tt_mean_sort = np.sort(tt_mean)
            ttms_diff = np.diff(tt_mean_sort)
            ind = np.argmax(ttms_diff)
            if ind > 0 and ttms_diff[ind] > 2.0 * max(ttms_diff[ind-1], ttms_diff[ind+1]):
                sep = 0.5 * (tt_mean_sort[ind] + tt_mean_sort[ind+1])
                break
        else:
            raise RuntimeError('Failed to get the threshold to separate ns signal out')

        ns_on = np.where(tt_mean > sep, True, False)
        nTs = []
        nFs = []
        nT = 1 if ns_on[0] else 0
        nF = 1 if not ns_on[0] else 0
        for i in range(1, len(ns_on)):
            if ns_on[i]:
                if ns_on[i] == ns_on[i-1]:
                    nT += 1
                else:
                    nT = 1
                    nFs.append(nF)
            else:
                if ns_on[i] == ns_on[i-1]:
                    nF += 1
                else:
                    nF = 1
                    nTs.append(nT)
        on_time = Counter(nTs).most_common(1)[0][0]
        off_time = Counter(nFs).most_common(1)[0][0]
        period = on_time + off_time

        if 'noisesource' in rt.iterkeys():
            if rt['noisesource'].shape[0] == 1: # only 1 noise source
                start, stop, cycle = rt['noisesource'][0, :]
                int_time = rt.attrs['inttime']
                true_on_time = np.round((stop - start)/int_time)
                true_period = np.round(cycle / int_time)
                if on_time != true_on_time and period != true_period: # inconsistant with the record in the data
                    if mpiutil.rank0:
                        warnings.warn('Detected noise source info is inconsistant with the record in the data for auto-correlation of Channel: %d: on_time %d != record_on_time %d, period != record_period %d, does not use it' % (this_chan, on_time, true_on_time, period, true_period))
            elif rt['noisesource'].shape[0] >= 2: # more than 1 noise source
                if mpiutil.rank0:
                    warnings.warn('More than 1 noise source, do not know how to deal with this currently')

        if mpiutil.rank0:
            print 'Detected noise source: period = %d, on_time = %d, off_time = %d' % (period, on_time, off_time)


        ns_on1 = mpiarray.MPIArray.from_numpy_array(ns_on)

        rt.create_main_time_ordered_dataset('ns_on', ns_on1)
        rt['ns_on'].attrs['period'] = period
        rt['ns_on'].attrs['on_time'] = on_time
        rt['ns_on'].attrs['off_time'] = off_time

        # set vis_mask corresponding to ns_on
        on_inds = np.where(rt['ns_on'].local_data[:])[0]
        rt.local_vis_mask[on_inds] = True

        if mask_near > 0:
            on_inds = np.where(ns_on)[0]
            new_on_inds = on_inds.tolist()
            for i in xrange(1, mask_near+1):
                new_on_inds = new_on_inds + (on_inds-i).tolist() + (on_inds+i).tolist()
            new_on_inds = np.unique(new_on_inds)

            if rt['vis_mask'].distributed:
                start = rt.vis_mask.local_offset[0]
                end = start + rt.vis_mask.local_shape[0]
            else:
                start = 0
                end = rt.vis_mask.shape[0]
            global_inds = np.arange(start, end).tolist()
            new_on_inds = np.intersect1d(new_on_inds, global_inds)
            local_on_inds = [ global_inds.index(i) for i in new_on_inds ]
            rt.local_vis_mask[local_on_inds] = True # set mask using global slicing

        return super(Detect, self).process(rt)
