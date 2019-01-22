"""Mask the data when signal of the Sun is strong.

Inheritance diagram
-------------------

.. inheritance-diagram:: Mask
   :parts: 2

"""

import numpy as np
import ephem
import aipy as a
import timestream_task


class Mask(timestream_task.TimestreamTask):
    """Mask the data when signal of the Sun is strong."""

    params_init = {
                    'span': 30, # minutes
                  }

    prefix = 'sm_'

    def process(self, ts):

        span = self.params['span']

        nt = ts.local_vis.shape[0] # number of time points of this process

        if nt > 0:

            srclist, cutoff, catalogs = a.scripting.parse_srcs('Sun', 'misc')
            cat = a.src.get_catalog(srclist, cutoff, catalogs)
            s = cat.values()[0] # the Sun

            # get transit time of calibrator
            # array
            aa = ts.array
            local_juldate = ts['jul_date'].local_data
            aa.set_jultime(local_juldate[0]) # the first obs time point of this process

            mask_inds = []

            # previous transit
            prev_transit = aa.previous_transit(s)
            prev_transit_start = a.phs.ephem2juldate(prev_transit - 0.5 * span * ephem.minute) # Julian date
            prev_transit_end = a.phs.ephem2juldate(prev_transit + 0.5 * span * ephem.minute) # Julian date
            prev_transit_start_ind = np.searchsorted(local_juldate, prev_transit_start, side='left')
            prev_transit_end_ind = np.searchsorted(local_juldate, prev_transit_end, side='right')
            if prev_transit_end_ind > 0:
                mask_inds.append((prev_transit_start_ind, prev_transit_end_ind))

            # next transit
            next_transit = aa.next_transit(s)
            next_transit_start = a.phs.ephem2juldate(next_transit - 0.5 * span * ephem.minute) # Julian date
            next_transit_end = a.phs.ephem2juldate(next_transit + 0.5 * span * ephem.minute) # Julian date
            next_transit_start_ind = np.searchsorted(local_juldate, next_transit_start, side='left')
            next_transit_end_ind = np.searchsorted(local_juldate, next_transit_end, side='right')
            if next_transit_start_ind < nt:
                mask_inds.append((next_transit_start_ind, next_transit_end_ind))

            # then all next transit if data is long enough
            while (next_transit_end_ind < nt):
                aa.set_jultime(next_transit_end)
                next_transit = aa.next_transit(s)
                next_transit_start = a.phs.ephem2juldate(next_transit - 0.5 * span * ephem.minute) # Julian date
                next_transit_end = a.phs.ephem2juldate(next_transit + 0.5 * span * ephem.minute) # Julian date
                next_transit_start_ind = np.searchsorted(local_juldate, next_transit_start, side='left')
                next_transit_end_ind = np.searchsorted(local_juldate, next_transit_end, side='right')
                if next_transit_start_ind < nt:
                    mask_inds.append((next_transit_start_ind, next_transit_end_ind))

            # set mask
            for si, ei in mask_inds:
                ts.local_vis_mask[si:ei] = True # do not change vis directly

        return super(Mask, self).process(ts)
