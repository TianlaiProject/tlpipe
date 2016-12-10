"""Dispatch data.

Inheritance diagram
-------------------

.. inheritance-diagram:: Dispatch
   :parts: 2

"""

import numpy as np
import h5py
import tod_task
from tlpipe.core import constants as const

from caput import mpiutil


class Dispatch(tod_task.TaskTimestream):
    """Dispatch data.

    This task will (maybe iteratively) load data from the input data files
    according to the required data selection (time, frequency, baseline
    selection). If work iteratively, data will be iteratively loaded according
    to the time unit set in the input pipe file, and the loaded data (contained
    in a data container which is a
    :class:`~tlpipe.timestream.raw_timestream.RawTimestream` object) will be
    dispatched to other tasks to be further processed.

    .. note::
        This usually should be the first task in the input pipe file to select
        and load the data from input data files for other tasks.

    .. note::
        Current this task only works for a continuously observed data sets.
        Works need to do to make it work also for data observed in dis-continuous
        time periods.

    """

    params_init = {
                    'days': 1.0, # how many sidereal days in one iteration
                    'extra_inttime': 150, # extra int time to ensure smooth transition in the two ends
                    'exclude_bad': True, # exclude bad channels
                  }

    prefix = 'dp_'

    def read_input(self):
        """Method for (maybe iteratively) reading data from input data files."""

        days = self.params['days']
        extra_inttime = self.params['extra_inttime']
        mode = self.params['mode']
        start = self.params['start']
        stop = self.params['stop']
        dist_axis = self.params['dist_axis']

        if self._iter_cnt == 0: # the first iteration
            tmp_tod = self._Tod_class(self.input_files, mode, start, stop, dist_axis)
            self.abs_start = tmp_tod.main_data_start
            self.abs_stop = tmp_tod.main_data_stop

            with h5py.File(self.input_files[0], 'r') as f:
                self.int_time = f.attrs['inttime']
            if self.iterable and self.iter_num is None:
                self.iter_num = np.int(np.ceil(self.int_time * (self.abs_stop - self.abs_start - 2*extra_inttime) / (days * const.sday)))

        # num_int = np.int(np.around(days * const.sday / self.int_time)) # number of int_time
        iteration = self.iteration if self.iterable else 0
        start = self.abs_start + np.int(np.around(iteration * days * const.sday / self.int_time))
        stop = min(self.abs_stop, np.int(np.around((iteration+1) * days * const.sday / self.int_time)) + 2*extra_inttime)

        tod = self._Tod_class(self.input_files, mode, start, stop, dist_axis)

        tod = self.data_select(tod)

        tod.load_all() # load in all data

        if self._iter_cnt == 0: # the first iteration
            ra_dec = mpiutil.gather_array(tod['ra_dec'].local_data, root=None)
            self.start_ra = ra_dec[extra_inttime, 0]
        tod.vis.attrs['start_ra'] = self.start_ra # used for re_order

        return tod


    def data_select(self, tod):
        """Data select."""
        super(Dispatch, self).data_select(tod)

        if self.params['exclude_bad']:
            with h5py.File(self.input_files[0], 'r') as f:
                channo = f['channo'][:]
                feedno = f['feedno'][:]
                try:
                    badchn = f['channo'].attrs['badchn']
                except KeyError:
                    # no badchn
                    return tod

            bad_feed = [ feedno[np.where(channo == bc)[0][0]] for bc in badchn ]
            feed_select = self.params['feed_select']
            if isinstance(feed_select, tuple):
                feeds = feedno[slice(*feed_select)].tolist()
            elif isinstance(feed_select, list):
                feeds = feed_select
            # remove bad feeds from feeds
            for bf in bad_feed:
                if bf in feeds:
                    feeds.remove(bf)

            # feed select
            tod.feed_select(feeds, self.params['corr'])

        return tod

    def process(self, rt):
        """Return loaded data as a
        :class:`~tlpipe.timestream.raw_timestream.RawTimestream` object."""

        rt.add_history(self.history)

        # rt.info()

        return rt
