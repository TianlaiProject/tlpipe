"""Dispatch data."""

import numpy as np
import h5py
import tod_task

from caput import mpiutil
from caput import mpiarray


sidereal_day = 86164.0905 # seconds

class Dispatch(tod_task.IterRawTimestream):
    """Dispatch data."""

    params_init = {
                    'days': 1.0, # how many sidereal days
                    'exclude_bad': True, # exclude bad channels
                  }

    prefix = 'dp_'


    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(Dispatch, self).__init__(parameter_file_or_dict, feedback)

        days = self.params['days']
        mode = self.params['mode']
        start = self.params['start']
        stop = self.params['stop']
        dist_axis = self.params['dist_axis']

        tmp_tod = self._Tod_class(self.input_files, mode, start, stop, dist_axis)
        self.abs_start = tmp_tod.main_data_start
        self.abs_stop = tmp_tod.main_data_stop

        with h5py.File(self.input_files[0], 'r') as f:
            self.int_time = f.attrs['inttime']
        if self.iter_num is None:
            self.iter_num = np.int(np.ceil(self.int_time * (self.abs_stop - self.abs_start) / (days * sidereal_day)))


    def read_input(self):
        """Method for reading time ordered data input."""

        days = self.params['days']
        mode = self.params['mode']
        start = self.params['start']
        stop = self.params['stop']
        dist_axis = self.params['dist_axis']

        num_int = np.int(np.ceil(days * sidereal_day / self.int_time))
        start = self.abs_start + self.iteration * num_int
        stop = min(self.abs_stop, self.abs_start + (self.iteration + 1) * num_int)

        tod = self._Tod_class(self.input_files, mode, start, stop, dist_axis)

        tod = self.data_select(tod)

        tod.load_all()

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
            feeds = feedno[feed_select].tolist()
        # remove bad feeds from feeds
        for bf in bad_feed:
            if bf in feeds:
                feeds.remove(bf)

        # feed select
        tod.feed_select(feeds, self.params['corr'])

        return tod

    def process(self, rt):

        rt.add_history(self.history)

        # rt.info()

        return rt
