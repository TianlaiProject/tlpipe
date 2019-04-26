"""Dispatch data.

Inheritance diagram
-------------------

.. inheritance-diagram:: Dispatch
   :parts: 2

"""

import itertools
import numpy as np
import h5py
import timestream_task
from tlpipe.core import constants as const

from caput import mpiutil


class Dispatch(timestream_task.TimestreamTask):
    """Dispatch data.

    This task will (maybe iteratively) load data from the input data files
    according to the required data selection (time, frequency, baseline
    selection). If work iteratively, data will be iteratively loaded according
    to the time unit set in the input pipe file, and the loaded data (contained
    in a data container which is a
    :class:`~tlpipe.container.raw_timestream.RawTimestream` object) will be
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
                    'drop_days': 0.0, # drop data if time is less than this factor of days
                  }

    prefix = 'dp_'

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(Dispatch, self).__init__(parameter_file_or_dict, feedback)

        # group state
        self.grp_cnt = 0
        self.next_grp = True
        self.abs_start = None
        self.abs_stop = None

        # int time for later use
        self.int_time = None
        # record start RA for later use
        self.start_ra = None

    def _init_input_files(self):
        input_files = self.params['input_files']
        start = self.params['start']
        stop = self.params['stop']

        # regularize file groups
        if input_files is None: # no file
            self.input_grps = [] # empty group
            self.start = []
            self.stop = []
        elif isinstance(input_files, str): # in case a single file name
            self.input_grps = [ [input_files] ] # a single group
            if isinstance(start, int):
                self.start = [ start ]
            else:
                raise ValueError('Paramter start should be a integer, not %s' % start)
            if isinstance(stop, int) or stop is None:
                self.stop = [ stop ]
            else:
                raise ValueError('Paramter stop should be a integer or None, not %s' % stop)
        elif isinstance(input_files, list): # in case of a list
            if len(input_files) == 0:
                self.input_grps = [] # empty group
                self.start = []
                self.stop = []
            else:
                if isinstance(input_files[0], str): # in case a list of file names
                    for fn in input_files:
                        if not isinstance(fn, str):
                            raise ValueError('Paramter input_files should be a list of file names, not contain %s', fn)
                    self.input_grps = [ input_files[:] ] # a single group

                    if isinstance(start, int):
                        self.start = [ start ]
                    else:
                        raise ValueError('Paramter start should be a integer, not %s' % start)
                    if isinstance(stop, int) or stop is None:
                        self.stop = [ stop ]
                    else:
                        raise ValueError('Paramter stop should be a integer or None, not %s' % stop)
                elif isinstance(input_files[0], list): # in case a list of lists, i.e., several groups
                    self.input_grps = []
                    for li, lt in enumerate(input_files):
                        if not isinstance(lt, list):
                            raise ValueError('Parameter input_files contains list, so its elements should all be a list of file names, not %s' % lt)
                        else:
                            if len(lt) == 0:
                                continue
                            for fn in lt:
                                if not isinstance(fn, str):
                                    raise ValueError('Paramter input_files[%d] should be a list of file names, not contain %s', (li, fn))

                            self.input_grps.append(lt[:])

                    if isinstance(start, int):
                        self.start = [ start ] * len(self.input_grps)
                    elif isinstance(start, list):
                        if not len(start) == len(self.input_grps):
                            raise ValueError('Parameter start is a list, so it must have a same length with input_files: %d != %d' % (len(start), len(self.input_grps)))
                        for st in start:
                            if not isinstance(st, int):
                                raise ValueError('Parameter start is a list, its elements should be integers, not %s' % st)
                        self.start = start
                    else:
                        raise ValueError('Paramter start should be a integer or a list of integers, not %s' % start)

                    if isinstance(stop, int) or stop is None:
                        self.stop = [ stop ] * len(self.input_grps)
                    elif isinstance(stop, list):
                        if not len(stop) == len(self.input_grps):
                            raise ValueError('Parameter stop is a list, so it must have a same length with input_files: %d != %d' % (len(stop), len(self.input_grps)))
                        for sp in stop:
                            if not (isinstance(sp, int) or sp is None):
                                raise ValueError('Parameter start is a list, its elements should be integers or None, not %s', sp)
                        self.stop = stop
                    else:
                        raise ValueError('Paramter stop should be a integer or None or a list of integers and None, not %s' % stop)
                else:
                    raise ValueError('Parater input_files should be a list of file names or a list of lists of files names, not contain %s' % input_files[0])

        else:
            raise ValueError('Parater input_files should be a single file name or a list with its elements being file names or lists of file names')

        # reset self.input_files
        self.input_files = list(itertools.chain(*self.input_grps)) # flat input_grps


    def read_process_write(self, tod):
        """Reads input, executes any processing and writes output."""

        ngrp = len(self.input_grps)
        if self.grp_cnt >= ngrp:
            self.stop_iteration(True)
            return None

        # set input_files as this group of files
        self.input_files = self.input_grps[self.grp_cnt]

        return super(Dispatch, self).read_process_write(tod)

    def read_input(self):
        """Method for (maybe iteratively) reading data from input data files."""

        days = self.params['days']
        extra_inttime = self.params['extra_inttime']
        drop_days = self.params['drop_days']
        mode = self.params['mode']
        dist_axis = self.params['dist_axis']

        ngrp = len(self.input_grps)

        if self.next_grp and ngrp > 1:
            if mpiutil.rank0:
                print 'Start file group %d of %d...' % (self.grp_cnt, ngrp)
            self.restart_iteration() # re-start iteration for each group
            self.next_grp = False
            self.abs_start = None
            self.abs_stop = None

        input_files = self.input_grps[self.grp_cnt]
        start = self.start[self.grp_cnt]
        stop = self.stop[self.grp_cnt]

        if self.int_time is None:
            # NOTE: here assume all files have the same int_time
            with h5py.File(self.input_files[0], 'r') as f:
                self.int_time = f.attrs['inttime']

        if self.abs_start is None or self.abs_stop is None:
            tmp_tod = self._Tod_class(input_files, mode, start, stop, dist_axis)
            self.abs_start = tmp_tod.main_data_start
            self.abs_stop = tmp_tod.main_data_stop
            del tmp_tod

        iteration = self.iteration if self.iterable else 0
        this_start = self.abs_start + np.int(np.around(iteration * days * const.sday / self.int_time))
        this_stop = min(self.abs_stop, self.abs_start + np.int(np.around((iteration+1) * days * const.sday / self.int_time)) + 2*extra_inttime)
        if  this_stop >= self.abs_stop:
            self.next_grp = True
            self.grp_cnt += 1
        if this_start >= this_stop:
            self.next_grp = True
            self.grp_cnt += 1
            return None

        this_span = self.int_time * (this_stop - this_start) # in unit second
        if this_span < drop_days * const.sday:
            if mpiutil.rank0:
                print 'Not enough span time, drop it...'
            return None
        elif (this_stop - this_start) <= extra_inttime: # use int comparision
            if mpiutil.rank0:
                print 'Not enough span time (less than `extra_inttime`), drop it...'
            return None

        tod = self._Tod_class(input_files, mode, this_start, this_stop, dist_axis)

        tod, _ = self.data_select(tod)

        tod.load_all() # load in all data

        if self.start_ra is None: # the first iteration
            if 'time' == tod.main_data_axes[tod.main_data_dist_axis]:
                # ra_dec is distributed among processes
                # find the point of ra_dec[extra_inttime, 0] of the global array
                local_offset = tod['ra_dec'].local_offset[0]
                local_shape = tod['ra_dec'].local_shape[0]
                if local_offset <= extra_inttime and extra_inttime < local_offset + local_shape:
                    in_this = 1
                    start_ra = tod['ra_dec'].local_data[extra_inttime-local_offset, 0]
                else:
                    in_this = 0
                    start_ra = None

                # get the rank
                max_val, in_rank = mpiutil.allreduce((in_this, tod.rank), op=mpiutil.MAXLOC, comm=tod.comm)
                # bcast from this rank
                start_ra = mpiutil.bcast(start_ra, root=in_rank, comm=tod.comm)
                self.start_ra = start_ra
            else:
                self.start_ra = ra_dec[extra_inttime, 0]

        tod.vis.attrs['start_ra'] = self.start_ra # used for re_order

        return tod

    def data_select(self, tod):
        """Data select."""
        tod, full_data = super(Dispatch, self).data_select(tod)

        if self.params['exclude_bad']:
            with h5py.File(self.input_files[0], 'r') as f:
                channo = f['channo'][:]
                feedno = f['feedno'][:]
                try:
                    badchn = f['channo'].attrs['badchn']
                except KeyError:
                    # no badchn
                    return tod, full_data

            if badchn.size > 0:
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
                full_data = False

        return tod, full_data

    def process(self, rt):
        """Return loaded data as a
        :class:`~tlpipe.container.raw_timestream.RawTimestream` object."""

        return super(Dispatch, self).process(rt)
