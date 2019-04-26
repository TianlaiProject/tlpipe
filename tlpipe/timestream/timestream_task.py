"""Base pipeline task for operating on timestream objects.

Inheritance diagram
-------------------

.. inheritance-diagram:: tlpipe.pipeline.pipeline.TaskBase tlpipe.pipeline.pipeline.OneAndOne TimestreamTask
   :parts: 2

"""

import sys, traceback
from os import path
import logging
import h5py
from tlpipe.container.timestream_common import TimestreamCommon
from tlpipe.container.raw_timestream import RawTimestream
from tlpipe.container.timestream import Timestream
from tlpipe.utils.path_util import input_path, output_path
from tlpipe.pipeline.pipeline import OneAndOne
from caput import mpiutil


# Set the module logger.
logger = logging.getLogger(__name__)


class TimestreamTask(OneAndOne):
    """Task that provides raw timestream or timestream IO and data selection operations.

    Provides the methods `read_input`, `read_output` and `write_output` for
    raw timestream or timestream data.

    This is usually used as a direct base class as those tasks that operates on
    timestream data (can be data held in both
    :class:`~tlpipe.container.raw_timestream.RawTimestream` and
    :class:`~tlpipe.container.timestream.Timestream`), which can determine which
    data container the data is being held from the *input* or from data stored in
    the input data files.

    """

    _Tod_class = TimestreamCommon

    params_init = {
                    'mode': 'r',
                    'start': 0,
                    'stop': None,
                    'dist_axis': 0,
                    'exclude': [],
                    'check_status': True,
                    'write_hints': True,
                    'libver': 'earliest', # earliest to have best backward compatibility
                    'chunk_vis': False, # chunk vis and vis_mask in saved files
                    'chunk_shape': None,
                    'chunk_size': 64, # KB
                    'output_failed_continue': False, # continue to run if output to files failed
                    'time_select': (0, None),
                    'freq_select': (0, None),
                    'pol_select': (0, None), # only useful for ts
                    'feed_select': (0, None),
                    'corr': 'all',
                    'show_progress': False,
                    'progress_step': None,
                    'show_info': False,
                    'tag_input_iter': True, # tag current iteration to input file path
                    'tag_output_iter': True, # tag current iteration to output file path
                  }

    prefix = 'tt_'

    def read_process_write(self, tod):
        """Reads input, executes any processing and writes output."""

        # determine if rt or ts from the input tod, and set the correct _Tod_class
        if self._no_input:
            if not tod is None:
                # This should never happen.  Just here to catch bugs.
                raise RuntimeError("Somehow `input` was set.")
        else:
            # read from files
            if tod is None:
                if self.input_files is None or len(self.input_files) == 0:
                    if mpiutil.rank0:
                        msg = 'No file to read from, will stop then...'
                        logger.info(msg)
                    self.stop_iteration(True)
                    return None
                tag_input_iter = self.params['tag_input_iter']
                if self.iterable and tag_input_iter:
                    input_files = input_path(self.input_files, iteration=self.iteration)
                else:
                    input_files = self.input_files
                # ensure all input_files are exist
                for infile in input_files:
                    if not path.exists(infile):
                        if mpiutil.rank0:
                            msg = 'Missing input file %s, will stop then...' % infile
                            logger.info(msg)
                        self.stop_iteration(True)
                        return None
                # see 'vis' dataset from the first input file
                with h5py.File(input_files[0], 'r') as f:
                    vis_shp = f['vis'].shape
                if len(vis_shp) == 3:
                    self._Tod_class = RawTimestream
                elif len(vis_shp) == 4:
                    self._Tod_class = Timestream
                else:
                    raise RuntimeError('Something wrong happened, dimension of vis data != 3 or 4')
            # from arg
            else:
                if isinstance(tod, RawTimestream):
                    self._Tod_class = RawTimestream
                elif isinstance(tod, Timestream):
                    self._Tod_class = Timestream
                else:
                    raise ValueError('Invaid input %s, need either a RawTimestream or Timestream object' % tod)

                tod, full_data = self.subset_select(tod)
                if not full_data:
                    tod = tod.subset(return_copy=False)

        return super(TimestreamTask, self).read_process_write(tod)

    def read_input(self):
        """Method for reading time ordered data input."""

        mode = self.params['mode']
        start = self.params['start']
        stop = self.params['stop']
        dist_axis = self.params['dist_axis']
        tag_input_iter = self.params['tag_input_iter']

        if self.iterable and tag_input_iter:
            input_files = input_path(self.input_files, iteration=self.iteration)
        else:
            input_files = self.input_files
        tod = self._Tod_class(input_files, mode, start, stop, dist_axis)

        tod, full_data = self.data_select(tod)

        tod.load_all()

        return tod

    def full_data_select(self):
        """Check to see whether select all data or not."""
        # may need better check here in future...
        full_data = True
        if self.params['time_select'] != (0, None):
            full_data = False
        if self.params['freq_select'] != (0, None):
            full_data = False
        if self._Tod_class == Timestream and self.params['pol_select'] != (0, None):
            full_data = False
        if self.params['feed_select'] != (0, None) or self.params['corr'] != 'all':
            full_data = False

        return full_data

    def data_select(self, tod):
        """Data select."""
        # may need better check here in future...
        full_data = True
        if self.params['time_select'] != (0, None):
            full_data = False
            tod.time_select(self.params['time_select'])
        if self.params['freq_select'] != (0, None):
            full_data = False
            tod.frequency_select(self.params['freq_select'])
        if self._Tod_class == Timestream and self.params['pol_select'] != (0, None):
            full_data = False
            tod.polarization_select(self.params['pol_select'])
        if self.params['feed_select'] != (0, None) or self.params['corr'] != 'all':
            full_data = False
            tod.feed_select(self.params['feed_select'], self.params['corr'])

        return tod, full_data

    def subset_select(self, tod):
        """Data subset select."""
        # may need better check here in future...
        full_data = True
        if self.params['time_select'] != (0, None):
            full_data = False
            tod.subset_time_select(self.params['time_select'])
        if self.params['freq_select'] != (0, None):
            full_data = False
            tod.subset_frequency_select(self.params['freq_select'])
        if self._Tod_class == Timestream and self.params['pol_select'] != (0, None):
            full_data = False
            tod.subset_polarization_select(self.params['pol_select'])
        if self.params['feed_select'] != (0, None) and self.params['corr'] != 'all':
            full_data = False
            tod.subset_feed_select(self.params['feed_select'], self.params['corr'])

        return tod, full_data

    def copy_input(self, tod):
        """Return a copy of tod, so the original tod would not be changed."""
        return tod.copy()

    def process(self, tod):

        tod.add_history(self.history)

        if self.params['show_info']:
            tod.info()

        return tod

    def write_output(self, output):
        """Method for writing time ordered data output. """

        exclude = self.params['exclude']
        check_status = self.params['check_status']
        write_hints = self.params['write_hints']
        libver = self.params['libver']
        chunk_vis = self.params['chunk_vis']
        chunk_shape = self.params['chunk_shape']
        chunk_size = self.params['chunk_size']
        output_failed_continue = self.params['output_failed_continue']
        tag_output_iter = self.params['tag_output_iter']

        if self.iterable and tag_output_iter:
            output_files = output_path(self.output_files, relative=False, iteration=self.iteration)
        else:
            output_files = self.output_files

        try:
            output.to_files(output_files, exclude, check_status, write_hints, libver, chunk_vis, chunk_shape, chunk_size)
        except Exception as e:
            if output_failed_continue:
                msg = 'Process %d writing output to files failed...' % mpiutil.rank
                logger.warning(msg)
                traceback.print_exc(file=sys.stdout)
            else:
                raise e
