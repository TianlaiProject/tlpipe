"""Base pipeline tasks for time ordered data opteration."""

from tlpipe.pipeline.pipeline import SingleBase, IterBase


class SingleTod(SingleBase):
    """Provides time ordered data IO and data selection for non-iterating pipeline
    tasks.

    Provides the methods `read_input`, `read_output` and `write_output` for
    time ordered data.

    """

    from container import BasicTod
    _Tod_class = BasicTod

    params_init = {
                    'mode': 'r',
                    'start': 0,
                    'stop': None,
                    'dist_axis': 0,
                    'exclude': [],
                    'check_status': True,
                    'libver': 'latest',
                  }

    prefix = 'stod_'


    def process(self, tod):

        tod.add_history(self.history)

        return tod

    def data_select(self, tod):
        """Data select."""
        return tod

    def read_input(self):
        """Method for reading time ordered data input."""

        mode = self.params['mode']
        start = self.params['start']
        stop = self.params['stop']
        dist_axis = self.params['dist_axis']

        tod = self._Tod_class(self.input_files, mode, start, stop, dist_axis)

        tod = self.data_select(tod)

        tod.load_all()

        return tod

    def write_output(self, output):
        """Method for writing time ordered data output. """

        exclude = self.params['exclude']
        check_status = self.params['check_status']
        libver = self.params['libver']

        output.to_files(self.output_files, exclude, check_status, libver)


class SingleRawTimestream(SingleTod):
    """Provides raw timestream IO and data selection for non-iterating pipeline
    tasks.

    Provides the methods `read_input`, `read_output` and `write_output` for
    raw timestream data.

    """

    from raw_timestream import RawTimestream
    _Tod_class = RawTimestream

    params_init = {
                    'time_select': (0, None),
                    'freq_select': (0, None),
                    'feed_select': (0, None),
                    'corr': 'all',
                  }

    prefix = 'rt_'


    def data_select(self, tod):
        """Data select."""
        tod.time_select(self.params['time_select'])
        tod.frequency_select(self.params['freq_select'])
        tod.feed_select(self.params['feed_select'], self.params['corr'])

        return tod


class SingleTimestream(SingleTod):
    """Provides timestream IO and data selection for non-iterating pipeline tasks.

    Provides the methods `read_input`, `read_output` and `write_output` for
    timestream data.

    """

    from timestream import Timestream
    _Tod_class = Timestream

    params_init = {
                    'time_select': (0, None),
                    'freq_select': (0, None),
                    'pol_select': (0, None),
                    'feed_select': (0, None),
                    'corr': 'all',
                  }

    prefix = 'ts_'


    def data_select(self, tod):
        """Data select."""
        tod.time_select(self.params['time_select'])
        tod.frequency_select(self.params['freq_select'])
        tod.polarization_select(self.params['pol_select'])
        tod.feed_select(self.params['feed_select'], self.params['corr'])

        return tod



class IterTod(IterBase):
    """Provides time ordered data IO and data selection for iterating pipeline tasks.

    Provides the methods `read_input`, `read_output` and `write_output` for
    time ordered data.

    """

    from container import BasicTod
    _Tod_class = BasicTod

    params_init = {
                    'mode': 'r',
                    'start': 0,
                    'stop': None,
                    'dist_axis': 0,
                    'exclude': [],
                    'check_status': True,
                    'libver': 'latest',
                  }

    prefix = 'itod_'


    def process(self, tod):

        tod.add_history(self.history)

        return tod

    def data_select(self, tod):
        """Data select."""
        return tod

    def read_input(self):
        """Method for reading time ordered data input."""

        mode = self.params['mode']
        start = self.params['start']
        stop = self.params['stop']
        dist_axis = self.params['dist_axis']

        tod = self._Tod_class(self.input_files, mode, start, stop, dist_axis)

        tod = self.data_select(tod)

        tod.load_all()

        return tod

    def write_output(self, output):
        """Method for writing time ordered data output. """

        exclude = self.params['exclude']
        check_status = self.params['check_status']
        libver = self.params['libver']

        output.to_files(self.output_files, exclude, check_status, libver)


class IterRawTimestream(IterTod):
    """Provides raw timestream IO and data selection for iterating pipeline tasks.

    Provides the methods `read_input`, `read_output` and `write_output` for
    raw timestream data.

    """

    from raw_timestream import RawTimestream
    _Tod_class = RawTimestream

    params_init = {
                    'time_select': (0, None),
                    'freq_select': (0, None),
                    'feed_select': (0, None),
                    'corr': 'all',
                  }

    prefix = 'rt_'


    def data_select(self, tod):
        """Data select."""
        tod.time_select(self.params['time_select'])
        tod.frequency_select(self.params['freq_select'])
        tod.feed_select(self.params['feed_select'], self.params['corr'])

        return tod


class IterTimestream(IterTod):
    """Provides timestream IO and data selection for iterating pipeline tasks.

    Provides the methods `read_input`, `read_output` and `write_output` for
    timestream data.

    """

    from timestream import Timestream
    _Tod_class = Timestream

    params_init = {
                    'time_select': (0, None),
                    'freq_select': (0, None),
                    'pol_select': (0, None),
                    'feed_select': (0, None),
                    'corr': 'all',
                  }

    prefix = 'ts_'


    def data_select(self, tod):
        """Data select."""
        tod.time_select(self.params['time_select'])
        tod.frequency_select(self.params['freq_select'])
        tod.polarization_select(self.params['pol_select'])
        tod.feed_select(self.params['feed_select'], self.params['corr'])

        return tod
