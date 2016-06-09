import numpy as np
import container


class RawTimestream(container.BasicTod):
    """Container class for the raw timestream data.

    The raw timestream data are raw visibilities (the main data) and other data
    and meta data saved in HDF5 files which are recorded from the correlator.
    """

    _main_data_name = 'vis'
    _main_data_axes = ('time', 'frequency', 'channelpair')
    _main_time_ordered_datasets = ('vis',)
    _time_ordered_datasets = ('vis', 'weather')
    _time_ordered_attrs = ('obstime', 'sec1970')


    def time_select(self, value):
        """Select data to be loaded from input files along the time axis.

        Parameters
        ----------
        value : tuple or list
            If a tuple, which will be created as a slice(start, stop, step) object,
            so it can have one to three elements (integers or None); if a list, its
            elements must be strictly increasing non-negative integers, data in
            these positions will be selected.

        """
        self.data_select('time', value)

    def frequency_select(self, value):
        """Select data to be loaded from input files along the frequency axis.

        Parameters
        ----------
        value : tuple or list
            If a tuple, which will be created as a slice(start, stop, step) object,
            so it can have one to three elements (integers or None); if a list, its
            elements must be strictly increasing non-negative integers, data in
            these positions will be selected.

        """
        self.data_select('frequency', value)

    def channelpair_select(self, value):
        """Select data to be loaded from input files along the channelpair axis.

        Parameters
        ----------
        value : tuple or list
            If a tuple, which will be created as a slice(start, stop, step) object,
            so it can have one to three elements (integers or None); if a list, its
            elements must be strictly increasing non-negative integers, data in
            these positions will be selected.

        """
        self.data_select('channelpair', value)

    def channel_select(self, value=(0, None), corr='all'):
        """Select data to be loaded from inputs files corresponding to the specified channels.

        Parameters
        ----------
        value : tuple or list, optional
            If a tuple, which will be created as a slice(start, stop, step) object,
            so it can have one to three elements (integers or None); if a list,
            channel No. in this list will be selected. Default (0, None) select all.
        corr : 'all', 'auto' or 'cross', optional
            Correlation type. 'auto' for auto-correlations, 'cross' for
            cross-correlations, 'all' for all correlations. Default 'all'.

        """
        # get channo info from the first input file
        channo = self.infiles[0]['channo'][:]
        channo1d = np.sort(channo.flatten())

        if isinstance(value, tuple):
            channels = channo1d[slice(*value)]
        elif isinstance(value, list):
            channels = np.intersect1d(channo1d, value)
        else:
            raise ValueError('Unsupported data selection %s' % value)

        nchan = len(channels)
        # use set for easy comparison
        if corr == 'auto':
            channel_pairs = [ {channels[i]} for i in range(nchan) ]
        elif corr == 'cross':
            channel_pairs = [ {channels[i], channels[j]} for i in range(nchan) for j in range(i+1, nchan) ]
        elif corr == 'all':
            channel_pairs = [ {channels[i], channels[j]} for i in range(nchan) for j in range(i, nchan) ]
        else:
            raise ValueError('Unknown correlation type %s' % corr)

        # get blorder info from the first input file
        blorder = self.infiles[0]['blorder']
        blorder = [ set(bl) for bl in blorder ]

        # channel pair indices
        indices = { blorder.index(chp) for chp in channel_pairs }
        indices = sorted(list(indices))

        self.data_select('channelpair', indices)
