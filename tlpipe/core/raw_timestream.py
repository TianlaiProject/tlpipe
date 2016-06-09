import itertools
import numpy as np
import container
from caput import memh5
from caput import mpiutil


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

    def feed_select(self, value=(0, None), corr='all'):
        """Select data to be loaded from inputs files corresponding to the specified feeds.

        Parameters
        ----------
        value : tuple or list, optional
            If a tuple, which will be created as a slice(start, stop, step) object,
            so it can have one to three elements (integers or None); if a list,
            feed No. in this list will be selected. Default (0, None) select all.
        corr : 'all', 'auto' or 'cross', optional
            Correlation type. 'auto' for auto-correlations, 'cross' for
            cross-correlations, 'all' for all correlations. Default 'all'.

        """
        # get feed info from the first input file
        feedno = self.infiles[0]['feedno'][:].tolist()

        if isinstance(value, tuple):
            feeds = feedno[slice(*value)]
        elif isinstance(value, list):
            feeds = np.intersect1d(feedno, value)
        else:
            raise ValueError('Unsupported data selection %s' % value)

        # get channo info from the first input file
        channo = self.infiles[0]['channo'][:]
        # get corresponding channel_pairs
        channel_pairs = []
        if corr == 'auto':
            for fd in feeds:
                ch1, ch2 = channo[feedno.index(fd)]
                channel_pairs += [ {ch1}, {ch2}, {ch1, ch2} ]
        elif corr == 'cross':
            for fd1, fd2 in itertools.combinations(feeds, 2):
                ch1, ch2 = channo[feedno.index(fd1)]
                ch3, ch4 = channo[feedno.index(fd2)]
                channel_pairs += [ {ch1, ch3}, {ch1, ch4}, {ch2, ch3}, {ch2, ch4} ]
        elif corr == 'all':
            for fd1, fd2 in itertools.combinations_with_replacement(feeds, 2):
                ch1, ch2 = channo[feedno.index(fd1)]
                ch3, ch4 = channo[feedno.index(fd2)]
                channel_pairs += [ {ch1, ch3}, {ch1, ch4}, {ch2, ch3}, {ch2, ch4} ]
        else:
            raise ValueError('Unknown correlation type %s' % corr)

        # get blorder info from the first input file
        blorder = self.infiles[0]['blorder']
        blorder = [ set(bl) for bl in blorder ]

        # channel pair indices
        indices = { blorder.index(chp) for chp in channel_pairs }
        indices = sorted(list(indices))

        self.data_select('channelpair', indices)


    def _load_a_common_dataset(self, name):
        ### load a common dataset from the first file
        # special care need take for blorder, just load the selected blorders
        if name == 'blorder':
            bl_dset = self.infiles[0][name]
            # main_data_select = self._main_data_select[:] # copy here to not change self._main_data_select
            bl_axis = self.main_data_axes.index('channelpair')
            # bl_select = self._main_data_select[bl_axis]
            tmp = np.arange(bl_dset.shape[0]) # number of channel pairs
            sel = tmp[self._main_data_select[bl_axis]]
            shp = (len(sel),) + bl_dset.shape[1:]
            # if channelpair is just the distributed axis, load blorder distributed
            if bl_axis == self.main_data_dist_axis:
                sel = mpiutil.mpilist(sel, method='con', comm=self.comm).tolist() # must have tolist as a single number numpy array index will reduce one axis in h5py slice
                self.create_dataset(name, shape=shp, dtype=bl_dset.dtype, distributed=True, distributed_axis=0)
                self[name].local_data[:] = bl_dset[sel]
            else:
                self.create_dataset(name, data=bl_dset[sel])
            # copy attrs of this dset
            memh5.copyattrs(bl_dset.attrs, self[name].attrs)
        else:
            super(RawTimestream, self)._load_a_common_dataset(name)
