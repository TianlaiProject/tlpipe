import itertools
import numpy as np
from datetime import datetime
import ephem
import container
from caput import memh5
from caput import mpiutil
from tlpipe.utils import date_util


class RawTimestream(container.BasicTod):
    """Container class for the raw timestream data.

    The raw timestream data are raw visibilities (the main data) and other data
    and meta data saved in HDF5 files which are recorded from the correlator.
    """

    _main_data_name = 'vis'
    _main_data_axes = ('time', 'frequency', 'channelpair')
    _main_time_ordered_datasets = ('vis', 'sec1970', 'jul_date')
    _time_ordered_datasets = _main_time_ordered_datasets + ('weather',)
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

    _feed_select = None
    _channel_select = None

    # def channelpair_select(self, value):
    #     """Select data to be loaded from input files along the channelpair axis.

    #     Parameters
    #     ----------
    #     value : tuple or list
    #         If a tuple, which will be created as a slice(start, stop, step) object,
    #         so it can have one to three elements (integers or None); if a list, its
    #         elements must be strictly increasing non-negative integers, data in
    #         these positions will be selected.

    #     """
    #     self.data_select('channelpair', value)

    # def channel_select(self, value=(0, None), corr='all'):
    #     """Select data to be loaded from inputs files corresponding to the specified channels.

    #     Parameters
    #     ----------
    #     value : tuple or list, optional
    #         If a tuple, which will be created as a slice(start, stop, step) object,
    #         so it can have one to three elements (integers or None); if a list,
    #         channel No. in this list will be selected. Default (0, None) select all.
    #     corr : 'all', 'auto' or 'cross', optional
    #         Correlation type. 'auto' for auto-correlations, 'cross' for
    #         cross-correlations, 'all' for all correlations. Default 'all'.

    #     """
    #     # get channo info from the first input file
    #     channo = self.infiles[0]['channo'][:]
    #     channo1d = np.sort(channo.flatten())

    #     if isinstance(value, tuple):
    #         channels = channo1d[slice(*value)]
    #     elif isinstance(value, list):
    #         channels = np.intersect1d(channo1d, value)
    #     else:
    #         raise ValueError('Unsupported data selection %s' % value)

    #     nchan = len(channels)
    #     # use set for easy comparison
    #     if corr == 'auto':
    #         channel_pairs = [ {channels[i]} for i in range(nchan) ]
    #     elif corr == 'cross':
    #         channel_pairs = [ {channels[i], channels[j]} for i in range(nchan) for j in range(i+1, nchan) ]
    #     elif corr == 'all':
    #         channel_pairs = [ {channels[i], channels[j]} for i in range(nchan) for j in range(i, nchan) ]
    #     else:
    #         raise ValueError('Unknown correlation type %s' % corr)

    #     # get blorder info from the first input file
    #     blorder = self.infiles[0]['blorder']
    #     blorder = [ set(bl) for bl in blorder ]

    #     # channel pair indices
    #     indices = { blorder.index(chp) for chp in channel_pairs }
    #     indices = sorted(list(indices))

    #     self.data_select('channelpair', indices)

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
            feeds = np.array(feedno[slice(*value)])
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

        self._feed_select = feeds
        self._channel_select = np.array([ channo[feedno.index(fd)] for fd in feeds ])


    def _load_a_common_dataset(self, name):
        ### load a common dataset from the first file
        # special care need take for blorder, just load the selected blorders
        if name == 'blorder':
            bl_dset = self.infiles[0][name]
            bl_axis = self.main_data_axes.index('channelpair')
            tmp = np.arange(bl_dset.shape[0]) # number of channel pairs
            sel = tmp[self._main_data_select[bl_axis]].tolist()
            shp = (len(sel),) + bl_dset.shape[1:]
            # if channelpair is just the distributed axis, load blorder distributed
            if bl_axis == self.main_data_dist_axis:
                sel = mpiutil.mpilist(sel, method='con', comm=self.comm)
                self.create_dataset(name, shape=shp, dtype=bl_dset.dtype, distributed=True, distributed_axis=0)
                self[name].local_data[:] = bl_dset[sel]
            else:
                self.create_dataset(name, data=bl_dset[sel])
            # copy attrs of this dset
            memh5.copyattrs(bl_dset.attrs, self[name].attrs)
        elif name == 'feedno' and not self._feed_select is None:
            self.create_dataset(name, data=self._feed_select)
        elif name == 'channo' and not self._channel_select is None:
            self.create_dataset(name, data=self._channel_select)
        elif name in ('polerr', 'feedpos', 'antpointing') and not self._feed_select is None:
            fh = self.infiles[0]
            feedno = fh['feedno'][:].tolist()
            feed_inds = [ feedno.index(fd) for fd in self._feed_select ]
            if name == 'antpointing':
                self.create_dataset(name, data=fh[name][:, feed_inds])
            else:
                self.create_dataset(name, data=fh[name][feed_inds])
        else:
            super(RawTimestream, self)._load_a_common_dataset(name)


    def load_common(self):
        """Load common attributes and datasets from the first file.

        This supposes that all common data are the same as that in the first file.
        """

        super(RawTimestream, self).load_common()

        # generate frequency points
        freq_start = self.attrs['freqstart']
        freq_step = self.attrs['freqstep']
        nfreq = self.attrs['nfreq']
        freqs = np.array([ freq_start + i*freq_step for i in range(nfreq)], dtype=np.float32)
        freq_axis = self.main_data_axes.index('frequency')
        sel_freqs = freqs[self._main_data_select[freq_axis]]
        shp = sel_freqs.shape
        # if frequency is just the distributed axis, load freqs distributed
        if freq_axis == self.main_data_dist_axis:
            sel_freqs = mpiutil.mpilist(sel_freqs, method='con', comm=self.comm)
            self.create_dataset('freq', shape=shp, dtype=sel_freqs.dtype, distributed=True, distributed_axis=0)
            self['freq'].local_data[:] = sel_freqs
        else:
            self.create_dataset('freq', data=sel_freqs)
        # create attrs of this dset
        self['freq'].attrs["unit"] = 'MHz'

    def load_tod_excl_main_data(self):
        """Load time ordered attributes and datasets (exclude the main data) from all files."""

        super(RawTimestream, self).load_tod_excl_main_data()

        # generate sec1970
        sec1970_first = self.infiles[0].attrs['sec1970']
        int_time = self.attrs['inttime']
        sec1970_start = sec1970_first + int_time * self.main_data_start
        num_sec1970 = self.main_data_stop - self.main_data_start
        sec1970 = np.array([ sec1970_start + i*int_time for i in range(num_sec1970)], dtype=np.float32)
        time_axis = self.main_data_axes.index('time')
        sel_sec1970 = sec1970[self._main_data_select[time_axis]]
        shp = sel_sec1970.shape
        # if time is just the distributed axis, load sec1970 distributed
        if time_axis == self.main_data_dist_axis:
            sel_sec1970 = mpiutil.mpilist(sel_sec1970, method='con', comm=self.comm)
            self.create_dataset('sec1970', shape=shp, dtype=sel_sec1970.dtype, distributed=True, distributed_axis=0)
            self['sec1970'].local_data[:] = sel_sec1970
        else:
            self.create_dataset('sec1970', data=sel_sec1970)
        # create attrs of this dset
        self['sec1970'].attrs["unit"] = 'second'

        # generate julian date
        jul_date = np.array([ date_util.get_juldate(datetime.fromtimestamp(s), tzone=self.attrs['timezone']) for s in self['sec1970'].local_data[:] ], dtype=np.float32)
        # if time is just the distributed axis, load jul_date distributed
        if time_axis == self.main_data_dist_axis:
            self.create_dataset('jul_date', shape=shp, dtype=jul_date.dtype, distributed=True, distributed_axis=0)
            self['jul_date'].local_data[:] = jul_date
        else:
            self.create_dataset('jul_date', data=jul_date)
        # create attrs of this dset
        self['jul_date'].attrs["unit"] = 'day'


    @property
    def time(self):
        """Return the jul_date dataset for convenient use."""
        try:
            return self.attrs['jul_date']
        except KeyError:
            raise KeyError('jul_date does not exist, try to load it first')

    @property
    def freq(self):
        """Return the freq dataset for convenient use."""
        try:
            return self.attrs['freq']
        except KeyError:
            raise KeyError('freq does not exist, try to load it first')


    def redistribute(self, dist_axis):
        """Redistribute the main time ordered dataset along a specified axis.

        This will redistribute the main_data along the specified axis `dis_axis`,
        and also distribute other main_time_ordered_datasets along the first axis
        if `dis_axis` is the first axis, else concatenate all those data along the
        first axis.

        Parameters
        ----------
        dist_axis : int, string
            The axis can be specified by an integer index (positive or
            negative), or by a string label which must correspond to an entry in
            the `main_data_axes` attribute on the dataset.

        """

        axis = container.check_axis(dist_axis, self.main_data_axes)

        if axis == self.main_data_dist_axis:
            # already the distributed axis, nothing to do
            return
        else:
            super(RawTimestream, self).redistribute(dist_axis)

            if 'time' == self.main_data_axes[axis]:
                self.dataset_distributed_to_common('freq')
                self.dataset_distributed_to_common('blorder')

            # distribute freq
            elif 'frequency' == self.main_data_axes[axis]:
                self.dataset_common_to_distributed('freq', distributed_axis=0)
                self.dataset_distributed_to_common('blorder')

            # distribute blorder
            elif 'channelpair' == self.main_data_axes[axis]:
                self.dataset_common_to_distributed('blorder', distributed_axis=0)
                self.dataset_distributed_to_common('freq')

    def check_status(self):
        """Check that data hold in this container is consistent. """

        # basic checks
        super(RawTimestream, self).check_status()

        # additional checks
        if 'frequency' == self.main_data_axes[self.main_data_dist_axis]:
            if not self['freq'].distributed:
                raise RuntimeError('Dataset freq should be distributed when frequency is the distributed axis')
            if not self['blorder'].common:
                raise RuntimeError('Dataset blorder should be common when frequency is the distributed axis')
        elif 'channelpair' == self.main_data_axes[self.main_data_dist_axis]:
            if not self['freq'].common:
                raise RuntimeError('Dataset freq should be common when channelpair is the distributed axis')
            if not self['blorder'].distributed:
                raise RuntimeError('Dataset blorder should be distributed when channelpair is the distributed axis')
