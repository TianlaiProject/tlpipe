import itertools
import numpy as np
from datetime import datetime
import container
import timestream
from caput import mpiarray
from caput import memh5
from caput import mpiutil
from tlpipe.utils import date_util


class RawTimestream(container.BasicTod):
    """Container class for the raw timestream data.

    The raw timestream data are raw visibilities (the main data) and other data
    and meta data saved in HDF5 files which are recorded from the correlator.

    Parameters
    ----------
    Same as :class:`container.BasicTod`.

    Attributes
    ----------
    time
    freq

    Methods
    -------
    time_select
    frequency_select
    feed_select
    load_common
    load_tod_excl_main_data
    redistribute
    check_status
    all_data_operate
    time_data_operate
    freq_data_operate
    bl_data_operate
    separate_pol_and_bl

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

        if value == (0, None) and corr == 'all':
            # select all, no need to do anything
            return

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


    def _load_a_special_common_dataset(self, name, axis_name):
        ### load a common dataset that need to take specail care
        ### this dataset need to distributed along axis_name is axis_name is just the self.main_data_dist_axis
        dset = self.infiles[0][name]
        axis = self.main_data_axes.index(axis_name)
        tmp = np.arange(dset.shape[0])
        sel = tmp[self._main_data_select[axis]].tolist()
        shp = (len(sel),) + dset.shape[1:] # the global shape
        # if axis_name is just the distributed axis, load dataset distributed
        if axis == self.main_data_dist_axis:
            sel = mpiutil.mpilist(sel, method='con', comm=self.comm)
            self.create_dataset(name, shape=shp, dtype=dset.dtype, distributed=True, distributed_axis=0)
            self[name].local_data[:] = dset[sel]
        else:
            self.create_dataset(name, data=dset[sel])
        # copy attrs of this dset
        memh5.copyattrs(dset.attrs, self[name].attrs)

    def _load_a_common_dataset(self, name):
        ### load a common dataset from the first file
        # special care need take for blorder, just load the selected blorders
        if name == 'freq':
            self._load_a_special_common_dataset(name, 'frequency')
        elif name == 'blorder':
            self._load_a_special_common_dataset(name, 'channelpair')
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

        if 'freq' not in self.iterkeys():
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

        if 'sec1970' not in self.iterkeys():
            # generate sec1970
            sec1970_first = self.infiles[0].attrs['sec1970']
            int_time = self.infiles[0].attrs['inttime']
            sec1970_start = sec1970_first + int_time * self.main_data_start
            num_sec1970 = self.main_data_stop - self.main_data_start
            sec1970 = np.array([ sec1970_start + i*int_time for i in range(num_sec1970)], dtype=np.float64) # precision float32 is not enough
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
            jul_date = np.array([ date_util.get_juldate(datetime.fromtimestamp(s), tzone=self.infiles[0].attrs['timezone']) for s in sel_sec1970 ], dtype=np.float64) # precision float32 is not enough
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
            return self['jul_date']
        except KeyError:
            raise KeyError('jul_date does not exist, try to load it first')

    @property
    def freq(self):
        """Return the freq dataset for convenient use."""
        try:
            return self['freq']
        except KeyError:
            raise KeyError('freq does not exist, try to load it first')

    @property
    def bl(self):
        """Return the blorder dataset for convenient use."""
        try:
            return self['blorder']
        except KeyError:
            raise KeyError('blorder does not exist, try to load it first')


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
        if 'time' == self.main_data_axes[self.main_data_dist_axis]:
            for name in ('freq', 'blorder'):
                if not self[name].common:
                    raise RuntimeError('Dataset %s should be common when time is the distributed axis' % name)
        elif 'frequency' == self.main_data_axes[self.main_data_dist_axis]:
            if not self['freq'].distributed:
                raise RuntimeError('Dataset freq should be distributed when frequency is the distributed axis')
            if not self['blorder'].common:
                raise RuntimeError('Dataset blorder should be common when frequency is the distributed axis')
        elif 'channelpair' == self.main_data_axes[self.main_data_dist_axis]:
            if not self['freq'].common:
                raise RuntimeError('Dataset freq should be common when channelpair is the distributed axis')
            if not self['blorder'].distributed:
                raise RuntimeError('Dataset blorder should be distributed when channelpair is the distributed axis')


    def all_data_operate(self, func, **kwargs):
        """Operation to the whole main data.

        Note since the main data is distributed on different processes, `func`
        should not have operations that depend on elements not held in the local
        array of each process

        Parameters
        ----------
        func : function object
            The opertation function object. It is of type func(array, **kwargs),
            which will operate on the array and return an new array with the same
            shape and dtype.
        **kwargs : any other arguments
            Any other arguments that will passed to `func`.

        """
        self.data_operate(func, op_axis=None, axis_vals=0, full_data=False, keep_dist_axis=False, **kwargs)

    def time_data_operate(self, func, full_data=False, keep_dist_axis=False, **kwargs):
        """Data operation along the time axis.

        Parameters
        ----------
        func : function object
            The opertation function object. It is of type func(array,
            local_index=None, global_index=None, jul_date=None, **kwargs), which
            will be called in a loop along the time axis.
        full_data : bool, optional
            Whether the operations of `func` will need the full data section
            corresponding to the axis index, if True, the main data will first
            redistributed along the time axis. Default False.
        keep_dist_axis : bool, optional
            Whether to redistribute main data to time axis if the dist axis has
            changed during the operation. Default False.
        **kwargs : any other arguments
            Any other arguments that will passed to `func`.

        """
        self.data_operate(func, op_axis='time', axis_vals=self.time.local_data[:], full_data=full_data, keep_dist_axis=keep_dist_axis, **kwargs)

    def freq_data_operate(self, func, full_data=False, keep_dist_axis=False, **kwargs):
        """Data operation along the frequency axis.

        Parameters
        ----------
        func : function object
            The opertation function object. It is of type func(array,
            local_index=None, global_index=None, freq=None, **kwargs), which
            will be called in a loop along the frequency axis.
        full_data : bool, optional
            Whether the operations of `func` will need the full data section
            corresponding to the axis index, if True, the main data will first
            redistributed along frequency time axis. Default False.
        keep_dist_axis : bool, optional
            Whether to redistribute main data to frequency axis if the dist axis has
            changed during the operation. Default False.
        **kwargs : any other arguments
            Any other arguments that will passed to `func`.

        """
        self.data_operate(func, op_axis='frequency', axis_vals=self.freq.local_data[:], full_data=full_data, keep_dist_axis=keep_dist_axis, **kwargs)

    def bl_data_operate(self, func, full_data=False, keep_dist_axis=False, **kwargs):
        """Data operation along the channelpair axis.

        Parameters
        ----------
        func : function object
            The opertation function object. It is of type func(array,
            local_index=None, global_index=None, chanpair=None, **kwargs), which
            will be called in a loop along the channelpair axis.
        full_data : bool, optional
            Whether the operations of `func` will need the full data section
            corresponding to the axis index, if True, the main data will first
            redistributed along channelpair time axis. Default False.
        keep_dist_axis : bool, optional
            Whether to redistribute main data to channelpair axis if the dist axis
            has changed during the operation. Default False.
        **kwargs : any other arguments
            Any other arguments that will passed to `func`.

        """
        self.data_operate(func, op_axis='channelpair', axis_vals=self.bl.local_data[:], full_data=full_data, keep_dist_axis=keep_dist_axis, **kwargs)


    def separate_pol_and_bl(self, keep_dist_axis=False):
        """Separate channelpair axis to polarization and baseline.

        This will create and return a Timestream container holding the polarization
        and baseline separated data.

        Parameters
        ----------
        keep_dist_axis : bool, optional
            Whether to redistribute main data to the original dist axis if the
            dist axis has changed during the operation. Default False.

        """

        # if dist axis is channelpair, redistribute it along time
        original_dist_axis = self.main_data_dist_axis
        if 'channelpair' == self.main_data_axes[original_dist_axis]:
            self.redistribute(0)

        # create a Timestream container to hold the pol and bl separated data
        ts = timestream.Timestream(dist_axis=self.main_data_dist_axis, comm=self.comm)

        feedno = sorted(self['feedno'][:].tolist())
        xchans = [ self['channo'][feedno.index(fd)][0] for fd in feedno ]
        ychans = [ self['channo'][feedno.index(fd)][1] for fd in feedno ]

        nfeed = len(feedno)
        xx_pairs = [ (xchans[i], xchans[j]) for i in range(nfeed) for j in range(i, nfeed) ]
        yy_pairs = [ (ychans[i], ychans[j]) for i in range(nfeed) for j in range(i, nfeed) ]
        xy_pairs = [ (xchans[i], ychans[j]) for i in range(nfeed) for j in range(i, nfeed) ]
        yx_pairs = [ (ychans[i], xchans[j]) for i in range(nfeed) for j in range(i, nfeed) ]

        blorder = [ tuple(bl) for bl in self['blorder'] ]
        conj_blorder = [ tuple(bl[::-1]) for bl in self['blorder'] ]

        def _get_ind(chp):
            try:
                return False, blorder.index(chp)
            except ValueError:
                return True, conj_blorder.index(chp)
        # xx
        xx_list = [ _get_ind(chp) for chp in xx_pairs ]
        xx_inds = [ ind for (cj, ind) in xx_list ]
        xx_conj = [ cj for (cj, ind) in xx_list ]
        # yy
        yy_list = [ _get_ind(chp) for chp in yy_pairs ]
        yy_inds = [ ind for (cj, ind) in yy_list ]
        yy_conj = [ cj for (cj, ind) in yy_list ]
        # xy
        xy_list = [ _get_ind(chp) for chp in xy_pairs ]
        xy_inds = [ ind for (cj, ind) in xy_list ]
        xy_conj = [ cj for (cj, ind) in xy_list ]
        # yx
        yx_list = [ _get_ind(chp) for chp in yx_pairs ]
        yx_inds = [ ind for (cj, ind) in yx_list ]
        yx_conj = [ cj for (cj, ind) in yx_list ]

        # create a MPIArray to hold the pol and bl separated vis
        shp = self.main_data.shape[:2] + (4, len(xx_inds))
        dtype = self.main_data.dtype
        md = mpiarray.MPIArray(shp, axis=self.main_data_dist_axis, comm=self.comm, dtype=dtype)
        # xx
        md.local_array[:, :, 0] = self.main_data.local_data[:, :, xx_inds].copy()
        md.local_array[:, :, 0] = np.where(xx_conj, md.local_array[:, :, 0].conj(), md.local_array[:, :, 0])
        # yy
        md.local_array[:, :, 1] = self.main_data.local_data[:, :, yy_inds].copy()
        md.local_array[:, :, 1] = np.where(yy_conj, md.local_array[:, :, 1].conj(), md.local_array[:, :, 1])
        # xy
        md.local_array[:, :, 2] = self.main_data.local_data[:, :, xy_inds].copy()
        md.local_array[:, :, 2] = np.where(xy_conj, md.local_array[:, :, 2].conj(), md.local_array[:, :, 2])
        # yx
        md.local_array[:, :, 3] = self.main_data.local_data[:, :, yx_inds].copy()
        md.local_array[:, :, 3] = np.where(yx_conj, md.local_array[:, :, 3].conj(), md.local_array[:, :, 3])

        # create main data
        ts.create_dataset(self.main_data_name, shape=shp, dtype=dtype, data=md, distributed=True, distributed_axis=self.main_data_dist_axis)
        # create attrs of this dataset
        ts.main_data.attrs['dimname'] = 'Time, Frequency, Polarization, Baseline'

        # create other datasets needed
        ts.create_dataset('pol', data=np.array(['xx', 'yy', 'xy', 'yx']))
        ts['pol'].attrs['pol_type'] = 'linear'
        blorder = np.array([ [feedno[i], feedno[j]] for i in range(nfeed) for j in range(i, nfeed) ])
        ts.create_dataset('blorder', data=blorder)

        # copy other attrs
        for attrs_name, attrs_value in self.attrs.iteritems():
            if attrs_name not in self.time_ordered_attrs:
                ts.attrs[attrs_name] = attrs_value
        # copy other datasets
        for dset_name, dset in self.iteritems():
            if not dset_name in (self.main_data_name, 'channo', 'blorder'):
                if dset.common:
                    ts.create_dataset(dset_name, data=dset)
                else:
                    ts.create_dataset(dset_name, data=dset, shape=dset.shape, dtype=dset.dtype, distributed=True, distributed_axis=dset.distributed_axis)
                # copy attrs of this dset
                memh5.copyattrs(dset.attrs, ts[dset_name].attrs)

        # redistribute self to original axis
        if keep_dist_axis:
            self.redistribute(original_dist_axis)

        return ts
