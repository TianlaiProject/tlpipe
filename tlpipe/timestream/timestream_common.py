import numpy as np
from datetime import datetime
import container
from caput import mpiarray
from caput import memh5
from tlpipe.utils import date_util


class TimestreamCommon(container.BasicTod):
    """Common things for the raw timestream data and timestream data.

    This class hold the common data an operations for :class:`RawTimestream` and
     :class:`Timestream`. Usally you should not directly use this class, use
    :class:`RawTimestream` or :class:`Timestream` instead.

    Parameters
    ----------
    Same as :class:`container.BasicTod`.

    Attributes
    ----------
    time
    freq
    bl
    freq_ordered_datasets
    bl_ordered_datasets
    feed_ordered_datasets

    Methods
    -------
    time_select
    frequency_select
    feed_select
    load_common
    load_tod_excl_main_data
    create_freq_ordered_dataset
    create_bl_ordered_dataset
    create_feed_ordered_dataset
    redistribute
    check_status
    time_data_operate
    freq_data_operate
    bl_data_operate
    time_and_bl_data_operate
    freq_and_bl_data_operate

    """

    _main_data_name_ = 'vis'
    _main_data_axes_ = () # can be 'time', 'frequency', 'polarization', 'baseline'
    _main_axes_ordered_datasets_ = { 'vis': (0, 1, 2), # or (0, 1, 2, 3)
                                     'sec1970': (0,),
                                     'jul_date': (0,),
                                     'freq': (1,),
                                     'blorder': (2,), # 2 or 3
                                   }
    _time_ordered_datasets_ = {'weather': (0,)}
    _time_ordered_attrs_ = {'obstime', 'sec1970'}
    _feed_ordered_datasets_ = { 'antpointing': (None, 0),
                                'feedno': (0,),
                                'feedpos': (0,),
                                'polerr': (0,),
                              }


    @property
    def freq_ordered_datasets(self):
        """Frequency ordered datasets."""
        return { key: val for key, val in self._main_axes_ordered_datasets_.items() if self.main_data_axes.index('frequency') in val }

    # @freq_ordered_datasets.setter
    # def freq_ordered_datasets(self, value):
    #     if isinstance(value, basestring):
    #         self._freq_ordered_datasets_ = {value}
    #     elif hasattr(value, '__iter__'):
    #         for val in value:
    #             if not isinstance(val, basestring):
    #                 raise ValueError('Attribute freq_ordered_datasets must be a set of strings')
    #         self._freq_ordered_datasets_ = set(value)
    #     else:
    #         raise ValueError('Attribute freq_ordered_datasets must be a set of strings')

    @property
    def bl_ordered_datasets(self):
        """Baseline ordered datasets."""
        return { key: val for key, val in self._main_axes_ordered_datasets_.items() if self.main_data_axes.index('baseline') in val }

    # @bl_ordered_datasets.setter
    # def bl_ordered_datasets(self, value):
    #     if isinstance(value, basestring):
    #         self._bl_ordered_datasets_ = {value}
    #     elif hasattr(value, '__iter__'):
    #         for val in value:
    #             if not isinstance(val, basestring):
    #                 raise ValueError('Attribute bl_ordered_datasets must be a set of strings')
    #         self._bl_ordered_datasets_ = set(value)
    #     else:
    #         raise ValueError('Attribute bl_ordered_datasets must be a set of strings')

    @property
    def feed_ordered_datasets(self):
        """Feed ordered datasets."""
        return self._feed_ordered_datasets_

    # @feed_ordered_datasets.setter
    # def feed_ordered_datasets(self, value):
    #     if isinstance(value, basestring):
    #         self._feed_ordered_datasets_ = {value}
    #     elif hasattr(value, '__iter__'):
    #         for val in value:
    #             if not isinstance(val, basestring):
    #                 raise ValueError('Attribute feed_ordered_datasets must be a set of strings')
    #         self._feed_ordered_datasets_ = set(value)
    #     else:
    #         raise ValueError('Attribute feed_ordered_datasets must be a set of strings')


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

    # def baseline_select(self, value):
    #     """Select data to be loaded from input files along the baseline axis.

    #     Parameters
    #     ----------
    #     value : tuple or list
    #         If a tuple, which will be created as a slice(start, stop, step) object,
    #         so it can have one to three elements (integers or None); if a list, its
    #         elements must be strictly increasing non-negative integers, data in
    #         these positions will be selected.

    #     """
    #     self.data_select('baseline', value)


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

        raise NotImplementedError('feed_select not implemented in this base class')


    def _load_a_special_common_dataset(self, name, axis_name):
        ### load a common dataset that need to take specail care
        ### this dataset need to distributed along axis_name is axis_name is just the self.main_data_dist_axis
        dset = self.infiles[0][name]
        axis = self.main_data_axes.index(axis_name)
        tmp = np.arange(dset.shape[0])
        sel = tmp[self.main_data_select[axis]].tolist()
        shp = (len(sel),) + dset.shape[1:] # the global shape
        data = dset[sel]
        # if axis_name is just the distributed axis, load dataset distributed
        if axis == self.main_data_dist_axis:
            data  = mpiarray.MPIArray.from_numpy_array(data, axis=self.main_axes_ordered_datasets[name].index(axis))
        self.create_dataset(name, data=data)
        # copy attrs of this dset
        memh5.copyattrs(dset.attrs, self[name].attrs)

    def _load_a_common_dataset(self, name):
        ### load a common dataset from the first file
        # special care need take for blorder, just load the selected blorders
        if name in self.freq_ordered_datasets.keys():
            self._load_a_special_common_dataset(name, 'frequency')
        elif name in self.bl_ordered_datasets.keys():
            self._load_a_special_common_dataset(name, 'baseline')
        elif name == 'feedno' and not self._feed_select is None:
            self.create_dataset(name, data=self._feed_select)
            memh5.copyattrs(self.infiles[0][name].attrs, self[name].attrs)
        elif name in self.feed_ordered_datasets.keys() and not self._feed_select is None:
            fh = self.infiles[0]
            feedno = fh['feedno'][:].tolist()
            feed_inds = [ feedno.index(fd) for fd in self._feed_select ]
            feed_axis = self.feed_ordered_datasets[name].index(0)
            slc = [slice(0, None)] * (feed_axis + 1)
            slc[feed_axis] = feed_inds
            self.create_dataset(name, data=fh[name][tuple(slc)])
            # if name == 'antpointing':
            #     self.create_dataset(name, data=fh[name][:, feed_inds])
            # else:
            #     self.create_dataset(name, data=fh[name][feed_inds])
            memh5.copyattrs(self.infiles[0][name].attrs, self[name].attrs)
        else:
            super(TimestreamCommon, self)._load_a_common_dataset(name)


    def load_common(self):
        """Load common attributes and datasets from the first file.

        This supposes that all common data are the same as that in the first file.
        """

        super(TimestreamCommon, self).load_common()

        if 'freq' not in self.iterkeys():
            # generate frequency points
            freq_start = self.attrs['freqstart']
            freq_step = self.attrs['freqstep']
            nfreq = self.attrs['nfreq']
            freq = np.array([ freq_start + i*freq_step for i in range(nfreq)], dtype=np.float32)
            freq_axis = self.main_data_axes.index('frequency')
            freq = freq[self.main_data_select[freq_axis]]

            # if frequency is just the distributed axis, load freq distributed
            if 'frequency' == self.main_data_axes[self.main_data_dist_axis]:
                freq = mpiarray.MPIArray.from_numpy_array(freq)
            self.create_freq_ordered_dataset('freq', data=freq)
            # create attrs of this dset
            self['freq'].attrs["unit"] = 'MHz'

    def load_tod_excl_main_data(self):
        """Load time ordered attributes and datasets (exclude the main data) from all files."""

        super(TimestreamCommon, self).load_tod_excl_main_data()

        if 'sec1970' not in self.iterkeys():
            # generate sec1970
            sec1970_first = self.infiles[0].attrs['sec1970']
            int_time = self.infiles[0].attrs['inttime']
            sec1970_start = sec1970_first + int_time * self.main_data_start
            num_sec1970 = self.main_data_stop - self.main_data_start
            sec1970 = np.array([ sec1970_start + i*int_time for i in range(num_sec1970)], dtype=np.float64) # precision float32 is not enough
            time_axis = self.main_data_axes.index('time')
            sec1970 = sec1970[self.main_data_select[time_axis]]

            # if time is just the distributed axis, load sec1970 distributed
            if 'time' == self.main_data_axes[self.main_data_dist_axis]:
                sec1970 = mpiarray.MPIArray.from_numpy_array(sec1970)
            self.create_main_time_ordered_dataset('sec1970', data=sec1970)
            # create attrs of this dset
            self['sec1970'].attrs["unit"] = 'second'

            # generate julian date
            jul_date = np.array([ date_util.get_juldate(datetime.fromtimestamp(s), tzone=self.infiles[0].attrs['timezone']) for s in sec1970 ], dtype=np.float64) # precision float32 is not enough
            if 'time' == self.main_data_axes[self.main_data_dist_axis]:
                jul_date = mpiarray.MPIArray.wrap(jul_date, axis=0)
            # if time is just the distributed axis, load jul_date distributed
            self.create_main_time_ordered_dataset('jul_date', data=jul_date)
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


    # def _create_axis_ordered_dataset(self, axis_name, name, data, axis_order, recreate=False, copy_attrs=False):
    #     """Create a `axis_name` ordered dataset.

    #     Parameters
    #     ----------
    #     axis_name : string
    #         Name of the axis.
    #     name : string
    #         Name of the dataset.
    #     data : np.ndarray or MPIArray
    #         The data to create a dataset.
    #     axis_order : tuple
    #         A tuple denotes the corresponding axis of the created dataset.
    #     recreate : bool, optional
    #         If True will recreate a dataset with this name if it already exists,
    #         else a RuntimeError will be rasised. Default False.
    #     copy_attrs : bool, optional
    #         If True, when recreate the dataset, its original attributes will be
    #         copyed to the new dataset, else no copy is done. Default Fasle.

    #     """

    #     if not axis_name in self.main_data_axes:
    #         raise ValueError('Invalid axis name %s for axes %s' % (axis_name, self.main_data_axes))
    #     if axis_name == self.main_data_axes[0]:
    #         raise ValueError('Use create_time_ordered_dataset or create_main_time_ordered_dataset instead')

    #     if isinstance(data, mpiarray.MPIArray):
    #         shape = data.global_shape
    #     else:
    #         shape = data.shape
    #     axis = axis_order.index(self.main_data_axes.index(axis_name))
    #     if shape[axis] != self.main_data.shape[self.main_data_axes.index(axis_name)]:
    #         raise ValueError('%s axis does not align with main data, can not create a %s ordered dataset %s' % (axis_name.capitalize(), axis_name, name))

    #     if not name in self.iterkeys():
    #         if axis_name == self.main_data_axes[self.main_data_dist_axis]:
    #             self.create_dataset(name, data=data, distributed=True, distributed_axis=axis)
    #         else:
    #             self.create_dataset(name, data=data)
    #     else:
    #         if recreate:
    #             if copy_attrs:
    #                 attr_dict = {} # temporarily save attrs of this dataset
    #                 copyattrs(self[name].attrs, attr_dict)
    #             del self[name]
    #             if axis_name == self.main_data_axes[self.main_data_dist_axis]:
    #                 self.create_dataset(name, data=data, distributed=True, distributed_axis=axis)
    #             else:
    #                 self.create_dataset(name, data=data)
    #             if copy_attrs:
    #                 copyattrs(attr_dict, self[name].attrs)
    #         else:
    #             raise RuntimeError('Dataset %s already exists' % name)


    def create_freq_ordered_dataset(self, name, data, axis_order=None, recreate=False, copy_attrs=False):
        """Create a frequency ordered dataset.

        Parameters
        ----------
        name : string
            Name of the dataset.
        data : np.ndarray or MPIArray
            The data to create a dataset.
        axis_order : tuple
            A tuple denotes frequency axis of the created dataset.
        recreate : bool, optional
            If True will recreate a dataset with this name if it already exists,
            else a RuntimeError will be rasised. Default False.
        copy_attrs : bool, optional
            If True, when recreate the dataset, its original attributes will be
            copyed to the new dataset, else no copy is done. Default Fasle.

        """

        axis_order = axis_order or (self.main_data_axes.index('frequency'),)

        self.create_main_axis_ordered_dataset('frequency', name, data, axis_order, recreate, copy_attrs)

        # self.freq_ordered_datasets[name] = axis_order

    def create_bl_ordered_dataset(self, name, data, axis_order=None, recreate=False, copy_attrs=False):
        """Create a baseline ordered dataset.

        Parameters
        ----------
        name : string
            Name of the dataset.
        data : np.ndarray or MPIArray
            The data to create a dataset.
        axis_order : tuple
            A tuple denotes baseline axis of the created dataset.
        recreate : bool, optional
            If True will recreate a dataset with this name if it already exists,
            else a RuntimeError will be rasised. Default False.
        copy_attrs : bool, optional
            If True, when recreate the dataset, its original attributes will be
            copyed to the new dataset, else no copy is done. Default Fasle.

        """

        axis_order = axis_order or (self.main_data_axes.index('baseline'),)

        self.create_main_axis_ordered_dataset('baseline', name, data, axis_order, recreate, copy_attrs)

        # self.bl_ordered_datasets[name] = axis_order

    def create_feed_ordered_dataset(self, name, data, axis_order=(0,), recreate=False, copy_attrs=False):
        """Create a feed ordered dataset.

        Parameters
        ----------
        name : string
            Name of the dataset.
        data : np.ndarray
            The data to create a dataset.
        axis_order : tuple
            A tuple with the index 0 denotes feed axis of the created dataset.
        recreate : bool, optional
            If True will recreate a dataset with this name if it already exists,
            else a RuntimeError will be rasised. Default False.
        copy_attrs : bool, optional
            If True, when recreate the dataset, its original attributes will be
            copyed to the new dataset, else no copy is done. Default Fasle.

        """

        shape = data.shape
        feed_axis = axis_order.index(0)
        if 'feedno' in self.iterkeys() and shape[feed_axis] != self['feedno'].shape[0]:
            raise ValueError('Feed axis does not align with feedno, can not create a feed ordered dataset %s' % name)

        if not name in self.iterkeys():
            self.create_dataset(name, data=data)
        else:
            if recreate:
                if copy_attrs:
                    attr_dict = {} # temporarily save attrs of this dataset
                    copyattrs(self[name].attrs, attr_dict)
                del self[name]
                self.create_dataset(name, data=data)
                if copy_attrs:
                    copyattrs(attr_dict, self[name].attrs)
            else:
                raise RuntimeError('Dataset %s already exists' % name)

        self.feed_ordered_datasets[name] = axis_order


    # def redistribute(self, dist_axis):
    #     """Redistribute the main time ordered dataset along a specified axis.

    #     This will redistribute the main_data along the specified axis `dis_axis`,
    #     and also distribute other main_time_ordered_datasets along the first axis
    #     if `dis_axis` is the first axis, else concatenate all those data along the
    #     first axis.

    #     Parameters
    #     ----------
    #     dist_axis : int, string
    #         The axis can be specified by an integer index (positive or
    #         negative), or by a string label which must correspond to an entry in
    #         the `main_data_axes` attribute on the dataset.

    #     """

    #     axis = container.check_axis(dist_axis, self.main_data_axes)

    #     if axis == self.main_data_dist_axis:
    #         # already the distributed axis, nothing to do
    #         return
    #     else:
    #         super(TimestreamCommon, self).redistribute(dist_axis)

    #         if 'time' == self.main_data_axes[axis]:
    #             for name in set(self.freq_ordered_datasets.keys()) | (self.bl_ordered_datasets.keys()):
    #                 if name in self.iterkeys() and self[name].distributed:
    #                     self.dataset_distributed_to_common(name)

    #         # distribute freq
    #         elif 'frequency' == self.main_data_axes[axis]:
    #             for name in self.freq_ordered_datasets.keys():
    #                 if name in self.iterkeys() and self[name].common:
    #                     self.dataset_common_to_distributed(name, distributed_axis=self.main_axes_ordered_datasets[name].index(self.main_data_axes.index('frequency')))
    #             for name in self.bl_ordered_datasets.keys():
    #                 if name in self.iterkeys() and self[name].distributed:
    #                     self.dataset_distributed_to_common(name)

    #         # distribute blorder
    #         elif 'baseline' == self.main_data_axes[axis]:
    #             for name in self.bl_ordered_datasets.keys():
    #                 if name in self.iterkeys() and self[name].common:
    #                     self.dataset_common_to_distributed(name, distributed_axis=self.main_axes_ordered_datasets[name].index(self.main_data_axes.index('baseline')))
    #             for name in self.freq_ordered_datasets.keys():
    #                 if name in self.iterkeys() and self[name].distributed:
    #                     self.dataset_distributed_to_common(name)

    def check_status(self):
        """Check that data hold in this container is consistent. """

        # basic checks
        super(TimestreamCommon, self).check_status()

        # additional checks
        # if 'time' == self.main_data_axes[self.main_data_dist_axis]:
        #     for name in set(self.freq_ordered_datasets.keys()) | set(self.bl_ordered_datasets.keys()):
        #         if name in self.iterkeys() and not self[name].common:
        #             raise RuntimeError('Dataset %s should be common when time is the distributed axis' % name)
        # elif 'frequency' == self.main_data_axes[self.main_data_dist_axis]:
        #     for name in self.freq_ordered_datasets.keys():
        #         if name in self.iterkeys() and not self[name].distributed:
        #             raise RuntimeError('Dataset %s should be distributed when frequency is the distributed axis' % name)
        #     for name in set(self.time_ordered_datasets.keys()) | set(self.bl_ordered_datasets.keys()):
        #         if name in self.iterkeys() and not self[name].common:
        #             raise RuntimeError('Dataset %s should be common when frequency is the distributed axis' % name)
        # elif 'baseline' == self.main_data_axes[self.main_data_dist_axis]:
        #     for name in self.bl_ordered_datasets:
        #         if name in self.iterkeys() and not self[name].distributed:
        #             raise RuntimeError('Dataset %s should be distributed when baseline is the distributed axis' % name)
        #     for name in self.time_ordered_datasets | self.freq_ordered_datasets:
        #         if name in self.iterkeys() and not self[name].common:
        #             raise RuntimeError('Dataset %s should be common when baseline is the distributed axis' % name)

        # additional checks for feed_ordered_datasets
        lens = []
        for name, val in self.feed_ordered_datasets.items():
            if name in self.items():
                lens.append(self[name].shape[val.index(0)])
        num = len(set(lens))
        if num != 0 and num != 1:
            raise RuntimeError('Not all feed_ordered_datasets have an aligned feed axis')


    def time_data_operate(self, func, full_data=False, keep_dist_axis=False, **kwargs):
        """Data operation along the time axis.

        Parameters
        ----------
        func : function object
            The opertation function object. It is of type func(array,
            local_index, global_index, jul_date, self, **kwargs), which
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
        self.data_operate(func, op_axis='time', axis_vals=self.time, full_data=full_data, keep_dist_axis=keep_dist_axis, **kwargs)

    def freq_data_operate(self, func, full_data=False, keep_dist_axis=False, **kwargs):
        """Data operation along the frequency axis.

        Parameters
        ----------
        func : function object
            The opertation function object. It is of type func(array,
            local_index=, global_index, freq, self, **kwargs), which
            will be called in a loop along the frequency axis.
        full_data : bool, optional
            Whether the operations of `func` will need the full data section
            corresponding to the axis index, if True, the main data will first
            redistributed along frequency axis. Default False.
        keep_dist_axis : bool, optional
            Whether to redistribute main data to frequency axis if the dist axis has
            changed during the operation. Default False.
        **kwargs : any other arguments
            Any other arguments that will passed to `func`.

        """
        self.data_operate(func, op_axis='frequency', axis_vals=self.freq, full_data=full_data, keep_dist_axis=keep_dist_axis, **kwargs)

    def bl_data_operate(self, func, full_data=False, keep_dist_axis=False, **kwargs):
        """Data operation along the baseline axis.

        Parameters
        ----------
        func : function object
            The opertation function object. It is of type func(array,
            local_index, global_index, chanpair, self, **kwargs), which
            will be called in a loop along the baseline axis.
        full_data : bool, optional
            Whether the operations of `func` will need the full data section
            corresponding to the axis index, if True, the main data will first
            redistributed along baseline axis. Default False.
        keep_dist_axis : bool, optional
            Whether to redistribute main data to baseline axis if the dist axis
            has changed during the operation. Default False.
        **kwargs : any other arguments
            Any other arguments that will passed to `func`.

        """
        self.data_operate(func, op_axis='baseline', axis_vals=self.bl, full_data=full_data, keep_dist_axis=keep_dist_axis, **kwargs)

    def time_and_bl_data_operate(self, func, full_data=False, keep_dist_axis=False, **kwargs):
        """Data operation along the time and baseline axis.

        Parameters
        ----------
        func : function object
            The opertation function object. It is of type func(array,
            local_index, global_index, tbl, self, **kwargs), which
            will be called in a loop along the time and baseline axis.
        full_data : bool, optional
            Whether the operations of `func` will need the full data section
            corresponding to the axis index, if True, the main data will first
            redistributed along time or baseline axis which is longer.
            Default False.
        keep_dist_axis : bool, optional
            Whether to redistribute main data to baseline axis if the dist axis
            has changed during the operation. Default False.
        **kwargs : any other arguments
            Any other arguments that will passed to `func`.

        """
        self.data_operate(func, op_axis=('time', 'baseline'), axis_vals=(self.time, self.bl), full_data=full_data, keep_dist_axis=keep_dist_axis, **kwargs)

    def freq_and_bl_data_operate(self, func, full_data=False, keep_dist_axis=False, **kwargs):
        """Data operation along the frequency and baseline axis.

        Parameters
        ----------
        func : function object
            The opertation function object. It is of type func(array,
            local_index, global_index, fbl, self, **kwargs), which
            will be called in a loop along the frequency and baseline axis.
        full_data : bool, optional
            Whether the operations of `func` will need the full data section
            corresponding to the axis index, if True, the main data will first
            redistributed along frequency or baseline axis which is longer.
            Default False.
        keep_dist_axis : bool, optional
            Whether to redistribute main data to baseline axis if the dist axis
            has changed during the operation. Default False.
        **kwargs : any other arguments
            Any other arguments that will passed to `func`.

        """
        self.data_operate(func, op_axis=('frequency', 'baseline'), axis_vals=(self.freq, self.bl), full_data=full_data, keep_dist_axis=keep_dist_axis, **kwargs)
