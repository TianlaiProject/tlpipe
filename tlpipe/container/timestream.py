"""Container class for the timestream data.


Inheritance diagram
-------------------

.. inheritance-diagram:: tlpipe.container.container.BasicTod tlpipe.container.timestream_common.TimestreamCommon tlpipe.container.raw_timestream.RawTimestream Timestream
   :parts: 2

"""

import itertools
import numpy as np
import container
import timestream_common
from caput import mpiarray
from caput import memh5


class Timestream(timestream_common.TimestreamCommon):
    """Container class for the timestream data.

    This timestream data container is to hold time stream data that has polarization
    and baseline separated from the channel pairs in the raw timestream.

    Parameters
    ----------
    Same as :class:`container.BasicTod`.

    """

    _main_data_name_ = 'vis'
    _main_data_axes_ = ('time', 'frequency', 'polarization', 'baseline')
    _main_axes_ordered_datasets_ = { 'vis': (0, 1, 2, 3),
                                     'vis_mask': (0, 1, 2, 3),
                                     'sec1970': (0,),
                                     'jul_date': (0,),
                                     'freq': (1,),
                                     'pol': (2,),
                                     'blorder': (3,),
                                   }
    _time_ordered_datasets_ = {'weather': (0,)}
    _time_ordered_attrs_ = {}
    _feed_ordered_datasets_ = { 'antpointing': (None, 0),
                                'feedno': (0,),
                                'feedpos': (0,),
                                'polerr': (0,),
                              }


    @property
    def pol_ordered_datasets(self):
        """Polarization ordered datasets."""
        return { key: val for key, val in self._main_axes_ordered_datasets_.items() if self.main_data_axes.index('polarization') in val }


    def polarization_select(self, value):
        """Select data to be loaded from input files along the polarization axis.

        Parameters
        ----------
        value : tuple or list
            If a tuple, which will be created as a slice(start, stop, step) object,
            so it can have one to three elements (integers or None); if a list, its
            elements must be strictly increasing non-negative integers, data in
            these positions will be selected.

        """
        self.data_select('polarization', value)

    def subset_polarization_select(self, value):
        """Select a subset of the data along the polarization axis.

        Parameters
        ----------
        value : tuple or list
            If a tuple, which will be created as a slice(start, stop, step) object,
            so it can have one to three elements (integers or None); if a list, its
            elements must be strictly increasing non-negative integers, data in
            these positions will be selected.

        """
        self.subset_select('polarization', value)

    def _inner_feed_select(self, data_source, value=(0, None), corr='all'):
        # inner method to select data corresponding to the specified feeds

        if value == (0, None) and corr == 'all':
            # select all, no need to do anything
            return

        # get feed info from data_source
        feedno = data_source['feedno'][:].tolist()

        if isinstance(value, tuple):
            feeds = np.array(feedno[slice(*value)])
        elif isinstance(value, list):
            feeds = np.intersect1d(feedno, value)
        else:
            raise ValueError('Unsupported data selection %s' % value)

        feeds = np.sort(feeds)
        if corr == 'auto':
            bls = [ {fd} for fd in feeds ]
        elif corr == 'cross':
            bls = [ {fd1, fd2} for fd1, fd2 in itertools.combinations(feeds, 2) ]
        elif corr == 'all':
            bls = [ {fd1, fd2} for fd1, fd2 in itertools.combinations_with_replacement(feeds, 2) ]
        else:
            raise ValueError('Unknown correlation type %s' % corr)

        # get blorder info from data_source
        if isinstance(data_source['blorder'], memh5.MemDatasetDistributed):
            blorder = data_source['blorder'].data.to_numpy_array(root=None)
        else:
            blorder = data_source['blorder']
        blorder = [ set(bl) for bl in blorder ]

        # baseline indices
        indices = { blorder.index(bl) for bl in bls }
        indices = sorted(list(indices))

        return indices, feeds

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

        results = self._inner_feed_select(self.infiles[0], value, corr)
        if results is not None:
            indices, feeds = results
            self.data_select('baseline', indices)
            self._feed_select = feeds

    def subset_feed_select(self, value=(0, None), corr='all'):
        """Select a subset of the data corresponding to the specified feeds.

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

        results = self._inner_feed_select(self, value, corr)
        if results is not None:
            indices, feeds = results
            self.subset_select('baseline', indices)
            self._subset_feed_select = feeds


    @property
    def pol(self):
        """Return the pol dataset for convenient use."""
        try:
            return self['pol']
        except KeyError:
            raise KeyError('pol does not exist, try to load it first')

    @property
    def local_pol(self):
        """A convenience for pol.local_data."""
        return self.pol.local_data


    def create_pol_ordered_dataset(self, name, data, axis_order=None, recreate=False, copy_attrs=False, check_align=True):
        """Create a polarization ordered dataset.

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
        check_align : bool, optional
            If True, check polarization axis of data align with that of the main data
            before dataset creating, otherwise create dataset without axis align
            checking, this may cause the created dataset does not align with the
            main data. Default True.

        """

        axis_order = axis_order or (self.main_data_axes.index('polarization'),)

        self.create_main_axis_ordered_dataset('polarization', name, data, axis_order, recreate, copy_attrs, check_align)


    def lin2stokes(self):
        """Convert the linear polarized data to Stokes polarization."""
        try:
            pol = self.pol
        except KeyError:
            raise RuntimeError('Polarization of the data is unknown, can not convert')

        if pol.attrs['pol_type'] == 'stokes' and pol.shape[0] == 4:
            warning.warn('Data is already Stokes polarization, no need to convert')
            return

        if pol.attrs['pol_type'] == 'linear' and pol.shape[0] == 4:

            # redistribute to 0 axis if polarization is the distributed axis
            original_dist_axis = self.main_data_dist_axis
            if 'polarization' == self.main_data_axes[self.main_data_dist_axis]:
                self.redistribute(0)

            pol = pol[:].tolist()
            p = self.pol_dict

            # create a new MPIArray to hold the new data
            md = mpiarray.MPIArray(self.main_data.shape, axis=self.main_data_dist_axis, comm=self.comm, dtype=self.main_data.dtype)
            # convert to Stokes I, Q, U, V
            md.local_array[:, :, 0] = 0.5 * (self.main_data.local_data[:, :, pol.index(p['xx'])] + self.main_data.local_data[:, :, pol.index(p['yy'])]) # I
            md.local_array[:, :, 1] = 0.5 * (self.main_data.local_data[:, :, pol.index(p['xx'])] - self.main_data.local_data[:, :, pol.index(p['yy'])]) # Q
            md.local_array[:, :, 2] = 0.5 * (self.main_data.local_data[:, :, pol.index(p['xy'])] + self.main_data.local_data[:, :, pol.index(p['yx'])]) # U
            md.local_array[:, :, 3] = -0.5J * (self.main_data.local_data[:, :, pol.index(p['xy'])] - self.main_data.local_data[:, :, pol.index(p['yx'])]) # V

            attr_dict = {} # temporarily save attrs of this dataset
            memh5.copyattrs(self.main_data.attrs, attr_dict)
            del self[self.main_data_name]
            # create main data
            self.create_dataset(self.main_data_name, shape=md.shape, dtype=md.dtype, data=md, distributed=True, distributed_axis=self.main_data_dist_axis)
            memh5.copyattrs(attr_dict, self.main_data.attrs)

            del self['pol']
            self.create_dataset('pol', data=np.array([p['I'], p['Q'], p['U'], p['V']]), dtype='i4')
            self['pol'].attrs['pol_type'] = 'stokes'

            # redistribute self to original axis
            self.redistribute(original_dist_axis)

        else:
            raise RuntimeError('Can not convert to Stokes polarization')

    def stokes2lin(self):
        """Convert the Stokes polarized data to linear polarization."""
        try:
            pol = self.pol
        except KeyError:
            raise RuntimeError('Polarization of the data is unknown, can not convert')

        if pol.attrs['pol_type'] == 'linear' and pol.shape[0] == 4:
            warning.warn('Data is already linear polarization, no need to convert')
            return

        if pol.attrs['pol_type'] == 'stokes' and pol.shape[0] == 4:

            # redistribute to 0 axis if polarization is the distributed axis
            original_dist_axis = self.main_data_dist_axis
            if 'polarization' == self.main_data_axes[self.main_data_dist_axis]:
                self.redistribute(0)

            pol = pol[:].tolist()
            p = self.pol_dict

            # create a new MPIArray to hold the new data
            md = mpiarray.MPIArray(self.main_data.shape, axis=self.main_data_dist_axis, comm=self.comm, dtype=self.main_data.dtype)
            # convert to linear xx, yy, xy, yx
            md.local_array[:, :, 0] = self.main_data.local_data[:, :, pol.index(p['I'])] + self.main_data.local_data[:, :, pol.index(p['Q'])] # xx
            md.local_array[:, :, 1] = self.main_data.local_data[:, :, pol.index(p['I'])] - self.main_data.local_data[:, :, pol.index(p['Q'])] # yy
            md.local_array[:, :, 2] = self.main_data.local_data[:, :, pol.index(p['U'])] + 1.0J * self.main_data.local_data[:, :, pol.index(p['V'])] # xy
            md.local_array[:, :, 3] = self.main_data.local_data[:, :, pol.index(p['U'])] - 1.0J * self.main_data.local_data[:, :, pol.index(p['V'])] # yx

            attr_dict = {} # temporarily save attrs of this dataset
            memh5.copyattrs(self.main_data.attrs, attr_dict)
            del self[self.main_data_name]
            # create main data
            self.create_dataset(self.main_data_name, shape=md.shape, dtype=md.dtype, data=md, distributed=True, distributed_axis=self.main_data_dist_axis)
            memh5.copyattrs(attr_dict, self.main_data.attrs)

            del self['pol']
            self.create_dataset('pol', data=np.array([p['xx'], p['yy'], p['xy'], p['yx']]), dtype='i4')
            self['pol'].attrs['pol_type'] = 'linear'

            # redistribute self to original axis
            self.redistribute(original_dist_axis)

        else:
            raise RuntimeError('Can not convert to linear polarization')


    def pol_data_operate(self, func, full_data=False, copy_data=False, show_progress=False, progress_step=None, keep_dist_axis=False, **kwargs):
        """Data operation along the polarization axis.

        Parameters
        ----------
        func : function object
            The opertation function object. It is of type func(vis, vis_mask,
            local_index, global_index, pol, self, \*\*kwargs), which
            will be called in a loop along the polarization axis.
        full_data : bool, optional
            Whether the operations of `func` will need the full data section
            corresponding to the axis index, if True, data will first
            redistributed along polarization axis. Default False.
        copy_data : bool, optional
            If True, `func` will operate on a copy of the data, so changes will
            have no impact on the data the container holds. Default False.
        show_progress : bool, optional
            If True, some progress info will show during the executing.
            Default False.
        progress_step : int or None
            Show progress info every this number of steps. If None, appropriate
            progress step will be chosen automatically. Default None.
        keep_dist_axis : bool, optional
            Whether to redistribute main data to the original axis if the dist
            axis has changed during the operation. Default False.
        \*\*kwargs : any other arguments
            Any other arguments that will passed to `func`.

        """
        self.data_operate(func, op_axis='polarization', axis_vals=self.pol, full_data=full_data, copy_data=copy_data, show_progress=show_progress, progress_step=progress_step, keep_dist_axis=keep_dist_axis, **kwargs)


    def time_and_pol_data_operate(self, func, full_data=False, copy_data=False, show_progress=False, progress_step=None, keep_dist_axis=False, **kwargs):
        """Data operation along the time and polarization axis.

        Parameters
        ----------
        func : function object
            The opertation function object. It is of type func(vis, vis_mask,
            local_index, global_index, tp, self, \*\*kwargs), which
            will be called in a loop along the time and polarization axis.
        full_data : bool, optional
            Whether the operations of `func` will need the full data section
            corresponding to the axis index, if True, data will first
            redistributed along time or polarization axis which is longer.
            Default False.
        copy_data : bool, optional
            If True, `func` will operate on a copy of the data, so changes will
            have no impact on the data the container holds. Default False.
        show_progress : bool, optional
            If True, some progress info will show during the executing.
            Default False.
        progress_step : int or None
            Show progress info every this number of steps. If None, appropriate
            progress step will be chosen automatically. Default None.
        keep_dist_axis : bool, optional
            Whether to redistribute main data to the original axis if the dist
            axis has changed during the operation. Default False.
        \*\*kwargs : any other arguments
            Any other arguments that will passed to `func`.

        """
        self.data_operate(func, op_axis=('time', 'polarization'), axis_vals=(self.time, self.pol), full_data=full_data, copy_data=copy_data, show_progress=show_progress, progress_step=progress_step, keep_dist_axis=keep_dist_axis, **kwargs)

    def freq_and_pol_data_operate(self, func, full_data=False, copy_data=False, show_progress=False, progress_step=None, keep_dist_axis=False, **kwargs):
        """Data operation along the frequency and polarization axis.

        Parameters
        ----------
        func : function object
            The opertation function object. It is of type func(vis, vis_mask,
            local_index, global_index, fp, self, \*\*kwargs), which
            will be called in a loop along the frequency and polarization axis.
        full_data : bool, optional
            Whether the operations of `func` will need the full data section
            corresponding to the axis index, if True, data will first
            redistributed along frequency or polarization axis which is longer.
            Default False.
        copy_data : bool, optional
            If True, `func` will operate on a copy of the data, so changes will
            have no impact on the data the container holds. Default False.
        show_progress : bool, optional
            If True, some progress info will show during the executing.
            Default False.
        progress_step : int or None
            Show progress info every this number of steps. If None, appropriate
            progress step will be chosen automatically. Default None.
        keep_dist_axis : bool, optional
            Whether to redistribute main data to the original axis if the dist
            axis has changed during the operation. Default False.
        \*\*kwargs : any other arguments
            Any other arguments that will passed to `func`.

        """
        self.data_operate(func, op_axis=('frequency', 'polarization'), axis_vals=(self.freq, self.pol), full_data=full_data, copy_data=copy_data, show_progress=show_progress, progress_step=progress_step, keep_dist_axis=keep_dist_axis, **kwargs)

    def pol_and_bl_data_operate(self, func, full_data=False, copy_data=False, show_progress=False, progress_step=None, keep_dist_axis=False, **kwargs):
        """Data operation along the polarization and baseline axis.

        Parameters
        ----------
        func : function object
            The opertation function object. It is of type func(vis, vis_mask,
            local_index, global_index, pbl, self, \*\*kwargs), which
            will be called in a loop along the polarization and baseline axis.
        full_data : bool, optional
            Whether the operations of `func` will need the full data section
            corresponding to the axis index, if True, data will first
            redistributed along polarization or baseline axis which is longer.
            Default False.
        copy_data : bool, optional
            If True, `func` will operate on a copy of the data, so changes will
            have no impact on the data the container holds. Default False.
        show_progress : bool, optional
            If True, some progress info will show during the executing.
            Default False.
        progress_step : int or None
            Show progress info every this number of steps. If None, appropriate
            progress step will be chosen automatically. Default None.
        keep_dist_axis : bool, optional
            Whether to redistribute main data to the original axis if the dist
            axis has changed during the operation. Default False.
        \*\*kwargs : any other arguments
            Any other arguments that will passed to `func`.

        """
        self.data_operate(func, op_axis=('polarization', 'baseline'), axis_vals=(self.pol, self.bl), full_data=full_data, copy_data=copy_data, show_progress=show_progress, progress_step=progress_step, keep_dist_axis=keep_dist_axis, **kwargs)

    def time_freq_and_pol_data_operate(self, func, full_data=False, copy_data=False, show_progress=False, progress_step=None, keep_dist_axis=False, **kwargs):
        """Data operation along the time, frequency and polarization axis.

        Parameters
        ----------
        func : function object
            The opertation function object. It is of type func(vis, vis_mask,
            local_index, global_index, tfp, self, \*\*kwargs), which
            will be called in a loop along the time, frequency and polarization
            axis.
        full_data : bool, optional
            Whether the operations of `func` will need the full data section
            corresponding to the axis index, if True, data will first
            redistributed along time or frequency or polarization axis which is
            longer. Default False.
        copy_data : bool, optional
            If True, `func` will operate on a copy of the data, so changes will
            have no impact on the data the container holds. Default False.
        show_progress : bool, optional
            If True, some progress info will show during the executing.
            Default False.
        progress_step : int or None
            Show progress info every this number of steps. If None, appropriate
            progress step will be chosen automatically. Default None.
        keep_dist_axis : bool, optional
            Whether to redistribute main data to the original axis if the dist
            axis has changed during the operation. Default False.
        \*\*kwargs : any other arguments
            Any other arguments that will passed to `func`.

        """
        self.data_operate(func, op_axis=('time', 'frequency', 'polarization'), axis_vals=(self.time, self.freq, self.pol), full_data=full_data, copy_data=copy_data, show_progress=show_progress, progress_step=progress_step, keep_dist_axis=keep_dist_axis, **kwargs)

    def time_freq_and_bl_data_operate(self, func, full_data=False, copy_data=False, show_progress=False, progress_step=None, keep_dist_axis=False, **kwargs):
        """Data operation along the time, frequency and baseline axis.

        Parameters
        ----------
        func : function object
            The opertation function object. It is of type func(vis, vis_mask,
            local_index, global_index, tfbl, self, \*\*kwargs), which
            will be called in a loop along the time, frequency and baseline
            axis.
        full_data : bool, optional
            Whether the operations of `func` will need the full data section
            corresponding to the axis index, if True, data will first
            redistributed along time or frequency or baseline axis which is
            longer. Default False.
        copy_data : bool, optional
            If True, `func` will operate on a copy of the data, so changes will
            have no impact on the data the container holds. Default False.
        show_progress : bool, optional
            If True, some progress info will show during the executing.
            Default False.
        progress_step : int or None
            Show progress info every this number of steps. If None, appropriate
            progress step will be chosen automatically. Default None.
        keep_dist_axis : bool, optional
            Whether to redistribute main data to the original axis if the dist
            axis has changed during the operation. Default False.
        \*\*kwargs : any other arguments
            Any other arguments that will passed to `func`.

        """
        self.data_operate(func, op_axis=('time', 'frequency', 'baseline'), axis_vals=(self.time, self.freq, self.bl), full_data=full_data, copy_data=copy_data, show_progress=show_progress, progress_step=progress_step, keep_dist_axis=keep_dist_axis, **kwargs)

    def time_pol_and_bl_data_operate(self, func, full_data=False, copy_data=False, show_progress=False, progress_step=None, keep_dist_axis=False, **kwargs):
        """Data operation along the time, polarization and baseline axis.

        Parameters
        ----------
        func : function object
            The opertation function object. It is of type func(vis, vis_mask,
            local_index, global_index, tpbl, self, \*\*kwargs), which
            will be called in a loop along the time, polarization and baseline
            axis.
        full_data : bool, optional
            Whether the operations of `func` will need the full data section
            corresponding to the axis index, if True, data will first
            redistributed along time or polarization or baseline axis which is
            longer. Default False.
        copy_data : bool, optional
            If True, `func` will operate on a copy of the data, so changes will
            have no impact on the data the container holds. Default False.
        show_progress : bool, optional
            If True, some progress info will show during the executing.
            Default False.
        progress_step : int or None
            Show progress info every this number of steps. If None, appropriate
            progress step will be chosen automatically. Default None.
        keep_dist_axis : bool, optional
            Whether to redistribute main data to the original axis if the dist
            axis has changed during the operation. Default False.
        \*\*kwargs : any other arguments
            Any other arguments that will passed to `func`.

        """
        self.data_operate(func, op_axis=('time', 'polarization', 'baseline'), axis_vals=(self.time, self.pol, self.bl), full_data=full_data, copy_data=copy_data, show_progress=show_progress, progress_step=progress_step, keep_dist_axis=keep_dist_axis, **kwargs)

    def freq_pol_and_bl_data_operate(self, func, full_data=False, copy_data=False, show_progress=False, progress_step=None, keep_dist_axis=False, **kwargs):
        """Data operation along the frequency, polarization and baseline axis.

        Parameters
        ----------
        func : function object
            The opertation function object. It is of type func(vis, vis_mask,
            local_index, global_index, fpbl, self, \*\*kwargs), which
            will be called in a loop along the frequency, polarization and
            baseline axis.
        full_data : bool, optional
            Whether the operations of `func` will need the full data section
            corresponding to the axis index, if True, data will first
            redistributed along frequency or polarization or baseline axis which
            is longer. Default False.
        copy_data : bool, optional
            If True, `func` will operate on a copy of the data, so changes will
            have no impact on the data the container holds. Default False.
        show_progress : bool, optional
            If True, some progress info will show during the executing.
            Default False.
        progress_step : int or None
            Show progress info every this number of steps. If None, appropriate
            progress step will be chosen automatically. Default None.
        keep_dist_axis : bool, optional
            Whether to redistribute main data to the original axis if the dist
            axis has changed during the operation. Default False.
        \*\*kwargs : any other arguments
            Any other arguments that will passed to `func`.

        """
        self.data_operate(func, op_axis=('frequency', 'polarization', 'baseline'), axis_vals=(self.freq, self.pol, self.bl), full_data=full_data, copy_data=copy_data, show_progress=show_progress, progress_step=progress_step, keep_dist_axis=keep_dist_axis, **kwargs)