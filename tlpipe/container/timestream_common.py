"""Common things for the raw timestream data and timestream data.


Inheritance diagram
-------------------

.. inheritance-diagram:: tlpipe.container.container.BasicTod TimestreamCommon tlpipe.container.raw_timestream.RawTimestream tlpipe.container.timestream.Timestream
   :parts: 2

"""

import itertools
import numpy as np
import ephem
from datetime import datetime
import container
from caput import mpiutil
from caput import mpiarray
from caput import memh5
from tlpipe.core import tl_array
from tlpipe.core import constants as const
from tlpipe.utils import date_util
from tlpipe.utils import progress


class TimestreamCommon(container.BasicTod):
    """Common things for the raw timestream data and timestream data.

    This class hold the common data an operations for :class:`RawTimestream` and
    :class:`Timestream`. Usally you should not directly use this class, use
    :class:`RawTimestream` or :class:`Timestream` instead.

    Parameters
    ----------
    Same as :class:`container.BasicTod`.

    """

    _main_data_name_ = 'vis'
    _main_data_axes_ = () # can be 'time', 'frequency', 'polarization', 'baseline'
    _main_axes_ordered_datasets_ = { 'vis': (0, 1, 2), # or (0, 1, 2, 3)
                                     'vis_mask': (0, 1, 2), # or (0, 1, 2, 3)
                                     'sec1970': (0,),
                                     'jul_date': (0,),
                                     'local_hour': (0,),
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

    @property
    def bl_ordered_datasets(self):
        """Baseline ordered datasets."""
        return { key: val for key, val in self._main_axes_ordered_datasets_.items() if self.main_data_axes.index('baseline') in val }

    @property
    def feed_ordered_datasets(self):
        """Feed ordered datasets."""
        return self._feed_ordered_datasets_


    _pol_dict = {0: 'xx', 1: 'yy', 2: 'xy', 3: 'yx', 4: 'I', 5: 'Q', 6: 'U', 7: 'V',
                 'xx': 0, 'yy': 1, 'xy': 2, 'yx': 3, 'I': 4, 'Q': 5, 'U': 6, 'V':7}

    @property
    def pol_dict(self):
        "A dict maps polarization code to integer."
        return self._pol_dict


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

    def subset_time_select(self, value):
        """Select a subset of the data along the time axis.

        Parameters
        ----------
        value : tuple or list
            If a tuple, which will be created as a slice(start, stop, step) object,
            so it can have one to three elements (integers or None); if a list, its
            elements must be strictly increasing non-negative integers, data in
            these positions will be selected.

        """
        self.subset_select('time', value)

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

    def subset_frequency_select(self, value):
        """Select a subset of the data along the frequency axis.

        Parameters
        ----------
        value : tuple or list
            If a tuple, which will be created as a slice(start, stop, step) object,
            so it can have one to three elements (integers or None); if a list, its
            elements must be strictly increasing non-negative integers, data in
            these positions will be selected.

        """
        self.subset_select('frequency', value)

    _feed_select = None
    _subset_feed_select = None

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

    # def subset_baseline_select(self, value):
    #     """Select a subset of the data along the baseline axis.

    #     Parameters
    #     ----------
    #     value : tuple or list
    #         If a tuple, which will be created as a slice(start, stop, step) object,
    #         so it can have one to three elements (integers or None); if a list, its
    #         elements must be strictly increasing non-negative integers, data in
    #         these positions will be selected.

    #     """
    #     self.subset_select('baseline', value)


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

        raise NotImplementedError('subset_feed_select not implemented in this base class')


    def _load_a_common_dataset(self, name):
        ### load a common dataset from the first file
        if name == 'feedno' and not self._feed_select is None:
            self.create_dataset(name, data=self._feed_select)
            memh5.copyattrs(self.infiles[0][name].attrs, self[name].attrs)
        elif name in self.feed_ordered_datasets.keys() and not self._feed_select is None:
            fh = self.infiles[0]
            feedno = fh['feedno'][:].tolist()
            feed_inds = [ feedno.index(fd) for fd in self._feed_select ]
            feed_axis = self.feed_ordered_datasets[name].index(0)
            slc = [slice(0, None)] * len(fh[name].shape)
            slc[feed_axis] = feed_inds
            self.create_dataset(name, data=fh[name][tuple(slc)])
            memh5.copyattrs(fh[name].attrs, self[name].attrs)
        else:
            super(TimestreamCommon, self)._load_a_common_dataset(name)


    def load_all(self):
        """Load all attributes and datasets from files."""

        super(TimestreamCommon, self).load_all()

        # create some new necessary datasets is they do not already exist in file
        if 'vis_mask' not in self.iterkeys():
            # create the mask array
            vis_mask = np.where(np.isfinite(self.local_vis), False, True)
            vis_mask = mpiarray.MPIArray.wrap(vis_mask, axis=self.main_data_dist_axis)
            axis_order = self.main_axes_ordered_datasets[self.main_data_name]
            vis_mask = self.create_main_axis_ordered_dataset(axis_order, 'vis_mask', vis_mask, axis_order)

        if 'freq' not in self.iterkeys():
            # generate frequency points
            freq_start = self.attrs['freqstart']
            freq_step = self.attrs['freqstep']
            nfreq = self.attrs['nfreq']
            freq = np.array([ freq_start + i*freq_step for i in xrange(nfreq)], dtype=np.float32)
            freq_axis = self.main_data_axes.index('frequency')
            freq = freq[self.main_data_select[freq_axis]]

            # if frequency is just the distributed axis, load freq distributed
            if 'frequency' == self.main_data_axes[self.main_data_dist_axis]:
                freq = mpiarray.MPIArray.from_numpy_array(freq)
            self.create_freq_ordered_dataset('freq', data=freq)
            # create attrs of this dset
            self['freq'].attrs["unit"] = 'MHz'

        if 'sec1970' not in self.iterkeys():
            # generate sec1970
            int_time = self.infiles[0].attrs['inttime']
            sec1970s = []
            nts = []
            for fh in mpiutil.mpilist(self.infiles, method='con', comm=self.comm):
                sec1970s.append(fh.attrs['sec1970'])
                nts.append(fh[self.main_data_name].shape[0])
            sec1970 = np.zeros(sum(nts), dtype=np.float64) # precision float32 is not enough
            cum_nts = np.cumsum([0] + nts)
            for idx, (nt, sec) in enumerate(zip(nts, sec1970s)):
                sec1970[cum_nts[idx]:cum_nts[idx+1]] = np.array([ sec + i*int_time for i in xrange(nt)], dtype=np.float64) # precision float32 is not enough
            # gather local sec1970
            sec1970 = mpiutil.gather_array(sec1970, root=None, comm=self.comm)
            # select the corresponding section
            sec1970 = sec1970[self.main_data_start:self.main_data_stop][self.main_data_select[0]]

            # if time is just the distributed axis, load sec1970 distributed
            if 'time' == self.main_data_axes[self.main_data_dist_axis]:
                sec1970 = mpiarray.MPIArray.from_numpy_array(sec1970)
            self.create_main_time_ordered_dataset('sec1970', data=sec1970)
            # create attrs of this dset
            self['sec1970'].attrs["unit"] = 'second'
            # determine if it is continuous in time
            sec_diff = np.diff(sec1970)
            break_inds = np.where(sec_diff>1.5*int_time)[0]
            if len(break_inds) > 0:
                self['sec1970'].attrs["continuous"] = False
                self['sec1970'].attrs["break_inds"] = break_inds + 1
            else:
                self['sec1970'].attrs["continuous"] = True

            # generate julian date
            jul_date = np.array([ date_util.get_juldate(datetime.fromtimestamp(s), tzone=self.infiles[0].attrs['timezone']) for s in sec1970 ], dtype=np.float64) # precision float32 is not enough
            if 'time' == self.main_data_axes[self.main_data_dist_axis]:
                jul_date = mpiarray.MPIArray.wrap(jul_date, axis=0)
            # if time is just the distributed axis, load jul_date distributed
            self.create_main_time_ordered_dataset('jul_date', data=jul_date)
            # create attrs of this dset
            self['jul_date'].attrs["unit"] = 'day'

            # generate local time in hour from 0 to 24.0
            def _hour(t):
                return t.hour + t.minute/60.0 + t.second/3600.0 + t.microsecond/3.6e8
            local_hour = np.array([ _hour(datetime.fromtimestamp(s).time()) for s in sec1970 ], dtype=np.float64)
            if 'time' == self.main_data_axes[self.main_data_dist_axis]:
                local_hour = mpiarray.MPIArray.wrap(local_hour, axis=0)
            # if time is just the distributed axis, load local_hour distributed
            self.create_main_time_ordered_dataset('local_hour', data=local_hour)
            # create attrs of this dset
            self['local_hour'].attrs["unit"] = 'hour'

            # generate az, alt
            az_alt = np.zeros((self['sec1970'].local_data.shape[0], 2), dtype=np.float32) # radians
            if self.is_dish:
                # antpointing = rt['antpointing'][-1, :, :] # degree
                # pointingtime = rt['pointingtime'][-1, :, :] # degree
                az_alt[:, 0] = 0.0 # az
                az_alt[:, 1] = np.pi/2 # alt
            elif self.is_cylinder:
                az_alt[:, 0] = np.pi/2 # az
                az_alt[:, 1] = np.pi/2 # alt
            else:
                raise RuntimeError('Unknown antenna type %s' % self.attrs['telescope'])

            # generate ra, dec of the antenna pointing
            aa = self.array
            ra_dec = np.zeros_like(az_alt) # radians
            for ti in xrange(az_alt.shape[0]):
                az, alt = az_alt[ti]
                az, alt = ephem.degrees(az), ephem.degrees(alt)
                aa.set_jultime(self['jul_date'].local_data[ti])
                ra_dec[ti] = aa.radec_of(az, alt) # in radians, a point in the sky above the observer

            if self.main_data_dist_axis == 0:
                az_alt = mpiarray.MPIArray.wrap(az_alt, axis=0)
                ra_dec = mpiarray.MPIArray.wrap(ra_dec, axis=0)
            # if time is just the distributed axis, create distributed datasets
            self.create_main_time_ordered_dataset('az_alt', data=az_alt)
            self['az_alt'].attrs['unit'] = 'radian'
            self.create_main_time_ordered_dataset('ra_dec', data=ra_dec)
            self['ra_dec'].attrs['unit'] = 'radian'

            # determin if it is the same pointing
            if self.main_data_dist_axis == 0:
                az_alt = az_alt.local_array
                ra_dec = ra_dec.local_array
            # gather local az_alt
            az_alt = mpiutil.gather_array(az_alt, root=None, comm=self.comm)
            if np.allclose(az_alt[:, 0], az_alt[0, 0]) and np.allclose(az_alt[:, 1], az_alt[0, 1]):
                self['az_alt'].attrs['same_pointing'] = True
            else:
                self['az_alt'].attrs['same_pointing'] = False
            # determin if it is the same dec
            # gather local ra_dec
            ra_dec = mpiutil.gather_array(ra_dec, root=None, comm=self.comm)
            if np.allclose(ra_dec[:, 1], ra_dec[0, 1]):
                self['ra_dec'].attrs['same_dec'] = True
            else:
                self['ra_dec'].attrs['same_dec'] = False


    @property
    def vis(self):
        """Return the main data for convenient use."""
        return self.main_data

    @property
    def local_vis(self):
        """A convenience for vis.local_data."""
        return self.main_data.local_data

    @property
    def vis_mask(self):
        """A convenience for self['vis_mask']."""
        try:
            return self['vis_mask']
        except KeyError:
            raise KeyError('vis_mask does not exist, try to load it first')

    @property
    def local_vis_mask(self):
        """A convenience for vis_mask.local_data."""
        return self.vis_mask.local_data

    def apply_mask(self, fill_val=complex(np.nan, np.nan)):
        """Applying `vis_mask` to `vis` with the `fill_val`.

        NOTE: This is a non-invertible operation to the dataset `vis`, as its
        values corresponding to True of `vis_mask` will be replaced by `fill_val`
        and lost. Usually you may prefer to use :meth:`masked_vis` instead.
        """
        self.vis[:] = np.where(self.vis_mask[:], fill_val, self.vis[:])

    @property
    def masked_vis(self):
        """Return a copy of the masked `vis`.

        NOTE: This can only be used like self.masked_vis[slice], not
        self.masked_vis.data[slice] and self.masked_vis.local_data[slice].

        This will keep the underling `vis` dataset unchanged.
        """

        outer_self = self

        class _masked_vis(outer_self.vis.__class__):
            """This will be a subclass of ether :class:`caput.memh5.MemDatasetCommon`
            or :class:`caput.memh5.MemDatasetDistributed` depending on the type
            of `outer_self.vis.__class__`.
            """

            def __getitem__(self, obj):
                vis = super(_masked_vis, self).__getitem__(obj)
                vis_mask = outer_self.vis_mask.__getitem__(obj)
                if vis_mask is None:
                    return None
                else:
                    return np.where(vis_mask, complex(np.nan, np.nan), vis)

            def __setitem__(self, obj, val):
                raise RuntimeError("Can not set item as 'mask_vis' is not a real dataset, set 'vis' and 'vis_mask' instead")

            @property
            def data(self):
                raise ValueError('Can not get masked_vis.data')

            @property
            def local_data(self):
                raise ValueError('Can not get masked_vis.local_data')

        if outer_self.distributed:
            mv = _masked_vis.from_mpi_array(outer_self.vis.data)
        else:
            mv = _masked_vis.from_numpy_array(outer_self.vis.data)

        mv._name = 'masked_vis'

        return mv


    @property
    def time(self):
        """Return the jul_date dataset for convenient use."""
        try:
            return self['jul_date']
        except KeyError:
            raise KeyError('jul_date does not exist, try to load it first')

    @property
    def local_time(self):
        """A convenience for time.local_data."""
        return self.time.local_data

    @property
    def freq(self):
        """Return the freq dataset for convenient use."""
        try:
            return self['freq']
        except KeyError:
            raise KeyError('freq does not exist, try to load it first')

    @property
    def local_freq(self):
        """A convenience for freq.local_data."""
        return self.freq.local_data

    @property
    def bl(self):
        """Return the blorder dataset for convenient use."""
        try:
            return self['blorder']
        except KeyError:
            raise KeyError('blorder does not exist, try to load it first')

    @property
    def local_bl(self):
        """A convenience for bl.local_data."""
        return self.bl.local_data

    @property
    def is_dish(self):
        """True if data is get from dish array."""
        try:
            return 'Dish' in self.attrs['telescope']
        except KeyError:
            raise KeyError('Attribute telescope does not exist, try to load it first')

    @property
    def is_cylinder(self):
        """True if data is get from cylinder array."""
        try:
            return 'Cylinder' in self.attrs['telescope']
        except KeyError:
            raise KeyError('Attribute telescope does not exist, try to load it first')

    @property
    def array(self):
        """Return either a dish array or a cylinder array instance."""
        try:
            lon = self.attrs['sitelon'] # degree
            lat = self.attrs['sitelat'] # degree
            elev = self.attrs['siteelev'] # m
        except KeyError:
            raise KeyError('Attribute sitelon, sitelat or siteelev does not exist, try to load it first')

        freq = self.freq.local_data[:]
        pos = self['feedpos'].local_data[:] # in topocentric coordinate
        nfeed = pos.shape[0]
        pos -= pos[-1]
        # convert to equatorial (ns) coordinates
        m2ns = 1.0 / const.c * 1.0e9
        pos_ns = np.dot(tl_array.xyz2XYZ_m(np.radians(lat)), m2ns * pos.T).T
        if self.is_dish:
            try:
                diameter = self.attrs['dishdiam']
            except KeyError:
                raise KeyError('Attribute dishdiam does not exist, try to load it first')
            # beam = tl_array.DishBeam(freq, diameter)
            ants = [ tl_array.DishAntenna(pi, freq, diameter) for pi in pos_ns ]
        elif self.is_cylinder:
            try:
                factor = 1.2 # suppose an illumination efficiency
                width = factor * self.attrs['cywid']
                length = self.attrs['cylen']
            except KeyError:
                raise KeyError('Attribute dishdiam does not exist, try to load it first')
            # beam = tl_array.CylinderBeam(freq, width, length)
            ants = [ tl_array.CylinderFeed(pi, freq, width, length) for pi in pos_ns ]

        aa = tl_array.AntennaArray((str(lat), str(lon), elev), ants)

        return aa

    @property
    def is_continuous(self):
        """Data is observed continuous in time?"""
        return self['sec1970'].attrs["continuous"]

    @property
    def is_same_pointing(self):
        """The antenna array is in the same pointing in az, alt?"""
        return self['az_alt'].attrs['same_pointing']

    @property
    def is_same_dec(self):
        """Data is observed for the same declination in the sky?"""
        return self['ra_dec'].attrs['same_dec']


    def create_freq_ordered_dataset(self, name, data, axis_order=None, recreate=False, copy_attrs=False, check_align=True):
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
        check_align : bool, optional
            If True, check frequency axis of data align with that of the main data
            before dataset creating, otherwise create dataset without axis align
            checking, this may cause the created dataset does not align with the
            main data. Default True.

        """

        axis_order = axis_order or (self.main_data_axes.index('frequency'),)

        self.create_main_axis_ordered_dataset('frequency', name, data, axis_order, recreate, copy_attrs, check_align)

    def create_bl_ordered_dataset(self, name, data, axis_order=None, recreate=False, copy_attrs=False, check_align=True):
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
        check_align : bool, optional
            If True, check baseline axis of data align with that of the main data
            before dataset creating, otherwise create dataset without axis align
            checking, this may cause the created dataset does not align with the
            main data. Default True.

        """

        axis_order = axis_order or (self.main_data_axes.index('baseline'),)

        self.create_main_axis_ordered_dataset('baseline', name, data, axis_order, recreate, copy_attrs, check_align)

    def create_time_and_freq_ordered_dataset(self, name, data, axis_order=None, recreate=False, copy_attrs=False, check_align=True):
        """Create a time and frequency ordered dataset.

        Parameters
        ----------
        name : string
            Name of the dataset.
        data : np.ndarray or MPIArray
            The data to create a dataset.
        axis_order : tuple
            A tuple denotes time and frequency axis of the created dataset.
        recreate : bool, optional
            If True will recreate a dataset with this name if it already exists,
            else a RuntimeError will be rasised. Default False.
        copy_attrs : bool, optional
            If True, when recreate the dataset, its original attributes will be
            copyed to the new dataset, else no copy is done. Default Fasle.
        check_align : bool, optional
            If True, check time and frequency axis of data align with that of the
            main data before dataset creating, otherwise create dataset without
            axis align checking, this may cause the created dataset does not align
            with the main data. Default True.

        """

        axis_order = axis_order or (self.main_data_axes.index('time'), self.main_data_axes.index('frequency'))

        self.create_main_axis_ordered_dataset(('time', 'frequency'), name, data, axis_order, recreate, copy_attrs, check_align)

    def create_time_and_bl_ordered_dataset(self, name, data, axis_order=None, recreate=False, copy_attrs=False, check_align=True):
        """Create a time and baseline ordered dataset.

        Parameters
        ----------
        name : string
            Name of the dataset.
        data : np.ndarray or MPIArray
            The data to create a dataset.
        axis_order : tuple
            A tuple denotes time and baseline axis of the created dataset.
        recreate : bool, optional
            If True will recreate a dataset with this name if it already exists,
            else a RuntimeError will be rasised. Default False.
        copy_attrs : bool, optional
            If True, when recreate the dataset, its original attributes will be
            copyed to the new dataset, else no copy is done. Default Fasle.
        check_align : bool, optional
            If True, check time and baseline axis of data align with that of the
            main data before dataset creating, otherwise create dataset without
            axis align checking, this may cause the created dataset does not align
            with the main data. Default True.

        """

        axis_order = axis_order or (self.main_data_axes.index('time'), self.main_data_axes.index('baseline'))

        self.create_main_axis_ordered_dataset(('time', 'baseline'), name, data, axis_order, recreate, copy_attrs, check_align)

    def create_freq_and_bl_ordered_dataset(self, name, data, axis_order=None, recreate=False, copy_attrs=False, check_align=True):
        """Create a frequency and baseline ordered dataset.

        Parameters
        ----------
        name : string
            Name of the dataset.
        data : np.ndarray or MPIArray
            The data to create a dataset.
        axis_order : tuple
            A tuple denotes frequency and baseline axis of the created dataset.
        recreate : bool, optional
            If True will recreate a dataset with this name if it already exists,
            else a RuntimeError will be rasised. Default False.
        copy_attrs : bool, optional
            If True, when recreate the dataset, its original attributes will be
            copyed to the new dataset, else no copy is done. Default Fasle.
        check_align : bool, optional
            If True, check frequency and baseline axis of data align with that of the
            main data before dataset creating, otherwise create dataset without
            axis align checking, this may cause the created dataset does not align
            with the main data. Default True.

        """

        axis_order = axis_order or (self.main_data_axes.index('frequency'), self.main_data_axes.index('baseline'))

        self.create_main_axis_ordered_dataset(('frequency', 'baseline'), name, data, axis_order, recreate, copy_attrs, check_align)

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


    def delete_a_dataset(self, name):
        """Delete a dataset and also remove it from the hint if it is in it."""
        super(TimestreamCommon, self).delete_a_dataset(name)

        if name in self._feed_ordered_datasets_.iterkeys():
            del self._feed_ordered_datasets_[name]


    def check_status(self):
        """Check that data hold in this container is consistent. """

        # basic checks
        super(TimestreamCommon, self).check_status()

        # additional checks for feed_ordered_datasets
        lens = []
        for name, val in self.feed_ordered_datasets.items():
            if name in self.items():
                lens.append(self[name].shape[val.index(0)])
        num = len(set(lens))
        if num != 0 and num != 1:
            raise RuntimeError('Not all feed_ordered_datasets have an aligned feed axis')


    def data_operate(self, func, op_axis=None, axis_vals=0, full_data=False, copy_data=False, show_progress=False, progress_step=None, keep_dist_axis=False, **kwargs):
        """An overload data operation interface.

        This overloads the method in its super class :class:`container.BasicTod`
        to provide basic operation interface to both `vis` (the main data) and
        `vis_mask`.

        You can use this method to do some constrained operations to `vis` and
        `vis_mask` hold in this container, i.e., the operations will not change
        the shape and dtype of `vis` and `vis_mask` before and after the operation.

        Parameters
        ----------
        func : function object
            The opertation function object. It is of type func(vis, vis_mask,
            self, \*\*kwargs) if `op_axis=None`, func(vis, vis_mask, local_index,
            global_index, axis_val, self, \*\*kwargs) else.
        op_axis : None, string or integer, tuple of string or interger, optional
            Axis along which `func` will opterate. If None, `func` will operate on
            the whole main dataset (but note: since the main data is distributed
            on different processes, `func` should not have operations that depend
            on elements not held in the local array of each process), if string or
            interger `func` will opterate along the specified axis, that is, it
            will loop the specified axis and call `func` on data section
            corresponding to the axis index, if tuple of string or interger,
            `func` will operate along all these axes.
        axis_vals : scalar or array, tuple of scalar or array, optional
            Axis value (or tuple of axis values) corresponding to the local
            section along the `op_axis` that will be passed to `func` if
            `op_axis` is not None. Default 0.
        full_data : bool, optional
            Whether the operations of `func` will need the full data section
            corresponding to the axis index, if True, the main data will first
            redistributed along the `op_axis`. Default False.
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
            Whether to redistribute main data to the original dist axis if the
            dist axis has changed during the operation. Default False.
        \*\*kwargs : any other arguments
            Any other arguments that will passed to `func`.

        """

        if op_axis is None:
            if copy_data:
                func(self.local_vis.copy(), self.local_vis_mask.copy(), self, **kwargs)
            else:
                func(self.local_vis, self.local_vis_mask, self, **kwargs)
        elif isinstance(op_axis, int) or isinstance(op_axis, basestring):
            axis = container.check_axis(op_axis, self.main_data_axes)
            data_sel = [ slice(0, None) ] * len(self.main_data_axes)
            if full_data:
                original_dist_axis = self.main_data_dist_axis
                self.redistribute(axis)
            if self.main_data.distributed:
                lgind = list(self.main_data.data.enumerate(axis))
            else:
                lgind = list(enumerate(range(self.main_data.data.shape[axis])))
            if show_progress:
                pg = progress.Progress(len(lgind), step=progress_step)
            cnt = 0
            for lind, gind in lgind:
                if show_progress and mpiutil.rank0:
                    pg.show(cnt)
                cnt += 1
                data_sel[axis] = lind
                if isinstance(axis_vals, memh5.MemDataset):
                    # use the new dataset which may be different from axis_vals if it is redistributed
                    axis_val = self[axis_vals.name].local_data[lind]
                elif hasattr(axis_vals, '__iter__'):
                    axis_val = axis_vals[lind]
                else:
                    axis_val = axis_vals
                if copy_data:
                    func(self.local_vis[data_sel].copy(), self.local_vis_mask[data_sel].copy(), lind, gind, axis_val, self, **kwargs)
                else:
                    func(self.local_vis[data_sel], self.local_vis_mask[data_sel], lind, gind, axis_val, self, **kwargs)
            if full_data and keep_dist_axis:
                self.redistribute(original_dist_axis)
        elif isinstance(op_axis, tuple):
            axes = [ container.check_axis(axis, self.main_data_axes) for axis in op_axis ]
            data_sel = [ slice(0, None) ] * len(self.main_data_axes)
            if full_data:
                original_dist_axis = self.main_data_dist_axis
                if not original_dist_axis in axes:
                    shape = self.main_data.shape
                    axes_len = [ shape[axis] for axis in axes ]
                    # choose the longest axis in axes as the new dist axis
                    new_dist_axis = axes[np.argmax(axes_len)]
                    self.redistribute(new_dist_axis)
            if self.main_data.distributed:
                lgind = [ list(self.main_data.data.enumerate(axis)) for axis in axes ]
            else:
                lgind = [ list(enumerate(range(self.main_data.data.shape[axis]))) for axis in axes ]
            linds = [ [ li for (li, gi) in lg ] for lg in lgind ]
            ginds = [ [ gi for (li, gi) in lg ] for lg in lgind ]
            lgind = zip(itertools.product(*linds), itertools.product(*ginds))
            if show_progress:
                pg = progress.Progress(len(lgind), step=progress_step)
            cnt = 0
            for lind, gind in lgind:
                if show_progress and mpiutil.rank0:
                    pg.show(cnt)
                cnt += 1
                axis_val = ()
                for ai, axis in enumerate(axes):
                    data_sel[axis] = lind[ai]
                    if isinstance(axis_vals[ai], memh5.MemDataset):
                        # use the new dataset which may be different from axis_vals if it is redistributed
                        axis_val += (self[axis_vals[ai].name].local_data[lind[ai]],)
                    elif hasattr(axis_vals[ai], '__iter__'):
                        axis_val += (axis_vals[ai][lind[ai]],)
                    else:
                        axis_val += (axis_vals[ai],)
                if copy_data:
                    func(self.local_vis[data_sel].copy(), self.local_vis_mask[data_sel].copy(), lind, gind, axis_val, self, **kwargs)
                else:
                    func(self.local_vis[data_sel], self.local_vis_mask[data_sel], lind, gind, axis_val, self, **kwargs)
            if full_data and keep_dist_axis:
                self.redistribute(original_dist_axis)
        else:
            raise ValueError('Invalid op_axis: %s', op_axis)

    def all_data_operate(self, func, copy_data=False, **kwargs):
        """Operation to the whole `vis` and `vis_mask`.

        Note since `vis` and `vis_mask` is usually distributed on different
        processes, `func` should not have operations that depend on their
        elements not held in their local array of each process.

        Parameters
        ----------
        func : function object
            The opertation function object. It is of type func(vis, vis_mask, self,
            \*\*kwargs), which will operate on vis, vis_mask and return two new
            arrays with their original shape and dtype.
        copy_data : bool, optional
            If True, `func` will operate on a copy of the data, so changes will
            have no impact on the data the container holds. Default False.
        \*\*kwargs : any other arguments
            Any other arguments that will passed to `func`.

        """
        self.data_operate(func, op_axis=None, axis_vals=0, full_data=False, copy_data=copy_data, keep_dist_axis=False, **kwargs)

    def time_data_operate(self, func, full_data=False, copy_data=False, show_progress=False, progress_step=None, keep_dist_axis=False, **kwargs):
        """Data operation along the time axis.

        Parameters
        ----------
        func : function object
            The opertation function object. It is of type func(vis, vis_mask,
            local_index, global_index, jul_date, self, \*\*kwargs), which
            will be called in a loop along the time axis.
        full_data : bool, optional
            Whether the operations of `func` will need the full data section
            corresponding to the axis index, if True, data will first
            redistributed along the time axis. Default False.
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
        self.data_operate(func, op_axis='time', axis_vals=self.time, full_data=full_data, copy_data=copy_data, show_progress=show_progress, progress_step=progress_step, keep_dist_axis=keep_dist_axis, **kwargs)

    def freq_data_operate(self, func, full_data=False, copy_data=False, show_progress=False, progress_step=None, keep_dist_axis=False, **kwargs):
        """Data operation along the frequency axis.

        Parameters
        ----------
        func : function object
            The opertation function object. It is of type func(vis, vis_mask,
            local_index, global_index, freq, self, \*\*kwargs), which
            will be called in a loop along the frequency axis.
        full_data : bool, optional
            Whether the operations of `func` will need the full data section
            corresponding to the axis index, if True, data will first
            redistributed along frequency axis. Default False.
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
        self.data_operate(func, op_axis='frequency', axis_vals=self.freq, full_data=full_data, copy_data=copy_data, show_progress=show_progress, progress_step=progress_step, keep_dist_axis=keep_dist_axis, **kwargs)

    def bl_data_operate(self, func, full_data=False, copy_data=False, show_progress=False, progress_step=None, keep_dist_axis=False, **kwargs):
        """Data operation along the baseline axis.

        Parameters
        ----------
        func : function object
            The opertation function object. It is of type func(vis, vis_mask,
            local_index, global_index, chanpair, self, \*\*kwargs), which
            will be called in a loop along the baseline axis.
        full_data : bool, optional
            Whether the operations of `func` will need the full data section
            corresponding to the axis index, if True, data will first
            redistributed along baseline axis. Default False.
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
        self.data_operate(func, op_axis='baseline', axis_vals=self.bl, full_data=full_data, copy_data=copy_data, show_progress=show_progress, progress_step=progress_step, keep_dist_axis=keep_dist_axis, **kwargs)

    def time_and_freq_data_operate(self, func, full_data=False, copy_data=False, show_progress=False, progress_step=None, keep_dist_axis=False, **kwargs):
        """Data operation along the time and frequency axis.

        Parameters
        ----------
        func : function object
            The opertation function object. It is of type func(vis, vis_mask,
            local_index, global_index, tf, self, \*\*kwargs), which
            will be called in a loop along the time and frequency axis.
        full_data : bool, optional
            Whether the operations of `func` will need the full data section
            corresponding to the axis index, if True, data will first
            redistributed along time or frequency axis which is longer.
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
        self.data_operate(func, op_axis=('time', 'frequency'), axis_vals=(self.time, self.freq), full_data=full_data, copy_data=copy_data, show_progress=show_progress, progress_step=progress_step, keep_dist_axis=keep_dist_axis, **kwargs)

    def time_and_bl_data_operate(self, func, full_data=False, copy_data=False, show_progress=False, progress_step=None, keep_dist_axis=False, **kwargs):
        """Data operation along the time and baseline axis.

        Parameters
        ----------
        func : function object
            The opertation function object. It is of type func(vis, vis_mask,
            local_index, global_index, tbl, self, \*\*kwargs), which
            will be called in a loop along the time and baseline axis.
        full_data : bool, optional
            Whether the operations of `func` will need the full data section
            corresponding to the axis index, if True, data will first
            redistributed along time or baseline axis which is longer.
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
        self.data_operate(func, op_axis=('time', 'baseline'), axis_vals=(self.time, self.bl), full_data=full_data, copy_data=copy_data, show_progress=show_progress, progress_step=progress_step, keep_dist_axis=keep_dist_axis, **kwargs)

    def freq_and_bl_data_operate(self, func, full_data=False, copy_data=False, show_progress=False, progress_step=None, keep_dist_axis=False, **kwargs):
        """Data operation along the frequency and baseline axis.

        Parameters
        ----------
        func : function object
            The opertation function object. It is of type func(vis, vis_mask,
            local_index, global_index, fbl, self, \*\*kwargs), which
            will be called in a loop along the frequency and baseline axis.
        full_data : bool, optional
            Whether the operations of `func` will need the full data section
            corresponding to the axis index, if True, data will first
            redistributed along frequency or baseline axis which is longer.
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
        self.data_operate(func, op_axis=('frequency', 'baseline'), axis_vals=(self.freq, self.bl), full_data=full_data, copy_data=copy_data, show_progress=show_progress, progress_step=progress_step, keep_dist_axis=keep_dist_axis, **kwargs)


    def _copy_a_common_dataset(self, name, other):
        ### copy a common dataset from `other` to self
        if name == 'feedno' and not other._subset_feed_select is None:
            self.create_dataset(name, data=other._subset_feed_select)
            memh5.copyattrs(other[name].attrs, self[name].attrs)
        elif name in self.feed_ordered_datasets.keys() and not other._subset_feed_select is None:
            feedno = other['feedno'][:].tolist()
            feed_inds = [ feedno.index(fd) for fd in other._subset_feed_select ]
            feed_axis = self.feed_ordered_datasets[name].index(0)
            slc = [slice(0, None)] * len(other[name].shape)
            slc[feed_axis] = feed_inds
            self.create_dataset(name, data=other[name][tuple(slc)])
            memh5.copyattrs(other[name].attrs, self[name].attrs)
        else:
            super(TimestreamCommon, self)._copy_a_common_dataset(name, other)
