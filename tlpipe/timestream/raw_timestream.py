import itertools
import numpy as np
import timestream_common
import timestream
from caput import mpiarray
from caput import memh5


class RawTimestream(timestream_common.TimestreamCommon):
    """Container class for the raw timestream data.

    The raw timestream data are raw visibilities (the main data) and other data
    and meta data saved in HDF5 files which are recorded from the correlator.

    Parameters
    ----------
    Same as :class:`container.BasicTod`.

    Attributes
    ----------

    Methods
    -------
    feed_select
    separate_pol_and_bl

    """

    _main_data_name_ = 'vis'
    _main_data_axes_ = ('time', 'frequency', 'baseline')
    _main_time_ordered_datasets_ = {'vis', 'sec1970', 'jul_date'}
    _time_ordered_datasets_ = _main_time_ordered_datasets_ | {'weather'}
    _time_ordered_attrs_ = {'obstime', 'sec1970'}
    _freq_ordered_datasets_ = {'freq'}
    _bl_ordered_datasets_ = {'blorder'}
    _feed_ordered_datasets_ = {'antpointing', 'channo', 'feedno', 'feedpos', 'polerr'}


    _channel_select = None

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

    #     self.data_select('baseline', indices)

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

        self.data_select('baseline', indices)

        self._feed_select = feeds
        self._channel_select = np.array([ channo[feedno.index(fd)] for fd in feeds ])


    def _load_a_common_dataset(self, name):
        ### load a common dataset from the first file
        if name == 'channo' and not self._channel_select is None:
            self.create_dataset(name, data=self._channel_select)
            memh5.copyattrs(self.infiles[0][name].attrs, self[name].attrs)
        else:
            super(RawTimestream, self)._load_a_common_dataset(name)


    def separate_pol_and_bl(self, keep_dist_axis=False):
        """Separate baseline axis to polarization and baseline.

        This will create and return a Timestream container holding the polarization
        and baseline separated data.

        Parameters
        ----------
        keep_dist_axis : bool, optional
            Whether to redistribute main data to the original dist axis if the
            dist axis has changed during the operation. Default False.

        """

        # if dist axis is baseline, redistribute it along time
        original_dist_axis = self.main_data_dist_axis
        if 'baseline' == self.main_data_axes[original_dist_axis]:
            keep_dist_axis = False # can not keep dist axis in this case
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
        # pol ordered dataset
        ts.create_pol_ordered_dataset('pol', data=np.array(['xx', 'yy', 'xy', 'yx']))
        ts['pol'].attrs['pol_type'] = 'linear'

        # bl ordered dataset
        blorder = np.array([ [feedno[i], feedno[j]] for i in range(nfeed) for j in range(i, nfeed) ])
        ts.create_bl_ordered_dataset('blorder', data=blorder)
        # copy attrs of this dset
        memh5.copyattrs(self['blorder'].attrs, ts['blorder'].attrs)
        # other bl ordered dataset
        if len(self.bl_ordered_datasets - {'blorder'}) > 0:
            old_blorder = [ set(bl) for bl in self['blorder'][:] ]
            inds = [ old_blorder.index(set(bl)) for bl in ts['bl_order'][:] ]
            for name in (self.bl_ordered_datasets - {'blorder'}):
                if name in self.iterkeys():
                    ts.create_bl_ordered_dataset(name, data=self[name][inds])
                    # copy attrs of this dset
                    memh5.copyattrs(self[name].attrs, ts[name].attrs)

        # copy other attrs
        for attrs_name, attrs_value in self.attrs.iteritems():
            if attrs_name not in self.time_ordered_attrs:
                ts.attrs[attrs_name] = attrs_value

        # copy other datasets
        for dset_name, dset in self.iteritems():
            if dset_name == self.main_data_name:
                # already create above
                continue
            elif dset_name in self.main_time_ordered_datasets:
                ts.create_main_time_ordered_dataset(dset_name, data=dset.data)
            elif dset_name in self.time_ordered_datasets:
                ts.create_time_ordered_dataset(dset_name, data=dset.data)
            elif dset_name in self.freq_ordered_datasets:
                ts.create_freq_ordered_dataset(dset_name, data=dset.data)
            elif dset_name in self.bl_ordered_datasets:
                # already create above
                continue
            elif dset_name in self.feed_ordered_datasets:
                if dset_name == 'channo': # channo no useful for Timestream
                    continue
                else:
                    ts.create_feed_ordered_dataset(dset_name, data=dset.data)
            else:
                if dset.common:
                    ts.create_dataset(dset_name, data=dset)
                elif dset.distributed:
                    ts.create_dataset(dset_name, data=dset.data, shape=dset.shape, dtype=dset.dtype, distributed=True, distributed_axis=dset.distributed_axis)

            # copy attrs of this dset
            memh5.copyattrs(dset.attrs, ts[dset_name].attrs)

        # redistribute self to original axis
        if keep_dist_axis:
            self.redistribute(original_dist_axis)

        return ts
