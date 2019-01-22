"""Container class for the raw timestream data.


Inheritance diagram
-------------------

.. inheritance-diagram:: tlpipe.container.container.BasicTod tlpipe.container.timestream_common.TimestreamCommon RawTimestream tlpipe.container.timestream.Timestream
   :parts: 2

"""

import re
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

    """

    _main_data_name_ = 'vis'
    _main_data_axes_ = ('time', 'frequency', 'baseline')
    _main_axes_ordered_datasets_ = { 'vis': (0, 1, 2),
                                     'vis_mask': (0, 1, 2),
                                     'sec1970': (0,),
                                     'jul_date': (0,),
                                     'freq': (1,),
                                     'blorder': (2,),
                                   }
    _time_ordered_datasets_ = {'weather': (0,)}
    _time_ordered_attrs_ = {'obstime', 'sec1970'}
    _feed_ordered_datasets_ = { 'antpointing': (None, 0),
                                'feedno': (0,),
                                'channo': (0,),
                                'feedpos': (0,),
                                'polerr': (0,),
                              }


    _channel_select = None
    _subset_channel_select = None

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
    #         channel_pairs = [ {channels[i]} for i in xrange(nchan) ]
    #     elif corr == 'cross':
    #         channel_pairs = [ {channels[i], channels[j]} for i in xrange(nchan) for j in xrange(i+1, nchan) ]
    #     elif corr == 'all':
    #         channel_pairs = [ {channels[i], channels[j]} for i in xrange(nchan) for j in xrange(i, nchan) ]
    #     else:
    #         raise ValueError('Unknown correlation type %s' % corr)

    #     # get blorder info from the first input file
    #     blorder = self.infiles[0]['blorder']
    #     blorder = [ set(bl) for bl in blorder ]

    #     # channel pair indices
    #     indices = { blorder.index(chp) for chp in channel_pairs }
    #     indices = sorted(list(indices))

    #     self.data_select('baseline', indices)

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

        # get channo info from data source
        channo = data_source['channo'][:]
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

        # get blorder info from data_source
        if isinstance(data_source['blorder'], memh5.MemDatasetDistributed):
            blorder = data_source['blorder'].data.to_numpy_array(root=None)
        else:
            blorder = data_source['blorder']
        blorder = [ set(bl) for bl in blorder ]

        # channel pair indices
        indices = { blorder.index(chp) for chp in channel_pairs }
        indices = sorted(list(indices))

        # selected channels corresponding to feeds
        channels = np.array([ channo[feedno.index(fd)] for fd in feeds ])

        return indices, feeds, channels

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
            indices, feeds, channels = results
            self.data_select('baseline', indices)
            self._feed_select = feeds
            self._channel_select = channels

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
            indices, feeds, channels = results
            self.subset_select('baseline', indices)
            self._subset_feed_select = feeds
            self._subset_channel_select = channels


    def _load_a_common_dataset(self, name):
        ### load a common dataset from the first file
        if name == 'channo' and not self._channel_select is None:
            self.create_dataset(name, data=self._channel_select)
            memh5.copyattrs(self.infiles[0][name].attrs, self[name].attrs)
        else:
            super(RawTimestream, self)._load_a_common_dataset(name)


    def load_all(self):
        """Load all attributes and datasets from files."""

        super(RawTimestream, self).load_all()

        # create some new necessary datasets is they do not already exist in file
        if 'true_blorder' not in self.iterkeys():
            feed_no = self['feedno'][:].tolist()
            xch_no = self['channo'][:, 0].tolist()
            ych_no = self['channo'][:, 1].tolist()
            bl_order = self['blorder'].local_data
            # create a bl_ordered dataset that is the feed numbered blorder
            true_bl = np.zeros_like(bl_order)
            # create a bl_ordered dataset that is the pol status of blorder
            bl_pol = np.empty(bl_order.shape[0], dtype='i4')
            # fill the dataset
            for bi, (ch1, ch2) in enumerate(bl_order):
                if ch1 in xch_no:
                    fd1 = feed_no[xch_no.index(ch1)]
                    p1 = 'x'
                elif ch1 in ych_no:
                    fd1 = feed_no[ych_no.index(ch1)]
                    p1 = 'y'
                else:
                    raise RuntimeError('%d is not in dataset channo' % ch1)

                if ch2 in xch_no:
                    fd2 = feed_no[xch_no.index(ch2)]
                    p2 = 'x'
                elif ch2 in ych_no:
                    fd2 = feed_no[ych_no.index(ch2)]
                    p2 = 'y'
                else:
                    raise RuntimeError('%d is not in dataset channo' % ch2)

                true_bl[bi, 0] = fd1
                true_bl[bi, 1] = fd2
                bl_pol[bi] = self.pol_dict[p1+p2]

            # if baseline is just the distributed axis, load the datasets distributed
            if 'baseline' == self.main_data_axes[self.main_data_dist_axis]:
                true_bl = mpiarray.MPIArray.wrap(true_bl, axis=0, comm=self.comm)
                bl_pol = mpiarray.MPIArray.wrap(bl_pol, axis=0, comm=self.comm)
            self.create_bl_ordered_dataset('true_blorder', data=true_bl)
            self.create_bl_ordered_dataset('bl_pol', data=bl_pol)


    def separate_pol_and_bl(self, keep_dist_axis=False, destroy_self=False):
        """Separate baseline axis to polarization and baseline.

        This will create and return a Timestream container holding the polarization
        and baseline separated data.

        Parameters
        ----------
        keep_dist_axis : bool, optional
            Whether to redistribute main data to the original dist axis if the
            dist axis has changed during the operation. Default False.
        destroy_self : bool, optional
            If True, will gradually delete datasets and attributes of self to
            release memory. Default False.

        """

        # if dist axis is baseline, redistribute it along time
        original_dist_axis = self.main_data_dist_axis
        if 'baseline' == self.main_data_axes[original_dist_axis]:
            keep_dist_axis = False # can not keep dist axis in this case
            self.redistribute(0)

        # could not keep dist axis if destroy_self == True
        if destroy_self:
            keep_dist_axis = False

        # create a Timestream container to hold the pol and bl separated data
        ts = timestream.Timestream(dist_axis=self.main_data_dist_axis, comm=self.comm)

        feedno = sorted(self['feedno'][:].tolist())
        xchans = [ self['channo'][feedno.index(fd)][0] for fd in feedno ]
        ychans = [ self['channo'][feedno.index(fd)][1] for fd in feedno ]

        nfeed = len(feedno)
        xx_pairs = [ (xchans[i], xchans[j]) for i in xrange(nfeed) for j in xrange(i, nfeed) ]
        yy_pairs = [ (ychans[i], ychans[j]) for i in xrange(nfeed) for j in xrange(i, nfeed) ]
        xy_pairs = [ (xchans[i], ychans[j]) for i in xrange(nfeed) for j in xrange(i, nfeed) ]
        yx_pairs = [ (ychans[i], xchans[j]) for i in xrange(nfeed) for j in xrange(i, nfeed) ]

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
        rvis = self.main_data.local_data
        shp = rvis.shape[:2] + (4, len(xx_inds))
        vis = np.empty(shp, dtype=rvis.dtype)
        vis[:, :, 0] = np.where(xx_conj, rvis[:, :, xx_inds].conj(), rvis[:, :, xx_inds]) # xx
        vis[:, :, 1] = np.where(yy_conj, rvis[:, :, yy_inds].conj(), rvis[:, :, yy_inds]) # yy
        vis[:, :, 2] = np.where(xy_conj, rvis[:, :, xy_inds].conj(), rvis[:, :, xy_inds]) # xy
        vis[:, :, 3] = np.where(yx_conj, rvis[:, :, yx_inds].conj(), rvis[:, :, yx_inds]) # yx

        vis = mpiarray.MPIArray.wrap(vis, axis=self.main_data_dist_axis, comm=self.comm)

        # create main data
        ts.create_main_data(vis)
        # copy attrs from rt
        memh5.copyattrs(self.main_data.attrs, ts.main_data.attrs)
        # create attrs of this dataset
        ts.main_data.attrs['dimname'] = 'Time, Frequency, Polarization, Baseline'

        if destroy_self:
            self.delete_a_dataset('vis')

        # create a MPIArray to hold the pol and bl separated vis_mask
        rvis_mask = self['vis_mask'].local_data
        shp = rvis_mask.shape[:2] + (4, len(xx_inds))
        vis_mask = np.empty(shp, dtype=rvis_mask.dtype)
        vis_mask[:, :, 0] = rvis_mask[:, :, xx_inds] # xx
        vis_mask[:, :, 1] = rvis_mask[:, :, yy_inds] # yy
        vis_mask[:, :, 2] = rvis_mask[:, :, xy_inds] # xy
        vis_mask[:, :, 3] = rvis_mask[:, :, yx_inds] # yx

        vis_mask = mpiarray.MPIArray.wrap(vis_mask, axis=self.main_data_dist_axis, comm=self.comm)

        # create vis_mask
        axis_order = ts.main_axes_ordered_datasets[ts.main_data_name]
        ts.create_main_axis_ordered_dataset(axis_order, 'vis_mask', vis_mask, axis_order)

        if destroy_self:
            self.delete_a_dataset('vis_mask')

        # create other datasets needed
        # pol ordered dataset
        p = self.pol_dict
        ts.create_pol_ordered_dataset('pol', data=np.array([p['xx'], p['yy'], p['xy'], p['yx']], dtype='i4'))
        ts['pol'].attrs['pol_type'] = 'linear'

        # bl ordered dataset
        blorder = np.array([ [feedno[i], feedno[j]] for i in xrange(nfeed) for j in xrange(i, nfeed) ])
        ts.create_bl_ordered_dataset('blorder', data=blorder)
        # copy attrs of this dset
        memh5.copyattrs(self['blorder'].attrs, ts['blorder'].attrs)
        # other bl ordered dataset
        other_bl_dset = set(self.bl_ordered_datasets.keys()) - {'vis', 'vis_mask', 'blorder', 'true_blorder', 'bl_pol'}
        if len(other_bl_dset) > 0:
            raise RuntimeError('Should not have other bl_ordered_datasets %s' % other_bl_dset)

        # copy other attrs
        attrs_items = list(self.attrs.iteritems())
        for attrs_name, attrs_value in attrs_items:
            if attrs_name not in self.time_ordered_attrs:
                ts.attrs[attrs_name] = attrs_value
            if destroy_self:
                self.delete_an_attribute(attrs_name)

        # copy other datasets
        for dset_name, dset in self.iteritems():
            if dset_name == self.main_data_name or dset_name == 'vis_mask':
                if destroy_self:
                    self.delete_a_dataset(dset_name)
                # already created above
                continue
            elif dset_name in self.main_axes_ordered_datasets.keys():
                if dset_name in self.bl_ordered_datasets.keys():
                    if destroy_self:
                        self.delete_a_dataset(dset_name)
                    # already created above
                    continue
                else:
                    axis_order = self.main_axes_ordered_datasets[dset_name]
                    axis = None
                    for order in axis_order:
                        if isinstance(order, int):
                            axis = order
                    if axis is None:
                        raise RuntimeError('Invalid axis order %s for dataset %s' % (axis_order, dset_name))
                    ts.create_main_axis_ordered_dataset(axis, dset_name, dset.data, axis_order)
            elif dset_name in self.time_ordered_datasets.keys():
                axis_order = self.time_ordered_datasets[dset_name]
                ts.create_time_ordered_dataset(dset_name, dset.data, axis_order)
            elif dset_name in self.feed_ordered_datasets.keys():
                if dset_name == 'channo': # channo no useful for Timestream
                    if destroy_self:
                        self.delete_a_dataset(dset_name)
                    continue
                else:
                    axis_order = self.feed_ordered_datasets[dset_name]
                    ts.create_feed_ordered_dataset(dset_name, dset.data, axis_order)
            else:
                if dset.common:
                    ts.create_dataset(dset_name, data=dset)
                elif dset.distributed:
                    ts.create_dataset(dset_name, data=dset.data, shape=dset.shape, dtype=dset.dtype, distributed=True, distributed_axis=dset.distributed_axis)

            # copy attrs of this dset
            memh5.copyattrs(dset.attrs, ts[dset_name].attrs)

            if destroy_self:
                self.delete_a_dataset(dset_name)

        # resume hints of self, error may happen if not do so
        if destroy_self:
            for key in self.__class__.__dict__.keys():
                if re.match(self.hints_pattern, key):
                    setattr(self, key, self.__class__.__dict__[key])

        # redistribute self to original axis
        if keep_dist_axis:
            self.redistribute(original_dist_axis)

        return ts


    def _copy_a_common_dataset(self, name, other):
        ### copy a common dataset from `other` to self
        if name == 'channo' and not other._subset_channel_select is None:
            self.create_dataset(name, data=other._subset_channel_select)
            memh5.copyattrs(other[name].attrs, self[name].attrs)
        else:
            super(RawTimestream, self)._copy_a_common_dataset(name, other)
