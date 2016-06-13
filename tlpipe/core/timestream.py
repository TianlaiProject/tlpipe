import itertools
import numpy as np
import container
from caput import memh5
from caput import mpiutil


class Timestream(container.BasicTod):
    """Container class for the timestream data.

    This timestream data container is to hold time stream data that has polarization
    and baseline separated from the channelpair in the raw timestream.
    """

    _main_data_name = 'vis'
    _main_data_axes = ('time', 'frequency', 'polarization', 'baseline')
    _main_time_ordered_datasets = ('vis', 'sec1970', 'jul_date')
    _time_ordered_datasets = _main_time_ordered_datasets + ('weather',)
    _time_ordered_attrs = ()


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

    _feed_select = None

    def baseline_select(self, value):
        """Select data to be loaded from input files along the baseline axis.

        Parameters
        ----------
        value : tuple or list
            If a tuple, which will be created as a slice(start, stop, step) object,
            so it can have one to three elements (integers or None); if a list, its
            elements must be strictly increasing non-negative integers, data in
            these positions will be selected.

        """
        self.data_select('baseline', value)

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

        feeds = np.sort(feeds)
        if corr == 'auto':
            bls = [ {fd} for fd in feeds ]
        elif corr == 'cross':
            bls = [ {fd1, fd2} for fd1, fd2 in itertools.combinations(feeds, 2) ]
        elif corr == 'all':
            bls = [ {fd1, fd2} for fd1, fd2 in itertools.combinations_with_replacement(feeds, 2) ]
        else:
            raise ValueError('Unknown correlation type %s' % corr)

        # get blorder info from the first input file
        blorder = self.infiles[0]['blorder']
        blorder = [ set(bl) for bl in blorder ]

        # baseline indices
        indices = { blorder.index(bl) for bl in bls }
        indices = sorted(list(indices))

        self.data_select('baseline', indices)

        self._feed_select = feeds


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
        elif name == 'pol':
            self._load_a_special_common_dataset(name, 'polarization')
        elif name == 'blorder':
            self._load_a_special_common_dataset(name, 'channelpair')
        elif name == 'feedno' and not self._feed_select is None:
            self.create_dataset(name, data=self._feed_select)
        elif name in ('polerr', 'feedpos', 'antpointing') and not self._feed_select is None:
            fh = self.infiles[0]
            feedno = fh['feedno'][:].tolist()
            feed_inds = [ feedno.index(fd) for fd in self._feed_select ]
            if name == 'antpointing':
                self.create_dataset(name, data=fh[name][:, feed_inds])
            else:
                self.create_dataset(name, data=fh[name][feed_inds])

        else:
            super(Timestream, self)._load_a_common_dataset(name)


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
    def pol(self):
        """Return the pol dataset for convenient use."""
        try:
            return self['pol']
        except KeyError:
            raise KeyError('pol does not exist, try to load it first')


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
            super(Timestream, self).redistribute(dist_axis)

            if 'time' == self.main_data_axes[axis]:
                self.dataset_distributed_to_common('freq')
                self.dataset_distributed_to_common('pol')
                self.dataset_distributed_to_common('blorder')

            # distribute freq
            elif 'frequency' == self.main_data_axes[axis]:
                self.dataset_common_to_distributed('freq', distributed_axis=0)
                self.dataset_distributed_to_common('pol')
                self.dataset_distributed_to_common('blorder')

            # distribute pol
            elif 'polarization' == self.main_data_axes[axis]:
                self.dataset_common_to_distributed('pol', distributed_axis=0)
                self.dataset_distributed_to_common('freq')
                self.dataset_distributed_to_common('blorder')

            # distribute blorder
            elif 'baseline' == self.main_data_axes[axis]:
                self.dataset_common_to_distributed('blorder', distributed_axis=0)
                self.dataset_distributed_to_common('freq')
                self.dataset_distributed_to_common('pol')

    def check_status(self):
        """Check that data hold in this container is consistent. """

        # basic checks
        super(Timestream, self).check_status()

        # additional checks
        if 'time' == self.main_data_axes[self.main_data_dist_axis]:
            for name in ('freq', 'pol', 'blorder'):
                if not self[name].common:
                    raise RuntimeError('Dataset %s should be common when time is the distributed axis' % name)
        elif 'frequency' == self.main_data_axes[self.main_data_dist_axis]:
            if not self['freq'].distributed:
                raise RuntimeError('Dataset freq should be distributed when frequency is the distributed axis')
            if not self['pol'].common:
                raise RuntimeError('Dataset pol should be common when frequency is the distributed axis')
            if not self['blorder'].common:
                raise RuntimeError('Dataset blorder should be common when frequency is the distributed axis')
        elif 'polarization' == self.main_data_axes[self.main_data_dist_axis]:
            if not self['freq'].common:
                raise RuntimeError('Dataset freq should be common when polarization is the distributed axis')
            if not self['pol'].distributed:
                raise RuntimeError('Dataset pol should be distributed when polarization is the distributed axis')
            if not self['blorder'].common:
                raise RuntimeError('Dataset blorder should be common when polarization is the distributed axis')
        elif 'baseline' == self.main_data_axes[self.main_data_dist_axis]:
            if not self['freq'].common:
                raise RuntimeError('Dataset freq should be common when baseline is the distributed axis')
            if not self['pol'].common:
                raise RuntimeError('Dataset pol should be common when baseline is the distributed axis')
            if not self['blorder'].distributed:
                raise RuntimeError('Dataset blorder should be distributed when baseline is the distributed axis')
