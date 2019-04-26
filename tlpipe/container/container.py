"""Basic time ordered data container."""

import re
import glob
import pickle
import posixpath
import itertools
import warnings
from copy import deepcopy
import numpy as np
import h5py
from caput import mpiarray
from caput import memh5
from caput import mpiutil
from tlpipe.utils import progress


def _to_slice_obj(lst):
    ### convert a list to a slice object if possible
    if len(lst) == 0:
        return slice(0, 0)
    elif len(lst) == 1:
        return slice(lst[0], lst[0]+1)
    else:
        d = np.diff(lst)
        if np.all(d == d[0]):
            return slice(lst[0], lst[-1]+d[0], d[0])
        else:
            return lst


def ensure_file_list(files):
    """Tries to interpret the input as a sequence of files

    Expands file name wild-cards ("globs") and casts sequences to a list.

    """

    if memh5.is_group(files):
        files = [files]
    elif isinstance(files, basestring):
        fls = sorted(glob.glob(files))
        if len(fls) == 0:
            files = [ files ]
        else:
            files = fls
    elif hasattr(files, '__iter__'):
        # Copy the sequence and make sure it's mutable.
        files = list(files)
    else:
        raise ValueError('Input could not be interpreted as a list of files.')
    return files


def check_axis(axis, axes):
    """Check a given axis is valid.

    Parameters
    ----------
    axis : string or integer
        The axis to be checked.
    axes : tuple of strings
        A tuple of axis names.

    Returns
    -------
    valid_axis : interger
        A valid axis.

    """
    naxis = len(axes)
    # Process if axis is a string
    if isinstance(axis, basestring):
        try:
            valid_axis = axes.index(axis)
        except ValueError:
            raise ValueError('Axis %s does not exist' % axis)
    # Process if axis is an integer
    elif isinstance(axis, int):
        # Deal with negative axis index
        if axis < 0:
            valid_axis = naxis + axis
        else:
            valid_axis = axis

    if 0 <= valid_axis and valid_axis < naxis:
        return valid_axis
    else:
        raise ValueError('Invalid axis %d' % axis)


class BasicTod(memh5.MemDiskGroup):
    """Basic time ordered data container.

    Inherits from :class:`caput.memh5.MemDiskGroup`.

    Basic one-level time ordered data container that allows any number of
    datasets in the root group but no nesting.

    This container is intended to be an base class for other concrete data
    classes, so only basic input/output and a limited operations are provided.
    Usually you should not use this class directly, use a concrete sub-class
    instead.

    Parameters
    ----------
    files : None, string or list of strings
        File name or a list of file names that data will be loaded from. No files
        if None. Default None.
    mode : string, optional
        In which mode to open the input files. Default 'r' to open files read only.
    start : integer, optional
        Starting time point to load. Non-negative integer is relative to the first
        time point of the first file, negative integer is relative to the last time
        point of the last file. Default 0 is from the start of the first file.
    stop : None or integer, optional
        Stopping time point to load. Non-negative integer is relative to the first
        time point of the first file, negative integer is relative to the last time
        point of the last file. Default None is to the end of the last file.
    dist_axis : string or integer, optional
        Axis along which the main data is distributed.
    use_hints : bool, optional
        If True, will try to use the hints in the first file of `files` to
        construct this class. Default True.
    comm : None or MPI.Comm, optional
        MPI Communicator to distributed over. Default None to use mpiutil._comm.

    """

    _main_data_name_ = None
    _main_data_axes_ = () # time should be the first axis
    _main_axes_ordered_datasets_ = {_main_data_name_: (0,)}
    _time_ordered_datasets_ = {_main_data_name_: (0,)}
    _time_ordered_attrs_ = {}


    def __init__(self, files=None, mode='r', start=0, stop=None, dist_axis=0, use_hints=True, comm=None):

        super(BasicTod, self).__init__(data_group=None, distributed=True, comm=comm)

        self.nproc = 1 if self.comm is None else self.comm.size
        self.rank = 0 if self.comm is None else self.comm.rank
        self.rank0 = True if self.rank == 0 else False

        # hints pattern to match hint class attributes defined above
        self.hints_pattern = re.compile(r"(^_[^_]+_$)|(^_[^_]\w*[^_]_$)")
        # read and set hints from the first file if use_hints is True
        if files is not None and use_hints:
            fl = ensure_file_list(files)[0]
            with h5py.File(fl, 'r') as f:
                if 'hints' in f.attrs.iterkeys():
                    hints = pickle.loads(f.attrs['hints'])
                    for key, val in hints.iteritems():
                        setattr(self, key, val)

        self.infiles_mode = mode
        # self.infiles will be a list of opened hdf5 file handlers
        self.infiles, self.main_data_start, self.main_data_stop = self._select_files(files, self.main_data_name, start, stop)
        self.num_infiles = len(self.infiles)
        self.main_data_dist_axis = check_axis(dist_axis, self.main_data_axes)

        self.main_data_select = [ slice(0, None, None) for i in self._main_data_axes_ ]
        self.subset_data_select = [ slice(0, None, None) for i in self._main_data_axes_ ]

    def __del__(self):
        """Closes the opened file handlers."""
        for fh in self.infiles:
            fh.close()

    def _select_files(self, files, dset_name, start=0, stop=None):
        ### select the needed files from `files` which contain time ordered data from `start` to `stop`
        assert dset_name in self.time_ordered_datasets.keys(), '%s is not a time ordered dataset' % dset_name

        if files is None:
            return [], 0, 0

        files = ensure_file_list(files)
        if len(files) == 0:
            return [], 0, 0

        num_ts = []
        for fh in files:
            with h5py.File(fh, 'r') as f:
                num_ts.append(f[dset_name].shape[0])

        nt = sum(num_ts) # total length of the first axis along different files

        tmp_start = start if start >=0 else start + nt
        if tmp_start >= 0 and tmp_start < nt:
            start = tmp_start
        else:
            raise ValueError('Invalid start %d for nt = %d' % (start, nt))
        stop = nt if stop is None else stop
        tmp_stop = stop if stop >=0 else stop + nt
        if tmp_stop >= 0 and tmp_stop <= nt:
            stop = tmp_stop
        else:
            raise ValueError('Invalid stop %d for nt = %d' % (stop, nt))
        if start > stop:
            raise ValueError('Invalid start %d and stop %d for nt = %d' % (start, stop, nt))

        cum_num_ts = np.cumsum(num_ts)
        sf = np.searchsorted(cum_num_ts, start, side='right') # start file index
        new_start = start if sf == 0 else start - cum_num_ts[sf-1] # start relative the selected first file
        ef = np.searchsorted(cum_num_ts, stop, side='left') # stop file index, included
        new_stop = stop if sf == 0 else stop - cum_num_ts[sf-1] # stop relative the selected first file

        # open all selected files
        files = [ h5py.File(fh, self.infiles_mode) for fh in files[sf:ef+1] ]

        return files, new_start, new_stop

    def _gen_files_map(self, num_ts, start=0, stop=None):
        ### generate files map, i.e., a list of (file_idx, start, stop)
        ### num_ts: a list of number of time points allocated to each file
        ### start, stop are all relative to the first file

        nt = np.sum(num_ts) # total number of time points
        stop = nt if stop is None else stop
        assert (start <= stop and stop - start <= nt), 'Invalid start %d and stop %d' % (start, stop)

        lts, _, _ = mpiutil.split_all(stop-start, comm=self.comm) # selected total length distributed among different procs
        cum_lts = np.cumsum(lts) # cumsum of lengths by all procs
        cum_lts = (cum_lts + start).tolist()
        tmp_cum_lts = [start] + cum_lts
        cum_num_ts = np.cumsum(num_ts).tolist() # cumsum of lengths of all files
        tmp_cum_num_ts = [0] + cum_num_ts

        # start and stop (included) file indices owned by all procs
        sf, ef = np.searchsorted(cum_num_ts, tmp_cum_lts[:-1], side='right'), np.searchsorted(cum_num_ts, tmp_cum_lts[1:], side='left')

        # allocation interval by all procs
        intervals = sorted(list(set([start] + cum_lts + cum_num_ts[:-1] + [stop])))
        intervals = [ (intervals[i], intervals[i+1]) for i in xrange(len(intervals)-1) ]
        cum_num_lf_ind = np.cumsum([0] + (ef - sf + 1).tolist())
        # local intervals owned by this proc
        lits = intervals[cum_num_lf_ind[self.rank]: cum_num_lf_ind[self.rank+1]]
        # infiles_map: a list of (file_idx, start, stop)
        files_map = []
        for idx, fi in enumerate(range(sf[self.rank], ef[self.rank]+1)):
            files_map.append((fi, lits[idx][0]-tmp_cum_num_ts[fi], lits[idx][1]-tmp_cum_num_ts[fi]))

        return files_map

    def _get_input_info(self, dset_name, start=0, stop=None):
        ### get data shape and type and infile_map of time ordered datasets
        ### start, stop are all relative to the first file
        ### shape and type are get from the first file and suppose they are the same in all self.infiles

        num_ts = []
        for fh in self.infiles:
            num_ts.append(fh[dset_name].shape[0])

        if stop is None:
            stop = sum(num_ts) # total length of the first axis along different files
        infiles_map = self._gen_files_map(num_ts, start, stop)

        # get shape and type info from the first file
        tmp_shape = self.infiles[0][dset_name].shape
        dset_shape = ((stop-start),) + tmp_shape[1:]
        dset_type= self.infiles[0][dset_name].dtype

        return dset_shape, dset_type, infiles_map


    @property
    def main_data(self):
        """The main data in the container which is a convenience for self[self.main_data_name]."""
        try:
            return self[self.main_data_name]
        except KeyError:
            raise KeyError('Main data %s does not exist, try to load the main data first' % self.main_data_name)

    @property
    def main_data_name(self):
        """Main data in the data container."""
        return self._main_data_name_

    @main_data_name.setter
    def main_data_name(self, value):
        if isinstance(value, basestring):
            self._main_data_name_ = value
        else:
            raise ValueError('Attribute main_data_name must be a string')

    @property
    def main_data_axes(self):
        """Axes of the main data."""
        return self._main_data_axes_

    @main_data_axes.setter
    def main_data_axes(self, value):
        if isinstance(value, basestring):
            self._main_data_axes_ = (value,)
        elif hasattr(value, '__iter__'):
            for val in value:
                if not isinstance(val, basestring):
                    raise ValueError('Attribute main_data_axes must be a tuple of strings')
            self._main_data_axes_ = tuple(value)
        else:
            raise ValueError('Attribute main_data_axes must be a tuple of strings')

    @property
    def dist_axis(self):
        """Convenience for self.main_data_dist_axis."""
        return self.main_data_dist_axis

    @property
    def dist_axis_name(self):
        """Name of self.main_data_dist_axis."""
        return self._main_data_axes_[self.main_data_dist_axis]

    @property
    def main_axes_ordered_datasets(self):
        """Datasets that have axis aligned with the main data."""
        return self._main_axes_ordered_datasets_

    @property
    def main_time_ordered_datasets(self):
        """Datasets that have the first axis aligned with the main data."""
        return { key: val for key, val in self.main_axes_ordered_datasets.items() if 0 in val }

    @property
    def time_ordered_datasets(self):
        """Time ordered datasets."""
        self._time_ordered_datasets_.update(self.main_time_ordered_datasets)
        return self._time_ordered_datasets_

    @property
    def time_ordered_attrs(self):
        """Attributes that are different in different files."""
        return self._time_ordered_attrs_

    @time_ordered_attrs.setter
    def time_ordered_attrs(self, value):
        if isinstance(value, basestring):
            self._time_ordered_attrs_ = {value}
        elif hasattr(value, '__iter__'):
            for val in value:
                if not isinstance(val, basestring):
                    raise ValueError('Attribute time_ordered_attrs must be a set of strings')
            self._time_ordered_attrs_ = set(value)
        else:
            raise ValueError('Attribute time_ordered_attrs must be a set of strings')


    def _data_select(self, var, axis, value):
        # inner method for data select
        if isinstance(value, tuple):
            var[axis] = slice(*value)
        elif isinstance(value, list):
            if sorted(value) != value:
                raise TypeError("Indexing elements must be in increasing order")
            if value[0] < 0:
                raise TypeError("Indexing elements must be non-negative integers")
            var[axis] = value
        else:
            raise ValueError('Unsupported data selection %s' % value)

    def data_select(self, axis, value):
        """Select data to be loaded from input files along the specified axis.

        You can use this method to select data to be loaded from input files along
        an arbitrary axis except the first axis (which is not implemented yet).

        Parameters
        ----------
        axis : string or integer
            The distribute axis.
        value : tuple or list
            If a tuple, which will be created as a slice(start, stop, step) object,
            so it can have one to three elements (integers or None); if a list, its
            elements must be strictly increasing non-negative integers, data in
            these positions will be selected.

        """
        axis = check_axis(axis, self.main_data_axes)
        if axis == 0:
            if value != (0, None):
                raise NotImplementedError('Select data to be loaded along the first axis is not implemented yet')
        self._data_select(self.main_data_select, axis, value)

    def subset_select(self, axis, value):
        """Select a subset of the data along the specified axis.

        Parameters
        ----------
        axis : string or integer
            The distribute axis.
        value : tuple or list
            If a tuple, which will be created as a slice(start, stop, step) object,
            so it can have one to three elements (integers or None); if a list, its
            elements must be strictly increasing non-negative integers, data in
            these positions will be selected.

        """
        axis = check_axis(axis, self.main_data_axes)
        self._data_select(self.subset_data_select, axis, value)


    def _load_a_common_attribute(self, name):
        ### load a common attribute from the first file

        fh = self.infiles[0]
        self.attrs[name] = fh.attrs[name]

    def _load_a_time_ordered_attribute(self, name):
        ### load a time ordered attribute from all the file

        self.attrs[name] = []
        for fh in self.infiles:
            self.attrs[name].append(fh.attrs[name])

    def _load_an_attribute(self, name):
        ### load an attribute (either a commmon or a time ordered)

        if name in self.time_ordered_attrs:
            self._load_a_time_ordered_attribute(name)
        else:
            self._load_a_common_attribute(name)

    def _load_a_common_dataset(self, name):
        ### load a common dataset from the first file

        fh = self.infiles[0]
        dset = fh[name]
        self.create_dataset(name, data=dset, shape=dset.shape, dtype=dset.dtype)
        # copy attrs of this dset
        memh5.copyattrs(dset.attrs, self[name].attrs)

    def _load_a_main_axes_ordered_dataset(self, name):
        ### load a main_axes_ordered_dataset from the first file if it is not time
        ### ordered, else from all files, distribute the data along
        ### self.main_data_dist_axis if data has this axis

        if name in self.main_time_ordered_datasets.keys():
            dset_shape, dset_type, infiles_map = self._get_input_info(name, self.main_data_start, self.main_data_stop)
            if len(infiles_map) == 0:
                first_start = 0
                last_stop = 0
            else:
                first_start = infiles_map[0][1]
                last_stop = infiles_map[-1][2]
            first_start = mpiutil.bcast(first_start, root=0, comm=self.comm) # start form the first file
            last_stop = mpiutil.bcast(last_stop, root=self.nproc-1, comm=self.comm) # stop from the last file
        else:
            dset_shape, dset_type = self.infiles[0][name].shape, self.infiles[0][name].dtype

        axes = list(self.main_axes_ordered_datasets[name])
        axes = axes + [None] * (len(dset_shape) - len(axes)) # complete the un-write axes
        if 0 in axes:
            ti = axes.index(0) # index of 0 axis
        if self.main_data_dist_axis in axes:
            di = axes.index(self.main_data_dist_axis) # index of dist axis
        fsel = [ ( self.main_data_select[a] if a is not None else slice(0, None, None) ) for a in axes ] # for data in file
        msel = [ slice(0, None, None) ] # for data in memory
        shp = tuple([ len(np.arange(ni)[si]) for (ni, si) in zip(dset_shape, fsel) ])

        if self.main_data_dist_axis in axes:
            # load as a distributed dataset
            # create a distributed dataset to hold the data to be load
            self.create_dataset(name, shape=shp, dtype=dset_type, distributed=True, distributed_axis=di)
            # copy attrs of this dset
            memh5.copyattrs(self.infiles[0][name].attrs, self[name].attrs)

            if 0 in axes:
                if self.main_data_dist_axis == 0:
                    # load data from all files as a distributed dataset
                    st = 0
                    for fi, start, stop in infiles_map:
                        et = st + (stop - start)
                        fsel[ti] = slice(start, stop)
                        msel[ti] = slice(st, et)
                        st = et
                        fh = self.infiles[fi]
                        if np.prod(self[name].local_data[tuple(msel)].shape) > 0:
                            # only read in data if non-empty, may get error otherwise
                            fsel = [  ( _to_slice_obj(s) if isinstance(s, list) else s ) for s in fsel ]
                            self[name].local_data[tuple(msel)] = fh[name][tuple(fsel)]

                else:
                    # load data from all files as a distributed dataset
                    linds = mpiutil.mpilist(np.arange(dset_shape[di])[fsel[di]].tolist(), comm=self.comm)
                    fsel[di] = linds

                    # load data from all files
                    st = 0
                    for fi, fh in enumerate(self.infiles):
                        num_ts = fh[name].shape[0]
                        if self.num_infiles == 1:
                            et = st + last_stop - first_start
                            fsel[ti] = slice(first_start, last_stop)
                        elif self.num_infiles > 1:
                            if fi == 0:
                                et = st + (num_ts - first_start)
                                fsel[ti] = slice(first_start, None)
                            elif fi == self.num_infiles-1:
                                et = st + last_stop
                                fsel[ti] = slice(0, last_stop)
                            else:
                                et = st + num_ts
                                fsel[ti] = slice(0, None)

                        msel[ti] = slice(st, et)
                        st = et
                        if np.prod(self[name].local_data[tuple(msel)].shape) > 0:
                            fsel = [  ( _to_slice_obj(s) if isinstance(s, list) else s ) for s in fsel ]
                            self[name].local_data[tuple(msel)] = fh[name][tuple(fsel)]

            else:
                if self.main_data_dist_axis == 0:
                    raise RuntimeError('Something wrong happened, this would never occur')
                else:
                    # load data from the first file as a distributed dataset
                    linds = mpiutil.mpilist(np.arange(dset_shape[di])[fsel[di]].tolist(), comm=self.comm)
                    fsel[di] = linds
                    if np.prod(self[name].local_data.shape) > 0:
                        fsel = [  ( _to_slice_obj(s) if isinstance(s, list) else s ) for s in fsel ]
                        self[name].local_data[:] = self.infiles[0][name][tuple(fsel)]

        else:
            # load as a common dataset
            # create a common dataset to hold the data to be load
            self.create_dataset(name, shape=shp, dtype=dset_type)
            # copy attrs of this dset
            memh5.copyattrs(self.infiles[0][name].attrs, self[name].attrs)

            if 0 in axes:
                # load data from all files
                st = 0
                for fi, fh in enumerate(self.infiles):
                    num_ts = fh[name].shape[0]
                    if self.num_infiles == 1:
                        et = st + last_stop - first_start
                        fsel[ti] = slice(first_start, last_stop)
                    elif self.num_infiles > 1:
                        if fi == 0:
                            et = st + (num_ts - first_start)
                            fsel[ti] = slice(first_start, None)
                        elif fi == self.num_infiles-1:
                            et = st + last_stop
                            fsel[ti] = slice(0, last_stop)
                        else:
                            et = st + num_ts
                            fsel[ti] = slice(0, None)

                    msel[ti] = slice(st, et)
                    st = et
                    if np.prod(self[name][tuple(msel)].shape) > 0:
                        fsel = [  ( _to_slice_obj(s) if isinstance(s, list) else s ) for s in fsel ]
                        self[name][tuple(msel)] = fh[name][tuple(fsel)] # not a distributed dataset

            else:
                # load data from the first file
                if np.prod(self[name][:].shape) > 0:
                    fsel = [  ( _to_slice_obj(s) if isinstance(s, list) else s ) for s in fsel ]
                    self[name][:] = self.infiles[0][name][tuple(fsel)] # not a distributed dataset

    def _load_a_time_ordered_dataset(self, name):
        ### load a time ordered dataset (except those also in main_axes_ordered_datasets) from all files

        dset_shape, dset_type, infiles_map = self._get_input_info(name, 0, None)
        axes = self.time_ordered_datasets[name]
        ti = axes.index(0) # index of 0 axis
        fsel = [ slice(0, None, None) for i in dset_shape ] # for data in file
        msel = [ slice(0, None, None) for i in dset_shape ] # for data in memory

        if self.main_data_dist_axis == 0:
            # load data as a distributed dataset
            # create a distributed dataset to hold the data to be load
            self.create_dataset(name, shape=dset_shape, dtype=dset_type, distributed=True, distributed_axis=ti)
            # copy attrs of this dset
            memh5.copyattrs(self.infiles[0][name].attrs, self[name].attrs)

            # load data from all files as a distributed dataset
            st = 0
            for fi, start, stop in infiles_map:
                et = st + (stop - start)
                fsel[ti] = slice(start, stop)
                msel[ti] = slice(st, et)
                st = et
                fh = self.infiles[fi]
                if np.prod(self[name].local_data[tuple(msel)].shape) > 0:
                    # only read in data if non-empty, may get error otherwise
                    fsel = [  ( _to_slice_obj(s) if isinstance(s, list) else s ) for s in fsel ]
                    self[name].local_data[tuple(msel)] = fh[name][tuple(fsel)]
        else:
            # load data as a common dataset
            # create a common dataset to hold the data to be load
            self.create_dataset(name, shape=dset_shape, dtype=dset_type)
            # copy attrs of this dset
            memh5.copyattrs(self.infiles[0][name].attrs, self[name].attrs)

            # load data from all files as a common dataset
            st = 0
            for fi, fh in enumerate(self.infiles):
                num_ts = fh[name].shape[0]
                if self.num_infiles == 1:
                    et = st + last_stop - first_start
                    fsel[ti] = slice(first_start, last_stop)
                elif self.num_infiles > 1:
                    if fi == 0:
                        et = st + (num_ts - first_start)
                        fsel[ti] = slice(first_start, None)
                    elif fi == self.num_infiles-1:
                        et = st + last_stop
                        fsel[ti] = slice(0, last_stop)
                    else:
                        et = st + num_ts
                        fsel[ti] = slice(0, None)

                msel[ti] = slice(st, et)
                st = et
                if np.prod(self[name][tuple(msel)].shape) > 0:
                    fsel = [  ( _to_slice_obj(s) if isinstance(s, list) else s ) for s in fsel ]
                    self[name][tuple(msel)] = fh[name][tuple(fsel)] # not a distributed dataset

    def _load_a_dataset(self, name):
        ### load a dataset (either a commmon or a main axis ordered or a time ordered)

        if name in self.main_axes_ordered_datasets.keys():
            self._load_a_main_axes_ordered_dataset(name)
        elif name in self.time_ordered_datasets.keys():
            self._load_a_time_ordered_dataset(name)
        else:
            self._load_a_common_dataset(name)


    def load_common_attrs(self):
        """Load common attributes from the first file."""
        if self.num_infiles == 0:
            warnings.warn('No input file')
            return

        fh = self.infiles[0]
        # read in top level common attrs
        for attr_name in fh.attrs.iterkeys():
            if attr_name not in self.time_ordered_attrs and attr_name != 'hints':
                self._load_a_common_attribute(attr_name)

    def load_time_ordered_attrs(self):
        """Load time ordered attributes from all files."""
        if self.num_infiles == 0:
            warnings.warn('No input file')
            return

        fh = self.infiles[0]
        for attr_name in fh.attrs.iterkeys():
            if attr_name in self.time_ordered_attrs:
                self._load_a_time_ordered_attribute(attr_name)

    def load_common_datasets(self):
        """Load common datasets from the first file."""
        if self.num_infiles == 0:
            warnings.warn('No input file')
            return

        fh = self.infiles[0]
        # read in top level common datasets
        for dset_name in fh.iterkeys():
            if (dset_name not in self.main_axes_ordered_datasets.keys()) and (dset_name not in self.time_ordered_datasets.keys()):
                self._load_a_common_dataset(dset_name)

    def load_time_ordered_datasets(self):
        """Load time ordered datasets (excepts those also in main_axes_ordered_datasets) from all files."""
        if self.num_infiles == 0:
            warnings.warn('No input file')
            return

        fh = self.infiles[0]
        for dset_name in fh.iterkeys():
            if (dset_name in self.time_ordered_datasets.keys()) and (not dset_name in self.main_axes_ordered_datasets.keys()):
                self._load_a_time_ordered_dataset(dset_name)

    def load_main_axes_ordered_data(self):
        """Load main axes ordered dataset."""
        if self.num_infiles == 0:
            warnings.warn('No input file')
            return

        fh = self.infiles[0]
        for dset_name in fh.iterkeys():
            if dset_name in self.main_axes_ordered_datasets.keys():
                self._load_a_main_axes_ordered_dataset(dset_name)

    def load_main_data(self):
        """Load main data from all files."""
        if self.num_infiles == 0:
            warnings.warn('No input file')
            return

        self._load_a_main_axes_ordered_dataset(self.main_data_name)

    def load_main_axes_excl_main_data(self):
        """Load main axes ordered datasets (exclude the main data)."""
        if self.num_infiles == 0:
            warnings.warn('No input file')
            return

        fh = self.infiles[0]
        for dset_name in fh.iterkeys():
            if dset_name in self.main_axes_ordered_datasets.keys() and dset_name != self.main_data_name:
                self._load_a_main_axes_ordered_dataset(dset_name)

    def load_all(self):
        """Load all attributes and datasets from files."""
        if self.num_infiles == 0:
            warnings.warn('No input file')
            return

        fh = self.infiles[0]
        # load in top level attrs
        for attr_name in fh.attrs.iterkeys():
            if attr_name != 'hints':
                self._load_an_attribute(attr_name)

        # load in top level datasets
        for dset_name in fh.iterkeys():
            self._load_a_dataset(dset_name)


    def group_name_allowed(self, name):
        """No groups are exposed to the user. Returns ``False``."""
        return False

    def dataset_name_allowed(self, name):
        """Datasets may only be created and accessed in the root level group.

        Returns ``True`` is *name* is a path in the root group i.e. '/dataset'.

        """

        parent_name, name = posixpath.split(name)
        return True if parent_name == '/' else False

    # def attrs_name_allowed(self, name):
    #     """Whether to allow the access of the given root level attribute."""
    #     return True if name not in self.time_ordered_attrs else False


    def create_time_ordered_dataset(self, name, data, axis_order=(0,), recreate=False, copy_attrs=False):
        """Create a time ordered dataset.

        Parameters
        ----------
        name : string
            Name of the dataset.
        data : np.ndarray or MPIArray
            The data to create a dataset.
        axis_order : tuple
            A tuple with the index 0 denotes time axis of the created dataset.
        recreate : bool, optional
            If True will recreate a dataset with this name if it already exists,
            else a RuntimeError will be rasised. Default False.
        copy_attrs : bool, optional
            If True, when recreate the dataset, its original attributes will be
            copyed to the new dataset, else no copy is done. Default Fasle.

        """

        time_axis = axis_order.index(0)

        if not name in self.iterkeys():
            if self.main_data_dist_axis == 0:
                self.create_dataset(name, data=data, distributed=True, distributed_axis=time_axis)
            else:
                self.create_dataset(name, data=data)
        else:
            if recreate:
                if copy_attrs:
                    attr_dict = {} # temporarily save attrs of this dataset
                    memh5.copyattrs(self[name].attrs, attr_dict)
                del self[name]
                if self.main_data_dist_axis == 0:
                    self.create_dataset(name, data=data, distributed=True, distributed_axis=time_axis)
                else:
                    self.create_dataset(name, data=data)
                if copy_attrs:
                    memh5.copyattrs(attr_dict, self[name].attrs)
            else:
                raise RuntimeError('Dataset %s already exists' % name)

        self.time_ordered_datasets[name] = axis_order

    def create_main_axis_ordered_dataset(self, axis, name, data, axis_order, recreate=False, copy_attrs=False, check_align=True):
        """Create a `axis_name` ordered dataset.

        Parameters
        ----------
        axis : string or integer or tuple
            The distribute axis.
        name : string
            Name of the dataset.
        data : np.ndarray or MPIArray
            The data to create a dataset.
        axis_order : tuple
            A tuple denotes the corresponding axis of the created dataset.
        recreate : bool, optional
            If True will recreate a dataset with this name if it already exists,
            else a RuntimeError will be rasised. Default False.
        copy_attrs : bool, optional
            If True, when recreate the dataset, its original attributes will be
            copyed to the new dataset, else no copy is done. Default Fasle.
        check_align : bool, optional
            If True, check main axis of data align with that of the main data
            before dataset creating, otherwise create dataset without axis align
            checking, this may cause the created dataset does not align with the
            main data. Default True.

        """

        if isinstance(data, mpiarray.MPIArray):
            shape = data.global_shape
        else:
            shape = data.shape

        if not isinstance(axis, tuple):
            axis = (axis,)
        axes = ()
        for ax in axis:
            tmp_ax = check_axis(ax, self.main_data_axes)
            axes = axes + (tmp_ax,)

            dist_axis = axis_order.index(tmp_ax)
            if check_align and shape[dist_axis] != self.main_data.shape[tmp_ax]:
                axis_name = self.main_data_axes[tmp_ax]
                raise ValueError('%s axis does not align with main data, can not create a %s ordered dataset %s' % (axis_name.capitalize(), axis_name, name))

        if not name in self.iterkeys():
            if self.main_data_dist_axis in axes:
                self.create_dataset(name, data=data, distributed=True, distributed_axis=axis_order.index(self.main_data_dist_axis))
            else:
                self.create_dataset(name, data=data)
        else:
            if recreate:
                if copy_attrs:
                    attr_dict = {} # temporarily save attrs of this dataset
                    memh5.copyattrs(self[name].attrs, attr_dict)
                del self[name]
                if self.main_data_dist_axis in axes:
                    self.create_dataset(name, data=data, distributed=True, distributed_axis=axis_order.index(self.main_data_dist_axis))
                else:
                    self.create_dataset(name, data=data)
                if copy_attrs:
                    memh5.copyattrs(attr_dict, self[name].attrs)
            else:
                raise RuntimeError('Dataset %s already exists' % name)

        self.main_axes_ordered_datasets[name] = axis_order

    def create_main_data(self, data, recreate=False, copy_attrs=False):
        """Create or recreate a main dataset.

        Parameters
        ----------
        data : np.ndarray or MPIArray
            The data to create a dataset.
        recreate : bool, optional
            If True will recreate a dataset with this name if it already exists,
            else a RuntimeError will be rasised. Default False.
        copy_attrs : bool, optional
            If True, when recreate the dataset, its original attributes will be
            copyed to the new dataset, else no copy is done. Default Fasle.

        """

        axis = tuple(xrange(len(data.shape)))
        name = self.main_data_name
        axis_order = axis

        self.create_main_axis_ordered_dataset(axis, name, data, axis_order, recreate, copy_attrs, False)


    def create_main_time_ordered_dataset(self, name, data, axis_order=(0,), recreate=False, copy_attrs=False, check_align=True):
        """Create a main type time ordered dataset.

        Parameters
        ----------
        name : string
            Name of the dataset.
        data : np.ndarray or MPIArray
            The data to create a dataset.
        axis_order : tuple
            A tuple with the index of 0 denotes time axis of the created dataset.
        recreate : bool, optional
            If True will recreate a dataset with this name if it already exists,
            else a RuntimeError will be rasised. Default False.
        copy_attrs : bool, optional
            If True, when recreate the dataset, its original attributes will be
            copyed to the new dataset, else no copy is done. Default Fasle.
        check_align : bool, optional
            If True, check time axis of data align with that of the main data
            before dataset creating, otherwise create dataset without axis align
            checking, this may cause the created dataset does not align with the
            main data. Default True.

        """

        self.create_main_axis_ordered_dataset(0, name, data, axis_order, recreate, copy_attrs, check_align)


    def delete_a_dataset(self, name, reserve_hint=True):
        """Delete a dataset. If `reserve_hint` is False, also remove it from
        the hint if it is in it.
        """
        if name in self.iterkeys():
            del self[name]
        else:
            warnings.warn('Dataset %s does not exist')

        if not reserve_hint:
            if name in self._main_axes_ordered_datasets_.iterkeys():
                del self._main_axes_ordered_datasets_[name]
            if name in self._time_ordered_datasets_.iterkeys():
                del self._time_ordered_datasets_[name]

    def delete_an_attribute(self, name, reserve_hint=True):
        """Delete an attribute. If `reserve_hint` is False, also remove it from
        the hint if it is in it.
        """
        if name in self.attrs.iterkeys():
            del self.attrs[name]
        else:
            warnings.warn('Attribute %s does not exist')

        if not reserve_hint:
            if name in self._time_ordered_attrs_:
                self._time_ordered_attrs_.remove(name)

    def empty(self, reserve_hints=True):
        """Delete all datasets and attributes in this container. If
        `reserve_hints` is False, also remove them from the hint if they are in it.
        """
        # delete attributes
        attrs_keys = list(self.attrs.iterkeys())
        for attrs_name in attrs_keys:
            self.delete_an_attribute(attrs_name, reserve_hint=reserve_hints)
        # delete datasets
        for dset_name in self.iterkeys():
            self.delete_a_dataset(dset_name, reserve_hint=reserve_hints)

    @property
    def history(self):
        """The analysis history for this data.

        Do not try to add a new history entry by assigning to this property.
        Use :meth:`~BasicTod.add_history` instead.

        Returns
        -------
        history : string

        """
        try:
            return self.attrs['history']
        except KeyError:
            raise KeyError('History does not exist, try to load it first')

    def add_history(self, history=''):
        """Create a new history entry."""

        if self.history and history is not '':
            self.attrs['history'] += ('\n' + history)

    def info(self):
        """List basic information of the data hold by this container."""
        if self.rank0:
            # list the opened files
            print
            print 'Input files:'
            for fh in self.infiles:
                print '  ', fh.filename
            print
            # distributed axis
            print '%s distribution axis: (%d, %s)' % (self.main_data_name, self.main_data_dist_axis, self.main_data_axes[self.main_data_dist_axis])
            print
            # hints for this class
            for key in self.__class__.__dict__.keys():
                if re.match(self.hints_pattern, key):
                    print '%s = %s' % (key, getattr(self, key))
            print
            # list all top level attributes
            for attr_name, attr_val in self.attrs.iteritems():
                print '%s:' % attr_name, attr_val
            # list all top level datasets
            for dset_name, dset in self.iteritems():
                if dset.distributed:
                    print '%s  shape = %s, dist_axis = %d' % (dset_name, dset.shape, dset.distributed_axis)
                else:
                    print '%s  shape = %s' % (dset_name, dset.shape)
                # list its attributes
                for attr_name, attr_val in dset.attrs.iteritems():
                    print '  %s:' % attr_name, attr_val
            print

        mpiutil.barrier(comm=self.comm)

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

        # no MPI, datasets are not distributed
        if self.comm is None:
            return

        axis = check_axis(dist_axis, self.main_data_axes)

        if axis == self.main_data_dist_axis:
            # already the distributed axis, nothing to do
            return
        else:
            # redistribute main data if it exists
            try:
                self.main_data.redistribute(axis)
            except KeyError:
                pass
            self.main_data_dist_axis = axis

            # redistribute other main_axes_ordered_datasets
            for name, val in self.main_axes_ordered_datasets.items():
                if name in self.iterkeys() and name != self.main_data_name:
                    if axis in val:
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            self.dataset_common_to_distributed(name, distributed_axis=val.index(axis))
                    else:
                        if self[name].distributed:
                            self.dataset_distributed_to_common(name)

            # redistribute other time_ordered_datasets
            for name, val in self.time_ordered_datasets.items():
                if name in self.iterkeys() and not name in self.main_axes_ordered_datasets.keys():
                    if axis == 0:
                        self.dataset_common_to_distributed(name, distributed_axis=val.index(0))
                    else:
                        if self[name].distributed:
                            self.dataset_distributed_to_common(name)

    def check_status(self):
        """Check that data hold in this container is consistent.

        One can do any check in this method for a concrete subclass, but very basic
        check has done here in this basic container.
        """

        for name, dset in self.iteritems():
            # check main_axes_ordered_datasets
            if name in self.main_axes_ordered_datasets.keys():
                val = self.main_axes_ordered_datasets[name]
                if self.main_data_dist_axis in val:
                    dist_axis = val.index(self.main_data_dist_axis)
                    if self.comm and (not dset.distributed or dset.distributed_axis != dist_axis):
                        raise RuntimeError('Dataset %s should be distributed along axis %d when %s is the distributed axis' % (name, dist_axis, self.main_data_axes[self.main_data_dist_axis]))
                else:
                    if dset.distributed:
                        raise RuntimeError('Dataset %s should be common when %s is the distributed axis' % (name, self.main_data_axes[self.main_data_dist_axis]))

            # check other time_ordered_datasets
            elif name in self.time_ordered_datasets.keys():
                val = self.time_ordered_datasets[name]
                if self.main_data_dist_axis == 0:
                    if self.comm and not (dset.distributed and dset.distributed_axis == val.index(0)):
                        raise RuntimeError('Dataset %s should be distributed along axis %d when %s is the distributed axis' % (name, val.index(0), self.main_data_axes[self.main_data_dist_axis]))
                else:
                    if not dset.common:
                        raise RuntimeError('Dataset %s should be common when %s is the distributed axis' % (name, self.main_data_axes[self.main_data_dist_axis]))

            # check for common datasets
            else:
                if not dset.common:
                    raise RuntimeError('Dataset %s should be common' % name)

        # check axis are aligned
        for axis in xrange(len(self.main_data_axes)):
            lens = [] # to save the length of axis
            for name, val in self.main_axes_ordered_datasets.items():
                if name in self.items() and axis in val:
                    lens.append(self[name].shape[val.index(axis)])
            num = len(set(lens))
            if num != 0 and num != 1:
                raise RuntimeError('Not all main_axes_ordered_datasets have an aligned %s axis' % self.main_data_axes[axis])


    def _get_output_info(self, dset_name, num_outfiles):
        ### get data shape and type and infile_map of time ordered datasets

        dset_shape = self[dset_name].shape
        dset_type = self[dset_name].dtype

        nt = self[dset_name].shape[0]
        # allocate nt to the given number of files
        num_ts, num_s, num_e = mpiutil.split_m(nt, num_outfiles)

        outfiles_map = self._gen_files_map(num_ts, start=0, stop=None)

        return dset_shape, dset_type, outfiles_map


    def to_files(self, outfiles, exclude=[], check_status=True, write_hints=True, libver='earliest'):
        """Save the data hold in this container to files.

        Parameters
        ----------
        outfiles : string or list of strings
             File name or a list of file names that data will be saved into.
        exclude : list of strings, optional
            Attributes and datasets in this list will be excluded when save to
            files. Default is an empty list, so all data will be saved.
        check_status : bool, optional
            Whether to check data consistency before save to files. Default True.
        write_hints : bool, optional
            If True, will write hint class attributes as a 'hints' attribute to
            files, which will be used to re-construct this class when reading data
            from the saved files. Default True.
        libver : 'latest' or 'earliest', optional
            HDF5 library version settings. 'latest' means that HDF5 will always use
            the newest version of these structures without particular concern for
            backwards compatibility, can be performance advantages. The 'earliest'
            option means that HDF5 will make a best effort to be backwards
            compatible. Default is 'earliest'.

        """

        outfiles = ensure_file_list(outfiles)
        num_outfiles = len(outfiles)

        # first redistribute main_time_ordered_datasets to the first axis
        if self.main_data_dist_axis != 0:
            self.redistribute(0)

        # check data is consistent before save
        if check_status:
            self.check_status()

        # split output files among procs
        for fi, outfile in mpiutil.mpilist(list(enumerate(outfiles)), method='rand', comm=self.comm):
            # first write top level common attrs and datasets to file
            with h5py.File(outfile, 'w', libver=libver) as f:

                # write hints if required
                if write_hints:
                    hint_keys = [ key for key in self.__class__.__dict__.keys() if re.match(self.hints_pattern, key) ]
                    hint_dict = { key: getattr(self, key) for key in hint_keys }
                    f.attrs['hints'] = pickle.dumps(hint_dict)

                # write top level common attrs
                for attrs_name, attrs_value in self.attrs.iteritems():
                    if attrs_name in exclude:
                        continue
                    if attrs_name not in self.time_ordered_attrs:
                        f.attrs[attrs_name] = self.attrs[attrs_name]

                for dset_name, dset in self.iteritems():
                    if dset_name in exclude:
                        continue
                    # write top level common datasets
                    if dset_name not in self.time_ordered_datasets.keys():
                        f.create_dataset(dset_name, data=dset, shape=dset.shape, dtype=dset.dtype)
                    # initialize time ordered datasets
                    else:
                        nt = dset.shape[0]
                        lt, et, st = mpiutil.split_m(nt, num_outfiles)
                        lshape = (lt[fi],) + dset.shape[1:]
                        f.create_dataset(dset_name, lshape, dtype=dset.dtype)
                        f[dset_name][:] = np.array(0.0).astype(dset.dtype)

                    # copy attrs of this dset
                    memh5.copyattrs(dset.attrs, f[dset_name].attrs)

        mpiutil.barrier(comm=self.comm)

        # then write time ordered datasets
        for dset_name, dset in self.iteritems():
            if dset_name in exclude:
                continue
            if dset_name in self.time_ordered_datasets.keys():
                dset_shape, dset_type, outfiles_map = self._get_output_info(dset_name, num_outfiles)

                st = 0
                # NOTE: if write simultaneously, will loss data with processes distributed in several nodes
                for ri in xrange(self.nproc):
                    if ri == self.rank:
                        for fi, start, stop in outfiles_map:

                            et = st + (stop - start)
                            with h5py.File(outfiles[fi], 'r+', libver=libver) as f:
                                f[dset_name][start:stop] = self[dset_name].local_data[st:et]
                            st = et
                    mpiutil.barrier(comm=self.comm)

                # mpiutil.barrier(comm=self.comm)

    def copy(self):
        """Return a deep copy of this container."""

        cont = self.__class__(dist_axis=self.main_data_dist_axis, comm=self.comm)

        # set hints
        hint_keys = [ key for key in self.__class__.__dict__.keys() if re.match(self.hints_pattern, key) ]
        for key in hint_keys:
            setattr(cont, key, getattr(self, key))

        # copy attrs
        for attrs_name, attrs_value in self.attrs.iteritems():
            cont.attrs[attrs_name] = deepcopy(attrs_value)

        # copy datasets
        for dset_name, dset in self.iteritems():
            cont.create_dataset(dset_name, data=dset.data.copy())
            memh5.copyattrs(dset.attrs, cont[dset_name].attrs)

        return cont



    def data_operate(self, func, op_axis=None, axis_vals=0, full_data=False, copy_data=False, show_progress=False, progress_step=None, keep_dist_axis=False, **kwargs):
        """A basic data operation interface.

        You can use this method to do some constrained operations to the main data
        hold in this container, i.e., the main data will not change its shape and
        dtype before and after the operation.

        Parameters
        ----------
        func : function object
            The opertation function object. It is of type func(array, self,
            \*\*kwargs) if `op_axis=None`, func(array, local_index, global_index,
            axis_val, self, \*\*kwargs) else.
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
                func(self.main_data.local_data.copy(), self, **kwargs)
            else:
                func(self.main_data.local_data, self, **kwargs)
        elif isinstance(op_axis, int) or isinstance(op_axis, basestring):
            axis = check_axis(op_axis, self.main_data_axes)
            data_sel = [ slice(0, None) ] * len(self.main_data_axes)
            if full_data:
                original_dist_axis = self.main_data_dist_axis
                self.redistribute(axis)
            if self.main_data.distributed:
                lgind = self.main_data.data.enumerate(axis)
            else:
                lgind = enumerate(range(self.main_data.data.shape[axis]))
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
                    func(self.main_data.local_data[tuple(data_sel)].copy(), lind, gind, axis_val, self, **kwargs)
                else:
                    func(self.main_data.local_data[tuple(data_sel)], lind, gind, axis_val, self, **kwargs)
            if full_data and keep_dist_axis:
                self.redistribute(original_dist_axis)
        elif isinstance(op_axis, tuple):
            axes = [ check_axis(axis, self.main_data_axes) for axis in op_axis ]
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
                    func(self.main_data.local_data[tuple(data_sel)].copy(), lind, gind, axis_val, self, **kwargs)
                else:
                    func(self.main_data.local_data[tuple(data_sel)], lind, gind, axis_val, self, **kwargs)
            if full_data and keep_dist_axis:
                self.redistribute(original_dist_axis)
        else:
            raise ValueError('Invalid op_axis: %s', op_axis)

    def all_data_operate(self, func, copy_data=False, **kwargs):
        """Operation to the whole main data.

        Note since the main data is distributed on different processes, `func`
        should not have operations that depend on elements not held in the local
        array of each process.

        Parameters
        ----------
        func : function object
            The opertation function object. It is of type func(array, self, \*\*kwargs),
            which will operate on the array and return an new array with the same
            shape and dtype.
        copy_data : bool, optional
            If True, `func` will operate on a copy of the data, so changes will
            have no impact on the data the container holds. Default False.
        \*\*kwargs : any other arguments
            Any other arguments that will passed to `func`.

        """
        self.data_operate(func, op_axis=None, axis_vals=0, full_data=False, copy_data=copy_data, keep_dist_axis=False, **kwargs)


    def _copy_a_common_attribute(self, name, other):
        # copy a common attribute from `other` to self
        self.attrs[name] = deepcopy(other.attrs[name])

    def _copy_a_time_ordered_attribute(self, name, other):
        # copy a time ordered attribute from `other` to self
        self.attrs[name] = deepcopy(other.attrs[name])

    def _copy_an_attribute(self, name, other):
        # copy an attribute (either a commmon or a time ordered) from `other` to self
        if name in self.time_ordered_attrs:
            self._copy_a_time_ordered_attribute(name, other)
        else:
            self._copy_a_common_attribute(name, other)

    def _copy_a_common_dataset(self, name, other):
        ### copy a common dataset from `other` to self
        self.create_dataset(name, data=other[name].data.copy())
        memh5.copyattrs(other[name].attrs, self[name].attrs)

    def _copy_a_main_axes_ordered_dataset(self, name, other):
        ### copy a main_axes_ordered_dataset from `other` to self
        axes = list(self.main_axes_ordered_datasets[name])
        axes = axes + [None] * (len(other[name].shape) - len(axes)) # complete the un-write axes
        sel = [ ( other.subset_data_select[a] if a is not None else slice(0, None, None) ) for a in axes ] # for data in file
        shp = [ len(np.arange(ni)[si]) for (ni, si) in zip(other[name].shape, sel) ]
        if not self.main_data_dist_axis in axes:
            # common data set
            self.create_dataset(name, data=other[name].data[sel].copy())
            memh5.copyattrs(other[name].attrs, self[name].attrs)
        else:
            # distributed data set
            di = axes.index(self.main_data_dist_axis) # index of dist axis
            # create a distributed dataset to hold the sub data set
            self.create_dataset(name, shape=shp, dtype=other[name].dtype, distributed=True, distributed_axis=di)
            # copy attrs of this dset
            memh5.copyattrs(other[name].attrs, self[name].attrs)

            # get start and end of the local data along axis di
            if other[name].distributed:
                s = other[name].data.local_offset[di]
                e = s + other[name].data.local_shape[di]
            else:
                s = 0
                e = other[name].shape[di]
            # get indices of the data corresponding to the sub data set
            sub_inds = np.arange(other[name].shape[di])[sel[di]]
            # split indices for each process
            ns, ss, es = mpiutil.split_all(len(sub_inds), comm=self.comm)
            # gather data to make each process to have its own data
            for ri, (si, ei) in enumerate(zip(ss, es)):
                linds = np.intersect1d(sub_inds[si:ei], np.arange(s, e))
                sel[di] = (linds - s).tolist()
                sel = [  ( _to_slice_obj(sl) if isinstance(sl, list) else sl ) for sl in sel ]
                ldata = mpiutil.gather_array(other[name].local_data[sel], axis=di, root=ri, comm=self.comm)
                if ri == mpiutil.rank:
                    self[name].local_data[:] = ldata

    def _copy_a_time_ordered_dataset(self, name, other):
        ### copy a time ordered dataset (except those also in main_axes_ordered_datasets) from `other` to self
        self.create_dataset(name, data=other[name].data.copy())
        memh5.copyattrs(other[name].attrs, self[name].attrs)

    def _copy_a_dataset(self, name, other):
        ### copy a dataset (either a commmon or a main axis ordered or a time ordered) from `other` to self
        if name in self.main_axes_ordered_datasets.keys():
            self._copy_a_main_axes_ordered_dataset(name, other)
        elif name in self.time_ordered_datasets.keys():
            self._copy_a_time_ordered_dataset(name, other)
        else:
            self._copy_a_common_dataset(name, other)


    def subset(self, return_copy=False):
        """Return a subset of the data as a new data container."""

        shp = self.main_data.shape
        new_shp = tuple([ len(np.arange(a)[s]) for (a, s) in zip(shp, self.subset_data_select) ])
        if shp == new_shp:
            if return_copy:
                return self.copy()
            else:
                return self

        # create a new container to hold the data subset
        cont = self.__class__(dist_axis=self.main_data_dist_axis, comm=self.comm)

        # set hints
        hint_keys = [ key for key in self.__class__.__dict__.keys() if re.match(self.hints_pattern, key) ]
        for key in hint_keys:
            setattr(cont, key, getattr(self, key))

        # copy attrs
        for name in self.attrs.iterkeys():
            cont._copy_an_attribute(name, self)

        # copy datasets
        for name in self.iterkeys():
            cont._copy_a_dataset(name, self)

        return cont
