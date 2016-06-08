import os
import time
import glob
import posixpath
import itertools
import warnings
import numpy as np
import h5py
from caput import mpiarray
from caput import memh5
from caput import mpiutil


def ensure_file_list(files):
    """Tries to interpret the input as a sequence of files

    Expands filename wildcards ("globs") and casts sequences to a list.

    """

    if memh5.is_group(files):
        files = [files]
    elif isinstance(files, basestring):
        files = sorted(glob.glob(files))
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
        The axis to be ckecked.
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

    Inherits from :class:`MemDiskGroup`.

    Basic one-level time ordered data container that allows any number of
    datasets in the root group but no nesting.

    This container is intended to be an base class for other concreate data
    classes, so only basic input/output and a limited operations are provided.

    Parameters
    ----------
    files : string or list of strings
        File name or a list of file names that data will be loaded from.
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
    comm : None or MPI.Comm, optional
        MPI Communicator to distributed over. Default None to use mpiutil._comm.

    Attributes
    ----------
    main_data
    main_data_name
    main_data_axes
    main_time_ordered_datasets
    time_ordered_datasets
    time_ordered_attrs
    history

    Methods
    -------
    data_select
    load_common
    load_main_data
    load_tod_excl_main_data
    load_time_ordered
    load_all
    group_name_allowed
    dataset_name_allowed
    attrs_name_allowed
    add_history
    info
    redistribute
    to_files

    """

    def __init__(self, files, mode='r', start=0, stop=None, dist_axis=0, comm=None):

        super(BasicTod, self).__init__(data_group=None, distributed=True, comm=comm)

        self.infiles_mode = mode
        # self.infiles will be a list of opened hdf5 file handlers
        self.infiles, self.main_data_start, self.main_data_stop = self._select_files(files, self.main_data_name, start, stop)
        self.num_infiles = len(self.infiles)
        self.main_data_dist_axis = check_axis(dist_axis, self.main_data_axes)

        self.nproc = 1 if self.comm is None else self.comm.size
        self.rank = 0 if self.comm is None else self.comm.rank
        self.rank0 = True if self.rank == 0 else False

        self.main_data_shape, self.main_data_type, self.infiles_map = self._get_input_info(self.main_data_name, self.main_data_start, self.main_data_stop)

        self._main_data_select = [ slice(0, None, None) for i in self._main_data_axes ]

    def __del__(self):
        """Closes the opened file handlers."""
        for fh in self.infiles:
            fh.close()

    def _select_files(self, files, dset_name, start=0, stop=None):
        ### select the needed files from `files` which contain time ordered data from `start` to `stop`
        assert dset_name in self.time_ordered_datasets, '%s is not a time ordered dataset' % dset_name

        files = ensure_file_list(files)
        num_ts = []
        for fh in mpiutil.mpilist(files, method='con', comm=self.comm):
            with h5py.File(fh, 'r') as f:
                num_ts.append(f[dset_name].shape[0])

        if self.comm is not None:
            num_ts = list(itertools.chain(*self.comm.allgather(num_ts)))
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

        lt, st, et = mpiutil.split_local(stop-start, comm=self.comm) # selected total length distributed among different procs
        if self.comm is not None:
            lts = self.comm.allgather(lt)
        else:
            lts = [ lt ]
        cum_lts = np.cumsum(lts) # cumsum of lengths by all procs
        cum_lts = (cum_lts + start).tolist()
        tmp_cum_lts = [start] + cum_lts
        cum_num_ts = np.cumsum(num_ts).tolist() # cumsum of lengths of all files
        tmp_cum_num_ts = [0] + cum_num_ts

        # start and stop (included) file indices owned by this proc
        sf, ef = np.searchsorted(cum_num_ts, tmp_cum_lts[self.rank], side='right'), np.searchsorted(cum_num_ts, tmp_cum_lts[self.rank+1], side='left')
        lf_indices = range(sf, ef+1) # file indices owned by this proc

        # allocation interval by all procs
        intervals = sorted(list(set([start] + cum_lts + cum_num_ts[:-1] + [stop])))
        intervals = [ (intervals[i], intervals[i+1]) for i in range(len(intervals)-1) ]
        if self.comm is not None:
            num_lf_ind = self.comm.allgather(len(lf_indices))
        else:
            num_lf_ind = [ len(lf_indices) ]
        cum_num_lf_ind = np.cumsum([0] +num_lf_ind)
        # local intervals owned by this proc
        lits = intervals[cum_num_lf_ind[self.rank]: cum_num_lf_ind[self.rank+1]]
        # infiles_map: a list of (file_idx, start, stop)
        files_map = []
        for idx, fi in enumerate(lf_indices):
            files_map.append((fi, lits[idx][0]-tmp_cum_num_ts[fi], lits[idx][1]-tmp_cum_num_ts[fi]))

        return files_map

    def _get_input_info(self, dset_name, start=0, stop=None):
        ### get data shape and type and infile_map of time ordered datasets
        ### start, stop are all relative to the first file
        ### shape and type are get from the first file and suppose they are the same in all self.infiles

        num_ts = []
        for fh in mpiutil.mpilist(self.infiles, method='con', comm=self.comm):
            num_ts.append(fh[dset_name].shape[0])

        if self.comm is not None:
            num_ts = list(itertools.chain(*self.comm.allgather(num_ts)))
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
            raise KeyError('Main data %s does not exist, try to load the main data first' % self.mapn_data_name)

    _main_data_name = None

    @property
    def main_data_name(self):
        """Main data in the data container."""
        return self._main_data_name

    @main_data_name.setter
    def main_data_name(self, value):
        if isinstance(value, basestring):
            self._main_data_name = value
        else:
            raise ValueError('Attribute main_data_name must be a string')

    _main_data_axes = ()

    @property
    def main_data_axes(self):
        """Axes of the main data."""
        return self._main_data_axes

    @main_data_axes.setter
    def main_data_axes(self, value):
        if isinstance(value, basestring):
            self._main_data_axes = (value,)
        elif hasattr(value, '__iter__'):
            for val in value:
                if not isinstance(val, basestring):
                    raise ValueError('Attribute main_data_axes must be a tuple of strings')
            self._main_data_axes = tuple(value)
        else:
            raise ValueError('Attribute main_data_axes must be a tuple of strings')

    _main_time_ordered_datasets = (_main_data_name,)

    @property
    def main_time_ordered_datasets(self):
        """Datasets that have same time points as the main data."""
        return self._main_time_ordered_datasets

    @main_time_ordered_datasets.setter
    def main_time_ordered_datasets(self, value):
        if isinstance(value, basestring):
            self._main_time_ordered_datasets = (value,)
        elif hasattr(value, '__iter__'):
            for val in value:
                if not isinstance(val, basestring):
                    raise ValueError('Attribute main_time_ordered_datasets must be a tuple of strings')
            self._main_time_ordered_datasets = tuple(value)
        else:
            raise ValueError('Attribute main_time_ordered_datasets must be a tuple of strings')

    _time_ordered_datasets = (_main_data_name,)

    @property
    def time_ordered_datasets(self):
        """Time ordered datasets."""
        return self._time_ordered_datasets

    @time_ordered_datasets.setter
    def time_ordered_datasets(self, value):
        if isinstance(value, basestring):
            self._time_ordered_datasets = (value,)
        elif hasattr(value, '__iter__'):
            for val in value:
                if not isinstance(val, basestring):
                    raise ValueError('Attribute time_ordered_datasets must be a tuple of strings')
            self._time_ordered_datasets = tuple(value)
        else:
            raise ValueError('Attribute time_ordered_datasets must be a tuple of strings')

    _time_ordered_attrs = ()

    @property
    def time_ordered_attrs(self):
        """Attributes that are different in different files."""
        return self._time_ordered_attrs

    @time_ordered_attrs.setter
    def time_ordered_attrs(self, value):
        if isinstance(value, basestring):
            self._time_ordered_attrs = (value,)
        elif hasattr(value, '__iter__'):
            for val in value:
                if not isinstance(val, basestring):
                    raise ValueError('Attribute time_ordered_attrs must be a tuple of strings')
            self._time_ordered_attrs = tuple(value)
        else:
            raise ValueError('Attribute time_ordered_attrs must be a tuple of strings')


    _main_data_select = None

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
            raise NotImplementedError('Select data to be loaded along the first axis is not implemented yet')
        if isinstance(value, tuple):
            self._main_data_select[axis] = slice(*value)
        elif isinstance(value, list):
            if sorted(value) != value:
                raise TypeError("Indexing elements must be in increasing order")
            if value[0] < 0:
                raise TypeError("Indexing elements must be non-negative integers")
            self._main_data_select[axis] = value
        else:
            raise ValueError('Unsupported data selection %s' % value)


    def _load_a_common_attribute(self, name):
        ### load a common attribute from the first file
        fh = self.infiles[0]
        self.attrs[name] = fh.attrs[name]

    def _load_a_tod_attribute(self, name):
        ### load a time ordered attribute from all the file
        self.attrs[name] = []
        for fh in mpiutil.mpilist(self.infiles, method='con', comm=self.comm):
            self.attrs[name].append(fh.attrs[name])

        # gather time ordered attrs
        if self.comm is not None:
            self.attrs[name] = list(itertools.chain(*self.comm.allgather(self.attrs[name])))

    def _load_an_attribute(self, name):
        ### load an attribute (either a commmon or a time ordered)
        if name in self.time_ordered_attrs:
            self._load_a_tod_attribute(name)
        else:
            self._load_a_common_attribute(name)

    def _load_a_common_dataset(self, name):
        ### load a common dataset from the first file
        fh = self.infiles[0]
        dset = fh[name]
        self.create_dataset(name, data=dset, shape=dset.shape, dtype=dset.dtype)
        # copy attrs of this dset
        memh5.copyattrs(dset.attrs, self[name].attrs)

    def _load_a_tod_dataset(self, name):
        ### load a time ordered dataset from all files, distributed along the first axis
        if name in self.main_time_ordered_datasets:
            dset_shape, dset_type, infiles_map = self._get_input_info(name, self.main_data_start, self.main_data_stop)
            first_start = mpiutil.bcast(infiles_map[0][1], root=0, comm=self.comm) # start form the first file
            last_stop = mpiutil.bcast(infiles_map[-1][2], root=self.nproc-1, comm=self.comm) # stop from the last file
        else:
            dset_shape, dset_type, infiles_map = self._get_input_info(name, 0, None)

        # for main data
        if name == self.main_data_name:
            main_data_select = self._main_data_select
            new_dset_shape = (dset_shape[0],)
            for axis in range(1, len(dset_shape)): # exclude the first axis
                tmp = np.arange(dset_shape[axis])
                sel = tmp[main_data_select[axis]]
                new_dset_shape += (len(sel),)
                if axis == self.main_data_dist_axis:
                    main_data_select[axis] = mpiutil.mpilist(sel, method='con', comm=self.comm).tolist() # must have tolist as a single number numpy array index will reduce one axis in h5py slice

            self.create_dataset(name, shape=new_dset_shape, dtype=dset_type, distributed=True, distributed_axis=self.main_data_dist_axis)
            # copy attrs of this dset
            memh5.copyattrs(self.infiles[0][name].attrs, self[name].attrs)

            if self.main_data_dist_axis == 0:
                st = 0
                for fi, start, stop in infiles_map:
                    main_data_select[0] = slice(start, stop)
                    et = st + (stop - start)
                    fh = self.infiles[fi]
                    if np.prod(self[name].local_data[st:et].shape) > 0:
                        # only read in data if non-empty, may get error otherwise
                        self[name].local_data[st:et] = fh[name][tuple(main_data_select)]
                    st = et
            # need to take special care when dist_axis != 0
            else:
                st = 0
                # every proc has to read from all files
                for fi, fh in enumerate(self.infiles):
                    num_ts = fh[name].shape[0]
                    if fi == 0:
                        et = st + (num_ts - first_start)
                        main_data_select[0] = slice(first_start, None)
                    elif fi == self.num_infiles-1:
                        et = st + last_stop
                        main_data_select[0] = slice(0, last_stop)
                    else:
                        et = st + num_ts

                    if np.prod(self[name].local_data[st:et].shape) > 0:
                        # only read in data if non-empty, may get error otherwise
                        self[name].local_data[st:et] = fh[name][tuple(main_data_select)] # h5py need the explicit tuple conversion
                    st = et

        # for other main_time_ordered_datasets
        elif name in self.main_time_ordered_datasets:
            if self.main_data_dist_axis == 0:
                # distribute it along the first axis as the main data
                self.create_dataset(name, shape=dset_shape, dtype=dset_type, distributed=True, distributed_axis=0)
                # copy attrs of this dset
                memh5.copyattrs(self.infiles[0][name].attrs, self[name].attrs)
                st = 0
                for fi, start, stop in infiles_map:
                    et = st + (stop - start)
                    fh = self.infiles[fi]
                    self[name].local_data[st:et] = fh[name][start:stop]
                    st = et
            else:
                # as the distributed axis of the main data is not the time axis,
                # these main_time_ordered_datasets should also not distributed along time axis,
                # so here load it as common datasets
                self.create_dataset(name, shape=dset_shape, dtype=dset_type)
                # copy attrs of this dset
                memh5.copyattrs(self.infiles[0][name].attrs, self[name].attrs)
                st = 0
                for fi, fh in enumerate(self.infiles):
                    num_ts = fh[name].shape[0]
                    if fi == 0:
                        et = st + (num_ts - first_start)
                        sel = slice(fist_start, None)
                    elif fi == self.num_infiles-1:
                        et = st + last_stop
                        sel = slice(0, last_stop)
                    else:
                        et = st + num_ts
                        sel = slice(0, None)
                    self[name][st:et] = fh[name][sel] # not a distributed dataset
                    st = et

        # for non main_time_ordered_datasets
        else:
            # for other time ordered data that are not main_time_ordered_datasets,
            # always distribute them along the first axis no matter what self.main_data_dist_axis is
            self.create_dataset(name, shape=dset_shape, dtype=dset_type, distributed=True, distributed_axis=0)
            # copy attrs of this dset
            memh5.copyattrs(self.infiles[0][name].attrs, self[name].attrs)
            st = 0
            for fi, start, stop in infiles_map:
                et = st + (stop - start)
                fh = self.infiles[fi]
                self[name].local_data[st:et] = fh[name][start:stop]
                st = et

    def _load_a_dataset(self, name):
        ### load a dataset (either a commmon or a time ordered)
        if name in self.time_ordered_datasets:
            self._load_a_tod_dataset(name)
        else:
            self._load_a_common_dataset(name)


    def load_common(self):
        """Load common attributes and datasets from the first file.

        This supposes that all common data are the same as that in the first file.
        """
        fh = self.infiles[0]
        # read in top level common attrs
        for attr_name in fh.attrs.iterkeys():
            if attr_name not in self.time_ordered_attrs:
                self._load_a_common_attribute(attr_name)
        # read in top level common datasets
        for dset_name in fh.iterkeys():
            if dset_name not in self.time_ordered_datasets:
                self._load_a_common_dataset(dset_name)

    def load_main_data(self):
        """Load main data from all files."""
        self._load_a_tod_dataset(self.main_data_name)

    def load_tod_excl_main_data(self):
        """Load time ordered attributes and datasets (exclude the main data) from all files."""
        # load time ordered attributes
        for ta in self.time_ordered_attrs:
            self._load_a_tod_attribute(ta)

        # load time ordered datasets
        for td in self.time_ordered_datasets:
            if td != self.main_data_name:
                self._load_a_tod_dataset(td)

    def load_time_ordered(self):
        """Load time ordered attributes and datasets from all files."""
        self.load_main_data()
        self.load_tod_excl_main_data()

    def load_all(self):
        """Load all attribures and datasets form files."""
        self.load_common()
        self.load_time_ordered()

    def group_name_allowed(self, name):
        """No groups are exposed to the user. Returns ``False``."""
        return False

    def dataset_name_allowed(self, name):
        """Datasets may only be created and accessed in the root level group.

        Returns ``True`` is *name* is a path in the root group i.e. '/dataset'.

        """

        parent_name, name = posixpath.split(name)
        return True if parent_name == '/' else False

    def attrs_name_allowed(self, name):
        """Whether to allow the access of the given root level attribute."""
        return True if name not in self.time_ordered_attrs else False

    @property
    def history(self):
        """The analysis history for this data.

        Do not try to add a new entry by assigning to an element of this
        property. Use :meth:`~BasicCont.add_history` instead.

        Returns
        -------
        history : string

        """

        return self.attrs['history']

    def add_history(self, history=''):
        """Create a new history entry."""

        if history is not '':
            self.attrs['history'] += '\n' + history

    def info(self):
        """List basic information of the data hold by this container."""
        if self.rank0:
            # list the opened files
            print 'Input files:'
            for fh in self.infiles:
                print '  ', fh.filename
            print
            # list all top level attributes
            for attr_name, attr_val in self.attrs.iteritems():
                print '%s:' % attr_name, attr_val
            # list all top level datasets
            for dset_name, dset in self.iteritems():
                print dset_name, '  shape = ', dset.shape
                # list its attributes
                for attr_name, attr_val in dset.attrs.iteritems():
                    print '  %s:' % attr_name, attr_val

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

        axis = check_axis(dist_axis, self.main_data_axes)

        if axis == self.main_data_dist_axis:
            # already the distributed axis, nothing to do
            return
        else:
            # redistribute main data
            self.main_data.redistribute(axis)
            self.main_data_dist_axis = axis
            # redistribute other main_time_ordered_datasets
            for dset_name in self.main_time_ordered_datasets:
                if dset_name != self.main_data_name:
                    dset_type = self[dset_name].dtype
                    dset_shape = self[dset_name].shape
                    if axis == 0:
                        # create axis 0 distributed dataset from non-distributed dataset
                        nt = dset_shape[0]
                        lt, st, et = mpiutil.split_local(nt, comm=self.comm)
                        md = mpiarray.MPIArray(dset_shape, axis=0, comm=self.comm, dtype=dset_type)
                        md.local_array[:] = self[dset_name][st:et].copy()
                        attr_dict = {} # temporarily save attrs of this dataset
                        memh5.copyattrs(self[dset_name].attrs, attr_dict)
                        del self[dset_name]
                        self.create_dataset(dest_name, shape=dset_shape, dtype=dset_type, data=md, distributed=True, distributed_axis=0)
                        memh5.copyattrs(attr_dict, self[dset_name].attrs)
                    else:
                        # gather local distributed dataset to a global array for all procs
                        global_array = np.zeros(dset_shape, dtype=dset_type)
                        local_start = self[dset_name].local_offset
                        for rank in range(self.nproc):
                            mpiutil.gather_local(global_array, self[dset_name].local_data, local_start, root=rank, comm=self.comm)
                        attr_dict = {} # temporarily save attrs of this dataset
                        memh5.copyattrs(self[dset_name].attrs, attr_dict)
                        del self[dset_name]
                        self.create_dataset(dest_name, data=global_array, shape=dset_shape, dtype=dset_type)
                        memh5.copyattrs(attr_dict, self[dset_name].attrs)


    def _get_output_info(self, dset_name, num_outfiles):
        ### get data shape and type and infile_map of time ordered datasets

        dset_shape = self[dset_name].shape
        dset_type = self[dset_name].dtype

        nt = self[dset_name].shape[0]
        # allocate nt to the given number of files
        num_ts, num_s, num_e = mpiutil.split_m(nt, num_outfiles)

        outfiles_map = self._gen_files_map(num_ts, start=0, stop=None)

        return dset_shape, dset_type, outfiles_map


    def to_files(self, outfiles):
        """Save the data hold in this container to files."""
        outfiles = ensure_file_list(outfiles)
        num_outfiles = len(outfiles)
        if num_outfiles > self.num_infiles:
            warnings.warn('Number of output files %d exceed number of input files %d may have some problem' % (num_outfiles, self.num_infiles))

        # split output files among procs
        for fi, outfile in enumerate(mpiutil.mpilist(outfiles, method='con', comm=self.comm)):
            # first write top level common attrs and datasets to file
            with h5py.File(outfile, 'w') as f:

                # write top level common attrs
                for attrs_name, attrs_value in self.attrs.iteritems():
                    if attrs_name not in self.time_ordered_attrs:
                        f.attrs[attrs_name] = self.attrs[attrs_name]

                for dset_name, dset in self.iteritems():
                    # write top level common datasets
                    if dset_name not in self.time_ordered_datasets:
                        f.create_dataset(dset_name, data=dset, shape=dset.shape, dtype=dset.dtype)
                    # initialize time ordered datasets
                    else:
                        nt = dset.global_shape[0]
                        lt, et, st = mpiutil.split_m(nt, num_outfiles)
                        lshape = (lt[fi],) + dset.global_shape[1:]
                        f.create_dataset(dset_name, lshape, dtype=dset.dtype)
                        # f[dset_name][:] = np.array(0.0).astype(dset.dtype)

                    # copy attrs of this dset
                    memh5.copyattrs(dset.attrs, f[dset_name].attrs)

        mpiutil.barrier(comm=self.comm)

        # open all output files for more efficient latter operations
        outfiles = [ h5py.File(fl, 'r+') for fl in outfiles ]

        # then write time ordered datasets
        for dset_name, dset in self.iteritems():
            if dset_name in self.time_ordered_datasets:

                # first redistribute main_time_ordered_datasets to the first axis
                if self.main_data_dist_axis != 0:
                    self.redistribute(0)

                dset_shape, dset_type, outfiles_map = self._get_output_info(dset_name, num_outfiles)
                st = 0
                for fi, start, stop in outfiles_map:

                    et = st + (stop - start)
                    outfiles[fi][dset_name][start:stop] = self[dset_name].local_data[st:et]
                    st = et

                mpiutil.barrier(comm=self.comm)

        # close all output files
        for fh in outfiles:
            fh.close()
