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


class BasicTod(memh5.MemDiskGroup):
    """Basic time ordered data container.

    Inherits from :class:`MemDiskGroup`.

    Basic one-level time ordered data container that allows any number of
    datasets in the root group but no nesting.

    This container is intended to be an base class for other concreate data
    classes, so only basic input/output and a limited operations are provided.

    Parameters
    ----------
    Parameters are passed through to the base class constructor.

    Attributes
    ----------
    index_map
    history

    Methods
    -------
    group_name_allowed
    dataset_name_allowed
    create_index_map
    del_index_map
    add_history
    redistribute

    """

    def __init__(self, files=None, comm=None):

        super(BasicTod, self).__init__(data_group=None, distributed=True, comm=comm)

        self.infiles = ensure_file_list(files)
        self.num_infiles = len(self.infiles)

        self.nproc = 1 if self.comm is None else self.comm.size
        self.rank = 0 if self.comm is None else self.comm.rank
        self.rank0 = True if self.rank == 0 else False

        self.main_data_shape, self.main_data_type, self.infiles_map = self._get_input_info(self.main_data)


    def _gen_files_map(self, nt, num_ts):
        ### generate files map, i.e., a list of (file_idx, start, stop)
        ### nt: total number of time points
        ### num_ts: a list of number of time points allocated to each file

        assert nt == np.sum(num_ts), 'Error: number of time points allocated to files are not correct'

        lt, st, et = mpiutil.split_local(nt, comm=self.comm) # total length distributed among different procs
        if self.comm is not None:
            lts = self.comm.allgather(lt)
        else:
            lts = [ lt ]
        cum_lts = np.cumsum(lts).tolist() # cumsum of lengths by all procs
        cum_num_ts = np.cumsum(num_ts).tolist() # cumsum of lengths of all files

        tmp_cum_lts = [0] + cum_lts
        tmp_cum_num_ts = [0] + cum_num_ts
        # start and stop (included) file indices owned by this proc
        sf, ef = np.searchsorted(cum_num_ts, tmp_cum_lts[self.rank], side='right'), np.searchsorted(cum_num_ts, tmp_cum_lts[self.rank+1], side='left')
        lf_indices = range(sf, ef+1) # file indices owned by this proc
        # allocation interval by all procs
        intervals = sorted(list(set([0] + cum_lts + cum_num_ts)))
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


    def _get_input_info(self, dset_name):
        ### get data shape and type and infile_map of time ordered datasets
        dset_type = None
        dset_shape = None
        tmp_shape = None
        num_ts = []
        lf, sf, ef = mpiutil.split_local(self.num_infiles, comm=self.comm)
        file_indices = range(sf, ef) # file indices owned by this proc
        for fi in file_indices:
            with h5py.File(self.infiles[fi], 'r') as f:
                num_ts.append(f[dset_name].shape[0])
                if fi == 0:
                    # get shape and type info from the first file
                    tmp_shape = f[dset_name].shape
                    dset_type= f[dset_name].dtype

        dset_type = mpiutil.bcast(dset_type, comm=self.comm)
        if self.comm is not None:
            num_ts = list(itertools.chain(*self.comm.allgather(num_ts)))
        nt = sum(num_ts) # total length of the first axis along different files
        if tmp_shape is not None:
            tmp_shape = (nt,) + tmp_shape[1:]
        dset_shape = mpiutil.bcast(tmp_shape, comm=self.comm)

        # lt, st, et = mpiutil.split_local(nt, comm=self.comm) # total length distributed among different procs
        # if self.comm is not None:
        #     lts = self.comm.allgather(lt)
        # else:
        #     lts = [ lt ]
        # cum_lts = np.cumsum(lts).tolist() # cumsum of lengths by all procs
        # cum_num_ts = np.cumsum(num_ts).tolist() # cumsum of lengths of all files

        # tmp_cum_lts = [0] + cum_lts
        # tmp_cum_num_ts = [0] + cum_num_ts
        # # start and stop (included) file indices owned by this proc
        # sf, ef = np.searchsorted(cum_num_ts, tmp_cum_lts[self.rank], side='right'), np.searchsorted(cum_num_ts, tmp_cum_lts[self.rank+1], side='left')
        # lf_indices = range(sf, ef+1) # file indices owned by this proc
        # # allocation interval by all procs
        # intervals = sorted(list(set([0] + cum_lts + cum_num_ts)))
        # intervals = [ (intervals[i], intervals[i+1]) for i in range(len(intervals)-1) ]
        # if self.comm is not None:
        #     num_lf_ind = self.comm.allgather(len(lf_indices))
        # else:
        #     num_lf_ind = [ len(lf_indices) ]
        # cum_num_lf_ind = np.cumsum([0] +num_lf_ind)
        # # local intervals owned by this proc
        # lits = intervals[cum_num_lf_ind[self.rank]: cum_num_lf_ind[self.rank+1]]
        # # infiles_map: a list of (file_idx, start, stop)
        # infiles_map = []
        # for idx, fi in enumerate(lf_indices):
        #     infiles_map.append((fi, lits[idx][0]-tmp_cum_num_ts[fi], lits[idx][1]-tmp_cum_num_ts[fi]))

        infiles_map = self._gen_files_map(nt, num_ts)

        return dset_shape, dset_type, infiles_map


    @property
    def main_data(self):
        """Main data in the data container."""
        return None

    @property
    def main_data_axes(self):
        """Axies of the main data."""
        return ()

    @property
    def time_ordered_datasets(self):
        return (self.main_data,)

    @property
    def time_ordered_attrs(self):
        return ()

    def _load_tod(self, dset_name, dset_shape, dset_type, infiles_map):
        ### load a time ordered dataset form all files
        md = mpiarray.MPIArray(dset_shape, axis=0, comm=self.comm, dtype=dset_type)
        st = 0
        attrs_dict = {}
        for fi, start, stop in infiles_map:
            et = st + (stop - start)
            with h5py.File(self.infiles[fi], 'r') as f:
                md[st:et] = f[dset_name][start:stop]
                memh5.copyattrs(f[dset_name].attrs, attrs_dict)
                st = et
        self.create_dataset(dset_name, shape=dset_shape, dtype=dset_type, data=md, distributed=True, distributed_axis=0)
        attrs_dict = mpiutil.bcast(attrs_dict, comm=self.comm)
        # copy attrs of this dset
        memh5.copyattrs(attrs_dict, self[dset_name].attrs)

    def _load_common(self):
        ### load common attributes and datasets from the first file
        ### this supposes that all common data are the same as that in the first file
        with h5py.File(self.infiles[0], 'r') as f:
            # read in top level common attrs
            for attrs_name, attrs_value in f.attrs.iteritems():
                if attrs_name not in self.time_ordered_attrs:
                    self.attrs[attrs_name] = attrs_value
            # read in top level common datasets
            for dset_name, dset in f.iteritems():
                if dset_name not in self.time_ordered_datasets:
                    self.create_dataset(dset_name, data=dset, shape=dset.shape, dtype=dset.dtype)
                    # copy attrs of this dset
                    memh5.copyattrs(dset.attrs, self[dset_name].attrs)

    def _load_main_data(self):
        ### load main data from all files
        self._load_tod(self.main_data, self.main_data_shape, self.main_data_type, self.infiles_map)

    def _load_time_ordered(self):
        ### load time ordered attributes and datasets from all files
        lf, sf, ef = mpiutil.split_local(self.num_infiles, comm=self.comm)
        for ta in self.time_ordered_attrs:
            self.attrs[ta] = []
        for fi in range(sf, ef):
            with h5py.File(self.infiles[fi], 'r') as f:
                # time ordered attrs
                for ta in self.time_ordered_attrs:
                    self.attrs[ta].append(f.attrs[ta])

        # gather time ordered attrs
        for ta in self.time_ordered_attrs:
            if self.comm is not None:
                self.attrs[ta] = list(itertools.chain(*self.comm.allgather(self.attrs[ta])))

        # load time ordered datasets
        for td in self.time_ordered_datasets:
            # # first load main data
            # self._load_main_data()
            if td != self.main_data:
                dset_shape, dset_type, infiles_map = self._get_input_info(td)
                self._load_tod(td, dset_shape, dset_type, infiles_map)

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

    def add_history(self, history=''):
        """Create a new history entry."""

        if history is not '':
            self.attrs['history'] += '\n' + history

    def redistribute(self, dist_axis):
        """Redistribute the main dataset along a specified axis.

        Parameters
        ----------
        dist_axis : int, string
            The axis can be specified by an integer index (positive or
            negative), or by a string label which must correspond to an entry in
            the `main_data_axes` attribute on the dataset.

        """

        naxis = len(self.main_data_axes)
        # Process if axis is a string
        if isinstance(dist_axis, basestring):
            try:
                axis = self.main_data_axes.index(dist_axis)
            except ValueError:
                raise ValueError('Can not redistribute data along an un-existed axis: %s' % dist_axis)
        # Process if axis is an integer
        elif isinstance(dist_axis, int):

            # Deal with negative axis index
            if dist_axis < 0:
                axis = naxis + dist_axis

        # Check axis is within bounds
        if axis < naxis:
            self.main_data.redistribute(axis)
        else:
            warnings.warn('Cannot not redistributed data to axis %d >= %d' % (axis, naxis))


    def _get_output_info(self, dset_name, num_outfiles):
        ### get data shape and type and infile_map of time ordered datasets

        dset_shape = self[dset_name].shape
        dset_type = self[dset_name].dtype

        nt = self[dset_name].shape[0]
        # allocate nt to the given number of files
        num_ts, num_s, num_e = mpiutil.split_m(nt, num_outfiles)

        # lt, st, et = mpiutil.split_local(nt, comm=self.comm) # total length distributed among different procs
        # if self.comm is not None:
        #     lts = self.comm.allgather(lt)
        # else:
        #     lts = [ lt ]
        # cum_lts = np.cumsum(lts).tolist() # cumsum of lengths by all procs
        # cum_num_ts = np.cumsum(num_ts).tolist() # cumsum of lengths of all files

        # tmp_cum_lts = [0] + cum_lts
        # tmp_cum_num_ts = [0] + cum_num_ts
        # # start and stop (included) file indices owned by this proc
        # sf, ef = np.searchsorted(cum_num_ts, tmp_cum_lts[self.rank], side='right'), np.searchsorted(cum_num_ts, tmp_cum_lts[self.rank+1], side='left')
        # lf_indices = range(sf, ef+1) # file indices owned by this proc
        # # allocation interval by all procs
        # intervals = sorted(list(set([0] + cum_lts + cum_num_ts)))
        # intervals = [ (intervals[i], intervals[i+1]) for i in range(len(intervals)-1) ]
        # if self.comm is not None:
        #     num_lf_ind = self.comm.allgather(len(lf_indices))
        # else:
        #     num_lf_ind = [ len(lf_indices) ]
        # cum_num_lf_ind = np.cumsum([0] +num_lf_ind)
        # # local intervals owned by this proc
        # lits = intervals[cum_num_lf_ind[self.rank]: cum_num_lf_ind[self.rank+1]]
        # # infiles_map: a list of (file_idx, start, stop)
        # outfiles_map = []
        # for idx, fi in enumerate(lf_indices):
        #     outfiles_map.append((fi, lits[idx][0]-tmp_cum_num_ts[fi], lits[idx][1]-tmp_cum_num_ts[fi]))

        outfiles_map = self._gen_files_map(nt, num_ts)

        return dset_shape, dset_type, outfiles_map


    def to_files(self, outfiles):
        """Save the data hold in this container to files."""
        outfiles = ensure_file_list(outfiles)
        num_outfiles = len(outfiles)
        if num_outfiles > self.num_infiles:
            warnings.warn('Number of output files %d exceed number of input files %d may have some problem' % (num_outfiles, self.num_infiles))

        # split output files among procs
        lf, sf, ef = mpiutil.split_local(num_outfiles, comm=self.comm)
        for fi in range(sf, ef):
            # first write top level common attrs and datasets to file
            with h5py.File(outfiles[fi], 'w') as f:
                # write top level common attrs
                for attrs_name, attrs_value in self.attrs.iteritems():
                    if attrs_name not in self.time_ordered_attrs:
                        f.attrs[attrs_name] = self.attrs[attrs_name]
                # write top level common datasets
                for dset_name, dset in self.iteritems():
                    if dset_name not in self.time_ordered_datasets:
                        f.create_dataset(dset_name, data=dset, shape=dset.shape, dtype=dset.dtype)
                        # copy attrs of this dset
                        memh5.copyattrs(dset.attrs, f[dset_name].attrs)
                # initialize time ordered datasets
                for td in self.time_ordered_datasets:
                    # if td == self.main_data:
                    #     continue
                    # get local data shape for this file
                    nt = self[td].global_shape[0]
                    lt, et,st = mpiutil.split_m(nt, num_outfiles)
                    lshape = (lt[fi],) + self[td].global_shape[1:]
                    f.create_dataset(td, lshape, dtype=self[td].dtype)
                    # f[td][:] = np.array(0.0).astype(self[td].dtype)
                    # copy attrs of this dset
                    memh5.copyattrs(self[td].attrs, f[td].attrs)

        # then write time ordered datasets
        for td in self.time_ordered_datasets:
            # if td == self.main_data:
            #     continue
            dset_shape, dset_type, outfiles_map = self._get_output_info(td, num_outfiles)
            st = 0
            for fi, start, stop in outfiles_map:
                et = st + (stop - start)
                with h5py.File(outfiles[fi], 'r+') as f:
                    f[td][start:stop] = self[td]._data.view(np.ndarray)[st:et]
                    st = et
