"""Module to do data conversion."""

try:
    import cPickle as pickle
except ImportError:
    import pickle

import os
import glob
import itertools as it
import numpy as np
import h5py

from tlpipe.kiyopy import parse_ini
from tlpipe.utils import mpiutil
# import params32ch
import data_set


# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
params_init = {
               # 'nprocs': mpiutil.size, # number of processes to run this module
               'nprocs': 1,
               'aprocs': range(mpiutil.size), # list of active process rank no.
               'start_time': 0,  # second; Assign starting time here
               'stop_time': 3600,  # second; Use -1 to load all of the raw data.
               'root_dir': '/home/data2', # Work directory; including 'data', 'graph', 'tmp' and so on.
               'data_dir': 'data', #
               'data_time:': '20151113001433',
               'graph_dir': 'graph', #
               'output_dir': 'data_hdf5_multibl',
               'pickle_file': None, # default
               'plot': False, # whether do the plot
              }
prefix = 'cv_'


def remove(x, lst):
    """Remove element `x` from list `lst`."""
    return filter(lambda a: a != x, lst)


def fname2ephemdttm(fname):
    """Get the ephem supported datetime string from file name."""
    return fname[:4]+'/'+fname[4:6]+'/'+fname[6:8]+' '+fname[8:10]+':'+fname[10:12]+':'+fname[12:14]


def ephemdttm2fname(ephemdttm):
    """Change file name to ephem supported datetime string."""
    dttm = ephem.date(ephemdttm)
    dttm = dttm.tuple()
    return format(dttm[0],'04d')+format(dttm[1],'02d')+format(dttm[2],'02d')+format(dttm[3],'02d')+format(dttm[4],'02d')+format(int(round(dttm[5])),'02d')


class Conversion(object):
    """Class to do data converion."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        # Read in the parameters.
        self.params = parse_ini.parse(parameter_file_or_dict, params_init,
                                 prefix=prefix, feedback=feedback)
        self.feedback = feedback
        nprocs = min(self.params['nprocs'], mpiutil.size)
        procs = set(range(mpiutil.size))
        aprocs = set(self.params['aprocs']) & procs
        self.aprocs = (list(aprocs) + list(set(range(nprocs)) - aprocs))[:nprocs]
        assert 0 in self.aprocs, 'Process 0 must be active'
        self.comm = mpiutil.active_comm(self.aprocs) # communicator consists of active processes

    def execute(self):

        # print 'rank %d executing...' % mpiutil.rank

        # if mpiutil.rank0:
        #     print 'Data conversion comes soon...'

        self.load_pickle()

        root_dir = self.params['root_dir']
        data_dir = root_dir + '/' + self.params['data_dir'] + '/' + self.params['data_time'] + '/'
        graph_dir = root_dir + '/' + self.params['graph_dir'] + '/'
        output_dir = root_dir + '/' + self.params['output_dir'] + '/'
        f12_lists, f43_lists, file_size = self.get_datafile_info(data_dir)
        nt_per_file = file_size / (2 * block_size)
        start_time = self.params['start_time']
        start_file, start_offset = divmod(start_time, nt_per_file)
        end_time = self.params['end_time']
        if end_time != -1:
            end_file, end_offset = divmod(end_time, nt_per_file)
        else:
            end_file, end_offset = None, None

        # load in data from files
        dataset = data_set.DataSet(f12_lists[start_file:end_file], f43_lists[start_file:end_file])
        chosen_data = dataset.data[start_offset:end_offset]
        ch_pairs = dataset.ch_pairs
        axes = dataset.axes
        ch_pairs = [(ch1, ch2) for (ch1, ch2) in ch_pairs]
        chosen_ch_pairs = list(it.combinations_with_replacement(self.chans, 2))
        inds = [ch_pairs.index(chp) for chp in chosen_data]
        vis = chosen_data[:, :, inds]

        # n_t = int((cnt_stop - cnt_start) / self.int_time)
        n_f = fft_len / 2
        # t_axis = np.arange(0, n_t * delta_t, delta_t) + t_start
        f_axis = np.arange(0, n_f * delta_f, delta_f) + 685

        fname_dttm_str = fname2ephemdttm(self.params['data_time'])
        chosen_dttm_str = str(ephem.date(ephem.date(fname_dttm_str) + t_start*ephem.second))
        chosen_fname_str = ephemdttm2fname(chosen_dttm_str)

        bl_dict = {}
        for ind, pair in enumerate(chosen_ch_pairs):
            bl_dict['%d_%d' % pair] = ind
        # save data to file
        with h5py.File(chosen_fname_str, 'w') as f:
            vis = f.create_dataset('vis', data=vis)
            vis.attrs['axes'] = axes
            vis.attrs['freq'] = f_axis
            vis.attrs['bl_dict'] = pickle.dumps(bl_dict)
            for key, val in self.pdict.items():
                vis.attrs[key] = val


    def load_pickle(self):
        """Load in parameters in the pickle file."""

        pdict = pickle.load(open(self.params['pickle_file'], "rb"))
        self.pdict = pdict
        # self.ants = pdict['ants']
        # self.xchans = pdict['xchans']
        # self.ychans = pdict['ychans']
        self.chans = sorted(remove(None, pdict['xchans'] + pdict['ychans']))
        # self.az, self.alt = np.radians(pdict['az_alt']) # radians
        self.timezone = pdict['timezone']
        self.start_obs_time = pdict['start_obs_time']
        self.end_obs_time = pdict['end_obs_time']
        self.int_time = pdict['int_time']
        # dataf12_prefix = 'f12_'
        # dataf12_suffix = '.dat'
        # dataf43_prefix = 'f43_'
        # dataf43_suffix = '.dat'

    def get_datafile_info(self, data_dir):
        """Get the data file information, including Names, Number of files, File sizes and so on."""

        # dataf12_prefix = 'f12_'; dataf12_suffix = '.dat'
        # dataf43_prefix = 'f43_'; dataf43_suffix = '.dat'
        dataf12_prefix = 'f12_'
        dataf12_suffix = '.dat'
        dataf43_prefix = 'f43_'
        dataf43_suffix = '.dat'
        # datafile_lists = os.listdir(datafile_dir)
        f12_glob = dataf12_prefix + '*' + dataf12_suffix
        f43_glob = dataf43_prefix + '*' + dataf43_suffix

        f12_lists = sort(glob.glob(data_dir + f12_glob))
        f12_sizes = [os.path.getsize(f) for f in f12_lists]
        f43_lists = sort(glob.glob(data_dir + f43_glob))
        f43_sizes = [os.path.getsize(f) for f in f43_lists]
        assert np.array_equal(f12_sizes, f43_sizes) and len(set(f12_sizes[:-1] + f43_sizes[:-1])) == 1, 'Data file sizes are incorrect'

        return f12_lists, f43_lists, f12_sizes[0]
