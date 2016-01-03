"""Convert data to 4 dimentianal (time, bl, pol, freq)."""

try:
    import cPickle as pickle
except ImportError:
    import pickle

import os
import numpy as np
import h5py
import ephem

from tlpipe.kiyopy import parse_ini
from tlpipe.utils import mpiutil


# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
params_init = {
               'nprocs': mpiutil.size, # number of processes to run this module
               'aprocs': range(mpiutil.size), # list of active process rank no.
               # 'data_dir': './',  # directory the data in
               'data_files': ['./data.hdf5'],
               'output_dir': './output/', # output directory
              }
prefix = 'cv_'


def get_value(val):
    try:
        val = pickle.loads(val)
    except:
        pass

    return val


def get_ephdate(local_time, tzone='UTC+08'):
    """Convert `local_time` to ephem utc date."""
    local_time = ephem.Date(local_time)
    tz = int(tzone[3:])
    utc_time = local_time - tz * ephem.hour

    return utc_time


def get_juldate(local_time, tzone='UTC+08'):
    """Convert `local_time` to Julian date."""

    return ephem.julian_date(get_ephdate(local_time, tzone))


class Convert(object):
    """Convert data."""

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

        output_dir = self.params['output_dir']
        if mpiutil.rank0:
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

        data_files = self.params['data_files']
        nfiles = len(data_files)
        assert nfiles > 0, 'No input data files'
        # if self.comm is not None:
        #     assert self.comm.size <= nfiles, 'Can not have nprocs (%d) > nfiles (%d)' % (self.comm.size, nfiles)

        for data_file in mpiutil.mpilist(data_files):
            output_file = output_dir + data_file.split('/')[-1].replace('.hdf5', 'conv.hdf5')
            with h5py.File(data_file, 'r') as fin, h5py.File(output_file, 'w') as fout:
                vis_dataset = fin['vis']
                time_zone = get_value(vis_dataset.attrs['timezone'])
                start_time = get_value(vis_dataset.attrs['start_time'])
                int_time = get_value(vis_dataset.attrs['int_time'])
                ants = get_value(vis_dataset.attrs['ants'])
                xchans = get_value(vis_dataset.attrs['xchans'])
                ychans = get_value(vis_dataset.attrs['ychans'])
                bl_dict = get_value(vis_dataset.attrs['bl_dict'])

                # convert time to Julian date
                stime_ephdate = get_ephdate(start_time, tzone=time_zone)
                nt = vis_dataset.shape[0]
                time_ephdate = [stime_ephdate + ti*int_time*ephem.second for ti in range(nt)]
                time_juldate = np.array([ephem.julian_date(te) for te in time_ephdate])
                # select valid antennas (have both x and y)
                valid_ants = [ants[i] for i in range(len(ants)) if xchans[i] is not None and ychans[i] is not None]
                valid_xchans = [xchans[i] for i in range(len(ants)) if xchans[i] is not None and ychans[i] is not None]
                valid_ychans = [ychans[i] for i in range(len(ants)) if xchans[i] is not None and ychans[i] is not None]

                nant = len(valid_ants)
                xx_pair = [(valid_xchans[i], valid_xchans[j]) for i in range(nant) for j in range(i, nant)]
                yy_pair = [(valid_ychans[i], valid_ychans[j]) for i in range(nant) for j in range(i, nant)]
                xy_pair = [(valid_xchans[i], valid_ychans[j]) for i in range(nant) for j in range(i, nant)]
                yx_pair = [(valid_ychans[i], valid_xchans[j]) for i in range(nant) for j in range(i, nant)]

                xx_inds = [bl_dict['%d_%d' % (xi, xj)] for (xi, xj) in xx_pair]
                yy_inds = [bl_dict['%d_%d' % (yi, yj)] for (yi, yj) in yy_pair]
                xy_inds = [bl_dict['%d_%d' % (xi, yj)] for (xi, yj) in xy_pair]
                # yx needs special processing

                nbls = nant * (nant + 1) / 2
                npol = 4
                nfreq = vis_dataset.shape[1]

                output_vis = np.zeros((nt, nbls, npol, nfreq), dtype=vis_dataset.dtype)

                output_vis[:, :, 0, :] = vis_dataset[:, :, xx_inds].swapaxes(1, 2) # xx
                output_vis[:, :, 1, :] = vis_dataset[:, :, yy_inds].swapaxes(1, 2) # yy
                output_vis[:, :, 2, :] = vis_dataset[:, :, xy_inds].swapaxes(1, 2) # xy
                for bi, (yi, xj) in enumerate(yx_pair):
                    try:
                        ind = bl_dict['%d_%d' % (yi, xj)]
                        output_vis[:, bi, 3, :] = vis_dataset[:, :, ind]
                    except KeyError:
                        ind = bl_dict['%d_%d' % (xj, yi)]
                        output_vis[:, bi, 3, :] = vis_dataset[:, :, ind].conj()

                # save data converted
                data = fout.create_dataset('data', data=output_vis)
                # copy metadata from input file
                for attrs_name, attrs_value in vis_dataset.attrs.iteritems():
                    data.attrs[attrs_name] = attrs_value
                # update some attrs
                data.attrs['ants'] = valid_ants
                data.attrs['xchans'] = valid_xchans
                data.attrs['ychans'] = valid_ychans
                # data.attrs['time'] = time_juldate # could not save into attributes
                fout.create_dataset('time', data=time_juldate)
                data.attrs['axes'] = ['time', 'bls', 'pol', 'freq']
                data.attrs['pol'] = ['xx', 'yy', 'xy', 'yx']
                data.attrs['history'] = 'Data conversion from (time, freq, chpairs) to (time, bls, pol, freq).\n'
                del data.attrs['bl_dict']
