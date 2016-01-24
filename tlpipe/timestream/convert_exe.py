"""Convert data to 4 dimentianal (time, bl, pol, freq)."""

try:
    import cPickle as pickle
except ImportError:
    import pickle

import os
import numpy as np
import h5py
import ephem

from tlpipe.utils import mpiutil
from tlpipe.core.base_exe import Base
from tlpipe.utils.pickle_util import get_value
from tlpipe.utils.date_util import get_ephdate
from tlpipe.utils.path_util import input_path, output_path


# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
params_init = {
               'nprocs': mpiutil.size, # number of processes to run this module
               'aprocs': range(mpiutil.size), # list of active process rank no.
               'input_file': ['cut_before_transit.hdf5', 'cut_after_transit.hdf5'],
               'output_file': ['cut_before_transit_conv.hdf5', 'cut_after_transit_conv.hdf5'],
               'exclude_ant': [], # a list of ants to exclude
               'exclude_freq_index': range(62) + range(473, 512), # make freq between 700-800 MHz
               'extra_history': '',
              }
prefix = 'cv_'



class Convert(Base):
    """Convert data."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(Convert, self).__init__(parameter_file_or_dict, params_init, prefix, feedback)

    def execute(self):

        input_file = input_path(self.params['input_file'])
        output_file = output_path(self.params['output_file'])
        exclude_freq_index = self.params['exclude_freq_index']

        nfiles = len(input_file)
        assert nfiles > 0, 'No input data file'

        for infile, outfile in zip(mpiutil.mpilist(input_file), mpiutil.mpilist(output_file)):
            with h5py.File(infile, 'r') as fin, h5py.File(outfile, 'w') as fout:
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

                # remove excluded ants
                for ant in self.params['exclude_ant']:
                    ant_ind = valid_ants.index(ant)
                    valid_ants.remove(valid_ants[ant_ind])
                    valid_xchans.remove(valid_xchans[ant_ind])
                    valid_ychans.remove(valid_ychans[ant_ind])

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
                freq_indices = list(set(range(nfreq)) - set(exclude_freq_index))
                freq = vis_dataset.attrs['freq'][freq_indices] # the included freq
                nfreq = len(freq)

                output_vis = np.zeros((nt, nbls, npol, nfreq), dtype=vis_dataset.dtype)

                # output_vis[:, :, 0, :] = vis_dataset[:, freq_indices, xx_inds].swapaxes(1, 2) # xx
                ### Only one indexing vector or array is currently allowed for advanced selection in h5py, so
                output_vis[:, :, 0, :] = vis_dataset[:, :, xx_inds][:, freq_indices, :].swapaxes(1, 2) # xx
                output_vis[:, :, 1, :] = vis_dataset[:, :, yy_inds][:, freq_indices, :].swapaxes(1, 2) # yy
                output_vis[:, :, 2, :] = vis_dataset[:, :, xy_inds][:, freq_indices, :].swapaxes(1, 2) # xy
                for bi, (yi, xj) in enumerate(yx_pair):
                    try:
                        ind = bl_dict['%d_%d' % (yi, xj)]
                        output_vis[:, bi, 3, :] = vis_dataset[:, :, ind][:, freq_indices]
                    except KeyError:
                        ind = bl_dict['%d_%d' % (xj, yi)]
                        output_vis[:, bi, 3, :] = vis_dataset[:, :, ind][:, freq_indices].conj()

                # save data converted
                data = fout.create_dataset('data', data=output_vis)
                # copy metadata from input file
                for attrs_name, attrs_value in vis_dataset.attrs.iteritems():
                    data.attrs[attrs_name] = attrs_value
                # update some attrs
                data.attrs['ants'] = valid_ants
                data.attrs['xchans'] = valid_xchans
                data.attrs['ychans'] = valid_ychans
                data.attrs['freq'] = freq
                # data.attrs['time'] = time_juldate # could not save into attributes
                fout.create_dataset('time', data=time_juldate)
                data.attrs['axes'] = ['time', 'bls', 'pol', 'freq']
                data.attrs['pol'] = ['xx', 'yy', 'xy', 'yx']
                data.attrs['history'] = self.history
                del data.attrs['bl_dict']
