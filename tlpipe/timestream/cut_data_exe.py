"""Cut a time section of data."""

import os
import numpy as np
import h5py
import ephem

from tlpipe.kiyopy import parse_ini
from tlpipe.utils import mpiutil
from tlpipe.utils.pickle_util import get_value
from tlpipe.utils.date_util import get_ephdate
from tlpipe.utils.path_util import input_path, output_path


# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
params_init = {
               'nprocs': mpiutil.size, # number of processes to run this module
               'aprocs': range(mpiutil.size), # list of active process rank no.
               'input_file': 'data.hdf5', # abs path if start with /, else relative to os.environ['TL_OUTPUT']
               'span': 5 * 60, # second, before and after transit time
               'output_file': ['cut_before_transit.hdf5', 'cut_after_transit.hdf5'],
              }
prefix = 'cut_'


class Cut(object):
    """Cut a time section of data."""

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

        input_file = input_path(self.params['input_file'])
        output_file = output_path(self.params['output_file'])
        span = self.params['span']

        if mpiutil.rank0:
            with h5py.File(input_file, 'r') as f:

                dset = f['vis']
                start_time = get_value(dset.attrs['start_time'])
                print 'start_time:', start_time
                end_time = get_value(dset.attrs['end_time'])
                print 'end_time:', end_time
                transit_time_lst = get_value(dset.attrs['transit_time'])
                print 'transit_time:', transit_time_lst[0]
                int_time = get_value(dset.attrs['int_time'])
                time_zone = get_value(dset.attrs['timezone'])

                start_time = get_ephdate(start_time, time_zone) # utc
                end_time = get_ephdate(end_time, time_zone) # utc
                transit_time = get_ephdate(transit_time_lst[0], time_zone) # utc
                new_start_utc_time = transit_time - span * ephem.second
                new_end_utc_time = transit_time + span * ephem.second
                tz = int(time_zone[3:])
                new_start_time = str(ephem.Date(new_start_utc_time + tz * ephem.hour))
                print 'new_start_time:', new_start_time
                new_end_time = str(ephem.Date(new_end_utc_time + tz * ephem.hour))
                print 'new_end_time:', new_end_time

                # cut data
                eph_time = np.arange(start_time, end_time, int_time * ephem.second)
                transit_ind = np.searchsorted(eph_time, transit_time)
                print 'transit_ind:', transit_ind
                vis_before_transit = dset[transit_ind-int(span * int_time):transit_ind]
                vis_after_transit = dset[transit_ind:transit_ind+int(span * int_time)]
                with h5py.File(output_file[0], 'w') as fb:
                    db = fb.create_dataset('vis', data=vis_before_transit)
                    # copy metadata from input file
                    for attrs_name, attrs_value in dset.attrs.iteritems():
                        db.attrs[attrs_name] = attrs_value
                    # update some attrs
                    db.attrs['start_time'] = new_start_time
                    db.attrs['end_time'] = transit_time_lst[0]

                with h5py.File(output_file[1], 'w') as fb:
                    db = fb.create_dataset('vis', data=vis_after_transit)
                    # copy metadata from input file
                    for attrs_name, attrs_value in dset.attrs.iteritems():
                        db.attrs[attrs_name] = attrs_value
                    # update some attrs
                    db.attrs['start_time'] = transit_time_lst[0]
                    db.attrs['end_time'] = new_end_time
