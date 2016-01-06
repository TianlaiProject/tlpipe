"""Cut a time section of data."""

import os
import numpy as np
import h5py
import ephem

from tlpipe.kiyopy import parse_ini
from tlpipe.utils import mpiutil
from tlpipe.utils.pickle_util import get_value
from tlpipe.utils.date_util import get_ephdate


# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
params_init = {
               'nprocs': mpiutil.size, # number of processes to run this module
               'aprocs': range(mpiutil.size), # list of active process rank no.
               'data_file': './data.hdf5',
               'span': 5 * 60, # second, before and after transit time
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

        output_dir = os.environ['TL_OUTPUT']
        span = self.params['span']

        if mpiutil.rank0:
            with h5py.File(self.params['data_file'], 'r') as f:

                dset = f['vis']
                start_time = get_value(dset.attrs['start_time'])
                print 'start_time:', start_time
                end_time = get_value(dset.attrs['end_time'])
                print 'end_time:', end_time
                # start_obs_time_lst = get_value(dset.attrs['start_obs_time'])
                # end_obs_time_lst = get_value(dset.attrs['end_obs_time'])
                transit_time_lst = get_value(dset.attrs['transit_time'])
                print 'transit_time:', transit_time_lst[0]
                int_time = get_value(dset.attrs['int_time'])
                time_zone = get_value(dset.attrs['timezone'])

                start_time = get_ephdate(start_time, time_zone) # utc
                end_time = get_ephdate(end_time, time_zone) # utc
                transit_time = get_ephdate(transit_time_lst[0], time_zone) # utc
                tz = int(time_zone[3:])
                new_start_time = str(ephem.Date(transit_time + tz * ephem.hour - span * ephem.second))
                print 'new_start_time:', new_start_time
                new_end_time = str(ephem.Date(transit_time + tz * ephem.hour + span * ephem.second))
                print 'new_end_time:', new_end_time
                # new_start_obs_time_lst = [str(ephem.Date(get_ephdate(st, time_zone) + tz * ephem.hour - span * ephem.second)) for st in start_obs_time_lst]
                # print new_start_obs_time_lst
                # new_end_obs_time_lst = [str(ephem.Date(get_ephdate(et, time_zone) + tz * ephem.hour + span * ephem.second)) for et in end_obs_time_lst]
                # print new_end_obs_time_lst

                # cut data
                eph_time = np.arange(start_time, end_time, int_time * ephem.second)
                transit_ind = np.searchsorted(eph_time, transit_time)
                print 'transit_ind:', transit_ind
                vis_before_transit = dset[transit_ind-int(span * int_time):transit_ind]
                vis_after_transit = dset[transit_ind:transit_ind+int(span * int_time)]
                with h5py.File(output_dir + 'test_before_transit.hdf5', 'w') as fb:
                    db = fb.create_dataset('vis', data=vis_before_transit)
                    # copy metadata from input file
                    for attrs_name, attrs_value in dset.attrs.iteritems():
                        db.attrs[attrs_name] = attrs_value
                    # update some attrs
                    db.attrs['start_time'] = new_start_time
                    db.attrs['end_time'] = transit_time_lst[0]

                with h5py.File(output_dir + 'test_after_transit.hdf5', 'w') as fb:
                    db = fb.create_dataset('vis', data=vis_after_transit)
                    # copy metadata from input file
                    for attrs_name, attrs_value in dset.attrs.iteritems():
                        db.attrs[attrs_name] = attrs_value
                    # update some attrs
                    db.attrs['start_time'] = transit_time_lst[0]
                    db.attrs['end_time'] = new_end_time
