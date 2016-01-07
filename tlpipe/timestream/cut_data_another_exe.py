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
               'input_file': ['data1.hdf5', 'data2.hdf5'], # abs path if start with /, else relative to os.environ['TL_OUTPUT']
               'span': 60 * 60, # second, before and after transit time
               'output_file': ['cut_before.hdf5', 'cut_after.hdf5'],
              }
prefix = 'cuta_'


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

            # operate on first file
            with h5py.File(input_file[0], 'r') as f:
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
                tz = int(time_zone[3:])
                new_start_time = str(ephem.Date(end_time + tz * ephem.hour - span * ephem.second))
                print 'new_start_time:', new_start_time

                # cut data
                vis_before = dset[-span:] # last span second
                with h5py.File(output_file[0], 'w') as fb:
                    db = fb.create_dataset('vis', data=vis_before)
                    # copy metadata from input file
                    for attrs_name, attrs_value in dset.attrs.iteritems():
                        db.attrs[attrs_name] = attrs_value
                    # update some attrs
                    db.attrs['start_time'] = new_start_time

            # operate on second file
            with h5py.File(input_file[1], 'r') as f:
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
                tz = int(time_zone[3:])
                new_end_time = str(ephem.Date(start_time + tz * ephem.hour + span * ephem.second))
                print 'new_end_time:', new_end_time

                # cut data
                vis_after = dset[:span] # first span second
                with h5py.File(output_file[1], 'w') as fb:
                    db = fb.create_dataset('vis', data=vis_after)
                    # copy metadata from input file
                    for attrs_name, attrs_value in dset.attrs.iteritems():
                        db.attrs[attrs_name] = attrs_value
                    # update some attrs
                    db.attrs['end_time'] = new_end_time
