"""Module to convert linear polarized visibility to Stokes visibility."""

try:
    import cPickle as pickle
except ImportError:
    import pickle

import os
import numpy as np
import h5py

from tlpipe.kiyopy import parse_ini
from tlpipe.utils import mpiutil


# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
params_init = {
               'nprocs': mpiutil.size, # number of processes to run this module
               'aprocs': range(mpiutil.size), # list of active process rank no.
               'data_file': 'data_cal.hdf5',
              }
prefix = 'st_'


pol_dict = {'I': 0, 'Q': 1, 'U': 2, 'V': 3}


class Lin2stokes(object):
    """Linear to Stokes conversion."""

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
        data_file = self.params['data_file']

        if mpiutil.rank0:
            output_file = output_dir + 'data_cal_stokes.hdf5'
            with h5py.File(data_file, 'r') as fin, h5py.File(output_file, 'w') as fout:
                in_dset = fin['data']

                # convert to Stokes I, Q, U, V
                data_stokes = np.zeros_like(in_dset)
                data_stokes[:, :, 0, :] = 0.5 * (in_dset[:, :, 0] + in_dset[:, :, 1]) # I
                data_stokes[:, :, 1, :] = 0.5 * (in_dset[:, :, 0] - in_dset[:, :, 1]) # Q
                data_stokes[:, :, 2, :] = 0.5 * (in_dset[:, :, 2] + in_dset[:, :, 3]) # U
                data_stokes[:, :, 3, :] = -0.5J * (in_dset[:, :, 2] - in_dset[:, :, 3]) # V

                # save stokes data
                out_dset = fout.create_dataset('data', data=data_stokes)
                # copy metadata from input file
                fout.create_dataset('time', data=fin['time'][...])
                for attrs_name, attrs_value in fin['data'].attrs.iteritems():
                    out_dset.attrs[attrs_name] = attrs_value
                # update some attrs
                out_dset.attrs['history'] = out_dset.attrs['history'] + 'Convert data to Stokes visibilities with parameters %s.\n' % self.params
