"""Module to subtract the foreground."""

#from caput import mpiutil
from tlpipe.pipeline.pipeline import SingleBase

import numpy as np
import scipy as sp
import h5py as h5
import healpy as hp
import find_modes


class FgSub(SingleBase):
    """Class to subtract the foreground."""
    params_init = {
            'input_files' :  '/data/ycli/tianlai/map_full.hdf5',
            'mode_list' : [0, 1, 2],
            }

    prefix = 'fg_'

    def read_input(self):

        print self.input_files
        data = h5.File(self.input_files[0])['map'][:,0,:]

        return data

    def process(self, input):

        # get the frequency covriance
        corr, weight = find_modes.freq_covariance(input, input, None, None, 
                range(input.shape[0]), range(input.shape[0]),no_weight=True)
        # get the svd modes
        mode_list = self.params['mode_list']
        mode_num = max(mode_list)
        svd = find_modes.get_freq_svd_modes(corr, mode_num)
        # subtract modes
        if mode_list[0] != 0: mode_list = [0,] + mode_list
        mode_sub_st = mode_list[:-1]
        mode_sub_ed = mode_list[1:]
        for i in range(len(mode_sub_st)):
            self.sub_modes(input, svd[1][mode_sub_st[i]:mode_sub_ed[i]])

        return input

    def sub_modes(self, input_map, modes):

        outmap = np.empty((len(modes), ) + input_map.shape[1:])

        for mode_index, mode_vector in enumerate(modes):
            mode_vector = mode_vector.reshape([input_map.shape[0],])

            amp = sp.tensordot(mode_vector, input_map, axes=(0,0))

            fitted = mode_vector[:, None] * amp[None, :]
            input_map -= fitted

            outmap[mode_index, ...] = amp


