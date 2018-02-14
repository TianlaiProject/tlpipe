"""Module to subtract the foreground."""

#from caput import mpiutil
from tlpipe.pipeline.pipeline import SingleBase

import matplotlib.pyplot as plt

import numpy as np
import scipy as sp
import h5py as h5
import healpy as hp
import find_modes
import copy

def read_wufq(data_file):

    data_dic  = np.load(data_file)
    data = data_dic['sky']
    freq = data_dic['freqs']
    ra   = data_dic['ra']
    dec  = data_dic['dec']

    data = np.swapaxes(data, 0, 2)
    #print data.shape
    #data = data.T
    #print data.shape

    return data, ra, dec, freq

def check_map(result_file, healpix=True):

    result = h5.File(result_file, 'r')
    ra  = result['ra'][:]
    dec = result['dec'][:]
    freq= result['freq'][:]
    map_cleaned = result['map_cleaned'][:]
    svd_amp     = result['svd_amp'][:]
    svd_eigval  = result['svd_eigval'][:]
    svd_eigvec_left  = result['svd_eigvec_left'][:]
    #print map_cleaned.shape
    #print svd_amp.shape
    print svd_eigval.shape
    result.close()

    plt.plot(range(svd_eigval.shape[0]), svd_eigval, 'r.-')
    plt.semilogy()
    plt.show()

    for i in range(svd_eigvec_left.shape[0]):
        plt.plot(svd_eigvec_left[i,...]-i)
    plt.show()


    for i in range(map_cleaned.shape[0]):
        if healpix:
            hp.mollview(np.mean(map_cleaned[i], axis=0), coord=('E',), norm='hist')
        else:
            plt.pcolormesh(dec[None, :], ra[:, None], map_cleaned[i, 0, ...])
            plt.colorbar()
            #plt.pcolormesh(dec[None, :], ra[:, None], svd_amp[i, 0, ...])
        plt.show()



class FgSub(SingleBase):
    """Class to subtract the foreground."""
    params_init = {
            'wufq' : False,
            'input_files' :  '/data/ycli/tianlai/map_full.hdf5',
            'output_files' : 'result.hdf5',
            'mode_list' : [0, 1, 2],
            }

    prefix = 'fg_'

    def read_input(self):

        if self.params['wufq']:
            data, self.ra, self.dec, self.freq = read_wufq(self.input_files[0])
        else: 
            data = h5.File(self.input_files[0])['map'][:,0,:]
            # to be add later
            self.ra = 0
            self.dec = 0
            self.freq = 0

        return data

    def write_output(self, input):

        #print input
        #print self.output_files

        result = h5.File(self.output_files[0], 'w')
        result['map_cleaned'] = input[0]
        result['svd_amp'] = input[1]
        result['svd_eigval'] = self.svd[0]
        result['svd_eigvec_left'] = self.svd[1]
        result['svd_eigvec_right'] = self.svd[2]
        result['mode_list'] = self.params['mode_list']
        result['ra']   = self.ra
        result['dec']  = self.dec
        result['freq'] = self.freq
        result.close()

        check_map(self.output_files[0], healpix = not self.params['wufq'])

    def process(self, input):

        # get the frequency covriance
        corr, weight = find_modes.freq_covariance(input, input, None, None, 
                range(input.shape[0]), range(input.shape[0]),no_weight=True)
        # get the svd modes
        mode_list = self.params['mode_list']
        mode_num = max(mode_list)
        svd = find_modes.get_freq_svd_modes(corr, mode_num)
        self.svd = svd
        # subtract modes
        #if mode_list[0] != 0: mode_list = [0,] + mode_list
        mode_sub_st = [0, ] + mode_list[:-1]
        mode_sub_ed = mode_list
        mode_amp = []
        map_cleaned = []
        for i in range(len(mode_sub_st)):
            amp, cln = self.sub_modes(input, svd[1][mode_sub_st[i]:mode_sub_ed[i]])
            mode_amp.append(amp)
            map_cleaned.append(cln)
            input = copy.copy(cln)

        #mode_amp = np.array(mode_amp)
        mode_amp = np.concatenate(mode_amp, axis=0)
        map_cleaned = np.array(map_cleaned)

        return map_cleaned, mode_amp

    def sub_modes(self, input_map, modes):

        mapshp = input_map.shape
        input_map = input_map.reshape([mapshp[0], np.product(mapshp[1:])])

        if len(modes) == 0:
            mode_n = 1
        else:
            mode_n = len(modes)
        outmap = np.empty((mode_n, np.product(mapshp[1:])))

        for mode_index, mode_vector in enumerate(modes):
            mode_vector = mode_vector.reshape([mapshp[0],])

            amp = sp.tensordot(mode_vector, input_map, axes=(0,0))

            fitted = mode_vector[:, None] * amp[None, :]
            input_map -= fitted

            outmap[mode_index, ...] = amp

        input_map.shape = mapshp
        outmap = outmap.reshape((mode_n, ) + mapshp[1:])
        return outmap, input_map

