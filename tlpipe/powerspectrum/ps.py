"""Module to estimate the power spectrum."""

from caput import mpiutil
from tlpipe.pipeline.pipeline import SingleBase

import h5py as h5
import numpy as np
import scipy as sp
import healpy as hp
import functions
import algebra

import matplotlib.pyplot as plt


class Ps(SingleBase):
    """Module to estimate the power spectrum."""

    params_init = {
            'pstype' : 'Cl', # Cl or Pk
            'lmax' : 1000,
            'input_files' : '',
            'output_files' : '',
            }

    prefix = 'ps_'

    def setup(self):
        if mpiutil.rank0:
            print 'Setting up Ps.'

    def read_input(self):

        result = h5.File(self.input_files[0], 'r')
        if self.params['pstype'] == 'Cl':
            map_cleaned = result['map_cleaned'][:]
        elif self.params['pstype'] == 'Pk':
            map_cleaned = result['map_cleaned'][:]
            self.ra   = result['ra'][:] * np.pi / 180.
            self.dec  = result['dec'][:] * np.pi / 180.
            self.freq = result['freq'][:] * 1.e9

            print self.ra
            print self.dec
            print self.freq

        return map_cleaned

    def process(self, input):

        if self.params['pstype'] == 'Cl':
            lmax = self.params['lmax']
            Cls = np.zeros((input.shape[0], lmax+1))
            for i in range(input.shape[0]):

                map_cleaned = np.mean(input[i], axis=0)
                Cls[i] = hp.anafast(map_cleaned, lmax=lmax)
            return Cls
        elif self.params['pstype'] == 'Pk':
            ps_1d_list = []
            k_ct = None
            for i in range(input.shape[0]):
                map_cleaned = input[i]
                map_axes = ('freq', 'ra', 'dec')
                map_info = {'axes': map_axes, 'type': 'vect'}
                map_info['freq_delta'] = self.freq[1] - self.freq[0]
                map_info['ra_delta']   = self.ra[1]   - self.ra[0]
                map_info['dec_delta']  = self.dec[1]  - self.dec[0]
                map_info['freq_centre']= self.freq[self.freq.shape[0]//2]
                map_info['ra_centre']  = self.ra[self.ra.shape[0]//2]
                map_info['dec_centre'] = self.dec[self.dec.shape[0]//2]
                map_cleaned = algebra.make_vect(map_cleaned, axis_names=map_axes)
                map_cleaned.info = map_info

                ps_2d, kn_2d, ps_1d, kn_1d, k_ct = est_power(map_cleaned)
                ps_1d_list.append(ps_1d)

            ps_1d_list = np.array(ps_1d_list)
            return ps_1d_list, k_ct

    def write_output(self, input):

        if self.params['pstype'] == 'Cl':
            ell = np.arange(input.shape[-1])

            for i in range(input.shape[0]):
                plt.plot(ell[1:], input[i][1:], '.-')

            #plt.loglog()
            plt.semilogy()
            plt.show()

            result = h5.File(self.output_files[0], 'w')
            result['Cls'] = input[:,1:]
            result['ell'] = ell[1:]
            result.close()

        elif self.params['pstype'] == 'Pk':
            print input[1].shape
            print input[0].shape
            for i in range(input[0].shape[0]):
                plt.plot(input[1], input[0][i], '.-')

            #plt.loglog()
            plt.semilogy()
            plt.show()

            result = h5.File(self.output_files[0], 'w')
            result['Pks'] = input[0]
            result['kc']  = input[1]
            result.close()


def est_power(map, map2=None, kbin_min=0.1, kbin_max=5., kbin_num=15):

    if map2 == None:
        map2 = map

    weight = np.ones_like(map)

    k_edges_p = np.logspace(np.log10(kbin_min), np.log10(kbin_max), num=kbin_num + 1)
    k_edges_v = np.logspace(np.log10(kbin_min), np.log10(kbin_max), num=kbin_num + 1)

    k_space = k_edges_p[-1]/k_edges_p[-2]
    k_centr = k_edges_p[:-1]*k_space

    ps_box = functions.BOX(map, map2, weight, weight)
    ps_box.mapping_to_xyz()
    ps_box.estimate_ps_3d()
    ps_box.convert_ps_to_unitless()
    ps_box.convert_3dps_to_2dps(k_edges_p, k_edges_v)
    ps_box.convert_3dps_to_1dps(k_edges_p)

    return ps_box.ps_2d, ps_box.kn_2d, ps_box.ps_1d, ps_box.kn_1d, k_centr

