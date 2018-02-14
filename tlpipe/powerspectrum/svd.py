#! /usr/bin/env python 

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
#from foreground_clean import find_modes
import find_modes
from core import algebra
from scipy.ndimage.filters import convolve
import beam as beam 
from mkpower import functions
import copy

import threading
from Queue import Queue
import tables
import sys
import os

from mpi4py import MPI

def add_pointsources(map_shape, freq, alpha0=4.5, sigma=0.5, A=1, number=1):

    map = np.zeros(map_shape)
    spec_list = []

    for i in range(number):
        ra  = np.random.randint(0, map_shape[1])
        dec = np.random.randint(0, map_shape[2])

        alpha = np.random.normal(alpha0, sigma, 1)
        spec = A * (freq/150.)**alpha
        spec_list.append(spec)

        map[:, ra, dec] += spec

    out = np.zeros(map_shape)
    for i in range(map_shape[0]):
        kernel = np.arange(41) - 20. #GBT
        #kernel = np.arange(21) - 10.
        kernel = sp.exp(-kernel**2 / (2. * 3 ** 2.))
        kernel *= 1. / (2. * sp.pi * 3 ** 2.)
        kernel = kernel[:, None] * kernel[None, :]
        convolve(map[i], kernel, output=out[i])

    map = out

    return map, spec_list

def add_pointsources_unresolved(map_shape, freq, alpha0=4.5, sigma=0.5):

    # ArXiv 1106.0007 Adrian Liu & Max Tegmark, eq. (10)
    x_max = np.random.poisson(lam=1., size=map_shape[1:]).reshape(map_shape[1:])

    gamma = 1.75
    B = 4.0
    alpha_ps = alpha0 - 2.
    sigma_alpha = sigma

    m_ps = (17.4 * x_max ** (2 - gamma)) * (B / 4.0) * ((2 - gamma)/0.25)**(-1)

    f_ps = (freq/150.) ** (-alpha_ps + sigma_alpha**2./2.*np.log(freq/150.))

    m_ps = m_ps[None, :, :] * f_ps[:, None, None]

    return m_ps

def convolve_beam(sim_map, apply_bandpass=True, frequency_depended=True, plot_beam=False):
    '''
        convolve the frequency depended beam to the map
    '''

    #freq_data = sp.array([1250, 1275, 1300, 1325, 1350, 1430], dtype=float)
    #beam_data = sp.array([14.4, 14.4, 14.4, 14.4, 14.4, 14.4])/60.
    #beam_data = beam_data*1420/freq_data 
    #freq_data *= 1.0e6

    # for GBT telescope
    if frequency_depended:
        beam_data = np.array([0.316148488246, 0.306805630985, 0.293729620792, 
                              0.281176247549, 0.270856788455, 0.26745856078,  
                              0.258910010848, 0.249188429031,])
    else:
        beam_data = np.array([0.316148488246, 0.316148488246, 0.316148488246, 
                              0.316148488246, 0.316148488246, 0.316148488246, 
                              0.316148488246, 0.316148488246,])
        beam_data *= 1.1
    freq_data = np.array([695, 725, 755, 785, 815, 845, 875, 905], dtype=float)
    freq_data *= 1.0e6

    # for Parkes telescope
    #freq_data = sp.array([1250, 1275, 1300, 1325, 1350, 1430], dtype=float)
    #beam_data = sp.array([14.4, 14.4, 14.4, 14.4, 14.4, 14.4])/60. 
    #if frequency_depended:
    #    beam_data = beam_data*1420/freq_data
    #freq_data *= 1.0e6

    #beamobj = beam.GaussianBeam(beam_data, freq_data)
    beamobj = beam.SincBeam(beam_data, freq_data)

    if apply_bandpass:
        bandpass = np.arange(sim_map.shape[0]) / (sim_map.shape[0] - 1.) * 2 * np.pi
        bandpass *= 10.
        bandpass = (np.sin(bandpass) + 20.)/20.
        #bandpass *= np.arange(0.5, 1., sim_map.shape[0])

        # consider a spacial varying bandpass
        bandpass_spacial = np.ones(sim_map.shape) + 0.2*np.random.random(sim_map.shape)
        sim_map *= bandpass_spacial
    else:
        bandpass = None


    sim_map_wb = beamobj.apply(sim_map, bandpass=bandpass, plot_beam=plot_beam)

    return sim_map_wb

def plot_eig(eigs, mode_n=5):

    color_list = ['r', 'b', 'g', 'c', 'k']

    fig = plt.figure(figsize=(8,8))
    ax_val = fig.add_axes([0.07, 0.07, 0.44, 0.88])
    ax_vec = []

    h = 0.88 / float(mode_n)
    for i in range(mode_n):
        lo = (0.52, 0.07 + (mode_n - 1 - i)*h, 0.44, h)
        ax_vec.append(fig.add_axes(lo))

    for i, svd in enumerate(eigs):

        (singular_values, left, right) = svd

        ax_val.plot(range(len(singular_values)), 
                singular_values/singular_values[0], color_list[i]+'-', marker='.')

        for j in range(mode_n):
            ax_vec[j].plot(range(len(left[0])), left[j], color_list[i]+'-')
            ax_vec[j].plot(range(len(left[0])), right[j], color_list[i]+':')

            ax_vec[j].set_yticklabels([])
            if j != mode_n - 1:
                ax_vec[j].set_xticklabels([])

    ax_val.set_ylabel('Singular Value')
    ax_val.set_xlim(xmin=-1, xmax=30)
    ax_val.semilogy()

    plt.savefig('eig.png', formate='png')

    plt.show()

def plot_map(map, freq_ind=None, filename='', freq=None, ra=None, dec=None):

    if freq == None:
        freq = map.get_axis('freq')
        ra   = map.get_axis('ra')
        dec  = map.get_axis('dec')

    X, Y = np.meshgrid(ra, dec)
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_axes([0.07, 0.07, 0.825, 0.88])
    cax = fig.add_axes([0.90, 0.07, 0.01, 0.88])
    
    c = ax.pcolormesh(X, Y, map[10].T, vmax=map[10].max(), vmin=map[10].min())
    ax.set_xlim(xmin=ra.min(), xmax=ra.max())
    ax.set_ylim(ymin=dec.min(), ymax=dec.max())
    ax.set_xlabel('RA')
    ax.set_ylabel('Dec')
    fig.colorbar(c, cax=cax, ax=ax)

    if filename!=None:
        plt.savefig(filename, formate='png')

    #plt.show()

def sub_modes(map, modes):

    outmap = np.empty((len(modes), ) + map.shape[1:])

    for mode_index, mode_vector in enumerate(modes):
        mode_vector = mode_vector.reshape([map.shape[0],])

        amp = sp.tensordot(mode_vector, map, axes=(0,0))

        fitted = mode_vector[:, None, None] * amp[None, :, :]
        map -= fitted

        outmap[mode_index, :, :] = amp

def est_power(map, map2=None, kbin_min=0.005, kbin_max=10., kbin_num=50):

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

def plot_power(ps_list, k_list, label_list=None, error_list=None, ax=None):

    if ax == None:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_axes([0.07, 0.07, 0.88, 0.88])

    color_list = ['r', 'b', 'g', 'c', 'k', 'y']
    if label_list == None:
        label_list = np.arange(len(ps_list))
        label_list = label_list.astype('str').tolist()

    k_space = k_list[0][1]/k_list[0][0]
    k_shift = k_space**(0.6/float(len(ps_list)))**(np.arange(len(ps_list))-len(ps_list)//2)

    for i in range(len(ps_list)):

        color = color_list[i]

        ps = ps_list[i]
        k = k_list[i]# * k_shift[i]

        good = ps!=0
        k  = k[good]
        ps = ps[good]

        k_negative = k[ps<0]
        ps_negative = ps[ps<0]

        if error_list!=None:
            error = error_list[i][good]

            k_positive = k[ps>0]
            ps_positive = ps[ps>0]
            err_positive = error[ps>0]

            err_u = err_positive[None,:]
            err_l = err_positive
            err_l[err_l>ps_positive] = ps_positive[err_l>ps_positive] - 1.e-20
            err_l = err_l[None, :]
            err = np.concatenate([err_l, err_u], axis=0)

            ax.errorbar(k_positive, ps_positive, err, 
                    fmt=color+'o', mfc=color, mec=color, ms=4, mew=1,
                    label=label_list[i])

            k_negative = k[ps<0]
            ps_negative = -ps[ps<0]
            err_negative = error[ps<0]

            err_u = err_negative[None,:]
            err_l = err_negative
            err_l[err_l>ps_negative] = ps_negative[err_l>ps_negative] - 1.e-20
            err_l = err_l[None, :]
            err = np.concatenate([err_l, err_u], axis=0)

            ax.errorbar(k_negative, ps_negative, err_negative, 
                    fmt=color+'o', mfc='none', mec=color, ms=4, mew=1)
        else:

            #ax.plot(k, ps, color_list[i]+'-', marker='o', label=label_list[i],
            #        mec=color_list[i], mfc=color_list[i], mew=2)
            #ax.plot(k_negative, -ps_negative, color_list[i]+'o', marker='o', 
            #        mec=color_list[i], mfc='none', mew=2)
            ax.step(k, ps, color_list[i], where='mid', label=label_list[i])
            ax.plot(k_negative, -ps_negative, color_list[i]+'o', marker='o', 
                    mec=color_list[i], mfc='none', mew=2)
        ax.loglog()
        ax.set_xlim(xmin=k.min(), xmax=k.max())
        ax.set_ylim(ymin=1.e-14, ymax=1.e-5)
        ax.legend(frameon=False, loc=0)

    return ax

class SimResult():

    def __init__(self, output_root, temp_array=None):

        self.file = tables.open_file(output_root, mode='w')
        self.root = self.file.root

        if temp_array != None:
            self.init_info(temp_array)

        self.path_list = ['raw', 'beam', 'beamf', 'beamb']
        self.kind_list = ['sim', 'sim_fg', 'sim_fg_cln']

        for path in self.path_list:
            self.file.create_group(self.root, path)
            for kind in self.kind_list:
                self.file.create_group('/' + path, kind)

    def init_info(self, temp_array):

        class Info(tables.IsDescription):
            ra_centre   = tables.Float32Col(pos=0)
            ra_delta    = tables.Float32Col(pos=1)
            dec_centre  = tables.Float32Col(pos=2)
            dec_delta   = tables.Float32Col(pos=3)
            freq_centre = tables.Float32Col(pos=4)
            freq_delta  = tables.Float32Col(pos=5)

        info_table = self.file.create_table(self.root, 'info', Info)

        info_row = info_table.row
        info_row['ra_centre']   = temp_array.info['ra_centre']
        info_row['ra_delta']    = temp_array.info['ra_delta']
        info_row['dec_centre']  = temp_array.info['dec_centre']
        info_row['dec_delta']   = temp_array.info['dec_delta']
        info_row['freq_centre'] = temp_array.info['freq_centre']
        info_row['freq_delta']  = temp_array.info['freq_delta']
        info_row.append()
        info_table.flush()

    def add_map(self, path, map, name_id=0, suffix=''):
        self.file.create_array(path, 'map_%03d'%name_id + suffix, map)

    def add_ps(self, path, ps_result, name_id = 0, suffix=''):
        #[ps_2d, kn_2d, ps_1d, kn_1d, k_bin] = ps_result
        self.file.create_array(path, 'ps2d_%03d'%name_id + suffix, ps_result[0])
        self.file.create_array(path, 'kn2d_%03d'%name_id + suffix, ps_result[1])
        self.file.create_array(path, 'ps1d_%03d'%name_id + suffix, ps_result[2])
        self.file.create_array(path, 'kn1d_%03d'%name_id + suffix, ps_result[3])
        self.file.create_array(path, 'kbin_%03d'%name_id + suffix, ps_result[4])

    def add_svd(self, path, svd, name_id=0, suffix=''):
        s, l, r = svd
        self.file.create_array(path, 'svd_s_%03d'%name_id + suffix, s)
        self.file.create_array(path, 'svd_l_%03d'%name_id + suffix, l)
        self.file.create_array(path, 'svd_r_%03d'%name_id + suffix, r)

    def __del__(self):

        self.file.close()


def add_info_array(file, group, info_array):

    class Info(tables.IsDescription):
        ra_centre   = tables.Float32Col(pos=0)
        ra_delta    = tables.Float32Col(pos=1)
        dec_centre  = tables.Float32Col(pos=2)
        dec_delta   = tables.Float32Col(pos=3)
        freq_centre = tables.Float32Col(pos=4)
        freq_delta  = tables.Float32Col(pos=5)

    info_table = file.create_table(group, 'info', Info)
    info_row = info_table.row
    print info_array.info['ra_centre']
    info_row['ra_centre'] = info_array.info['ra_centre']
    info_row['ra_delta']  = info_array.info['ra_delta']
    info_row['dec_centre'] = info_array.info['dec_centre']
    info_row['dec_delta']  = info_array.info['dec_delta']
    info_row['freq_centre'] = info_array.info['freq_centre']
    info_row['freq_delta']  = info_array.info['freq_delta']
    info_row.append()
    info_table.flush()

    info_array = file.create_array(group, 'map', info_array)

def svd_simulation(input_root, output_root='./', sim_num=100, fg_num=1, mode_num=4):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    for ii in range(sim_num):

        # initialize the hdf5 output
        sr = SimResult(output_root + 'RANK%02d_sim_raw_%03d.h5'%(rank, ii))
        print "using  simulation map: " + input_root + "sim_raw_%03d.npy"%ii
        print "output simulation map: " + output_root + "sim_raw_%03d.h5"%ii
        sys.stdout.flush()

        # load the raw simulation maps, and init the freq, ra, and dec
        sim_map_raw = algebra.make_vect(algebra.load(input_root + 'sim_raw_%03d.npy'%ii))
        freq = sim_map_raw.get_axis('freq')
        ra = sim_map_raw.get_axis('ra')
        dec = sim_map_raw.get_axis('dec')

        # estimate the ps for raw simulation map
        ps_result_raw = est_power(sim_map_raw)
        #[ps_2d_raw, kn_2d_raw, ps_1d_raw, kn_1d_raw, k_bin_raw] = ps_result_raw

        # initialize the map information
        sr.init_info(sim_map_raw)

        # save the raw simulation map
        if rank == 0:
            print "output sim_map_raw: map and ps_result"
        sr.add_map('/raw/sim/', sim_map_raw)
        sr.add_ps('/raw/sim/', ps_result_raw)

        # convolve the raw simulation map with common beam
        sim_map_raw_beam = convolve_beam(copy.deepcopy(sim_map_raw), 
                apply_bandpass=False, frequency_depended=False)
        ps_result_raw_beam_auto = est_power(sim_map_raw_beam)
        ps_result_raw_beam_cros = est_power(sim_map_raw_beam, sim_map_raw)
        if rank == 0:
            print "output sim_map_raw_beam: map and ps_result"
        sr.add_map('/beam/sim/', sim_map_raw_beam)
        sr.add_ps('/beam/sim/', ps_result_raw_beam_auto, suffix='_auto')
        sr.add_ps('/beam/sim/', ps_result_raw_beam_cros, suffix='_cros')

        # convolve the raw simulation map with frequency depended beam
        sim_map_raw_beamf = convolve_beam(copy.deepcopy(sim_map_raw), 
                apply_bandpass=False, frequency_depended=True)
        ps_result_raw_beamf_auto = est_power(sim_map_raw_beamf)
        ps_result_raw_beamf_cros = est_power(sim_map_raw_beamf, sim_map_raw)
        if rank == 0:
            print "output sim_map_raw_beamf: map and ps_result"
        sr.add_map('/beamf/sim/', sim_map_raw_beamf)
        sr.add_ps('/beamf/sim/', ps_result_raw_beamf_auto, suffix='_auto')
        sr.add_ps('/beamf/sim/', ps_result_raw_beamf_cros, suffix='_cros')

        # convolve the raw simulation map with beam * bandpass
        sim_map_raw_beamb = convolve_beam(copy.deepcopy(sim_map_raw), 
                apply_bandpass=True, frequency_depended=False)
        ps_result_raw_beamb_auto = est_power(sim_map_raw_beamb)
        ps_result_raw_beamb_cros = est_power(sim_map_raw_beamb, sim_map_raw)
        if rank == 0:
            print "output sim_map_raw_beamb: map and ps_result"
        sr.add_map('/beamb/sim/', sim_map_raw_beamb)
        sr.add_ps('/beamb/sim/', ps_result_raw_beamb_auto, suffix='_auto')
        sr.add_ps('/beamb/sim/', ps_result_raw_beamb_cros, suffix='_cros')

        if rank == 0:
            print 50*'='
        sys.stdout.flush()

        # add foreground , both resolved and unresolved
        for jj in range(fg_num):

            # make a copy of the simulation map
            sim_map = copy.deepcopy(sim_map_raw)

            pts_map, spec_list = add_pointsources(sim_map.shape, freq/1.e6, number=50)
            sim_map += pts_map
            pts_map_unresolved = add_pointsources_unresolved(sim_map.shape, freq/1.e6 )
            sim_map += pts_map_unresolved

            # save the simulatin maps with foreground added
            if rank == 0:
                print "output sim_map + fg: map"
            sr.add_map('/raw/sim_fg/', sim_map, name_id=jj)

            # convolved a beam of 1.1 times largest beam, which DO NOT have 
            # frequency evolution.
            # this case is similar to our common beam convolved result.
            sim_map_beam   = convolve_beam(copy.deepcopy(sim_map), 
                    apply_bandpass=False, frequency_depended=False)
            if rank == 0:
                print "output sim_map_beam + fg: map"
            sr.add_map('/beam/sim_fg/', sim_map_beam, name_id=jj)

            # convoled with a beam having frequency evolution
            sim_map_beamf = convolve_beam(copy.deepcopy(sim_map), 
                    apply_bandpass=False, frequency_depended=True)
            if rank == 0:
                print "output sim_map_beamf + fg: map"
            sr.add_map('/beamf/sim_fg/', sim_map_beamf, name_id=jj)

            # convoled with a beam, no frequency evolution, but have strange bandpass
            sim_map_beamb = convolve_beam(copy.deepcopy(sim_map), 
                    apply_bandpass=True, frequency_depended=False)
            if rank == 0:
                print "output sim_map_beamb + fg: map"
            sr.add_map('/beamb/sim_fg/', sim_map_beamb, name_id=jj)

            # remove the foreground
            corr, weight = find_modes.freq_covariance( sim_map, sim_map, None, None, 
                    range(freq.shape[0]), range(freq.shape[0]), no_weight=True)
            svd = find_modes.get_freq_svd_modes(corr, 30)
            if rank == 0:
                print "output sim_map + fg: svd"
            sr.add_svd('/raw/sim_fg_cln/', svd, name_id=jj)
            for i in range(mode_num):
                sub_modes(sim_map, svd[1][i:i+1])
                ps_result_auto = est_power(sim_map)
                ps_result_cros = est_power(sim_map, sim_map_raw)
                if rank == 0:
                    print "output sim_map + fg: %02d modes map and ps_result"%i
                sr.add_map('/raw/sim_fg_cln/', sim_map, name_id=jj, suffix='_%02dmode'%i)
                sr.add_ps('/raw/sim_fg_cln/', ps_result_auto, name_id=jj, suffix='_auto_%02dmode'%i)
                sr.add_ps('/raw/sim_fg_cln/', ps_result_cros, name_id=jj, suffix='_cros_%02dmode'%i)

            corr_beam, weight_beam = find_modes.freq_covariance( sim_map_beam, sim_map_beam, 
                    None, None, range(freq.shape[0]), range(freq.shape[0]), no_weight=True)
            if rank == 0:
                print "output sim_map_beam + fg: svd"
            svd_beam = find_modes.get_freq_svd_modes(corr_beam, 30)
            sr.add_svd('/beam/sim_fg_cln/', svd_beam, name_id=jj)
            for i in range(mode_num):
                sub_modes(sim_map_beam, svd_beam[1][i:i+1])
                ps_result_beam_auto = est_power(sim_map_beam)
                ps_result_beam_cros = est_power(sim_map_beam, sim_map_raw)
                if rank == 0:
                    print "output sim_map_beam + fg: %02d modes map and ps_result"%i
                sr.add_map('/beam/sim_fg_cln/', sim_map_beam, name_id=jj, suffix='_%02dmode'%i)
                sr.add_ps('/beam/sim_fg_cln/', ps_result_beam_auto, name_id=jj, suffix='_auto_%02dmode'%i)
                sr.add_ps('/beam/sim_fg_cln/', ps_result_beam_cros, name_id=jj, suffix='_cros_%02dmode'%i)

            corr_beamf, weight_beamf = find_modes.freq_covariance( sim_map_beamf, sim_map_beamf, 
                    None, None, range(freq.shape[0]), range(freq.shape[0]), no_weight=True)
            if rank == 0:
                print "output sim_map_beamf + fg: svd"
            svd_beamf = find_modes.get_freq_svd_modes(corr_beamf, 30)
            sr.add_svd('/beamf/sim_fg_cln/', svd_beamf, name_id=jj)
            for i in range(mode_num):
                sub_modes(sim_map_beamf, svd_beamf[1][i:i+1])
                ps_result_beamf_auto = est_power(sim_map_beamf)
                ps_result_beamf_cros = est_power(sim_map_beamf, sim_map_raw)
                if rank == 0:
                    print "output sim_map_beamf + fg: %02d modes map and ps_result"%i
                sr.add_map('/beamf/sim_fg_cln/', sim_map_beamf, name_id=jj, suffix='_%02dmode'%i)
                sr.add_ps('/beamf/sim_fg_cln/', ps_result_beamf_auto, name_id=jj, suffix='_auto_%02dmode'%i)
                sr.add_ps('/beamf/sim_fg_cln/', ps_result_beamf_cros, name_id=jj, suffix='_cros_%02dmode'%i)

            corr_beamb, weight_beamb = find_modes.freq_covariance( sim_map_beamb, sim_map_beamb, 
                    None, None, range(freq.shape[0]), range(freq.shape[0]), no_weight=True)
            if rank == 0:
                print "output sim_map_beamb + fg: svd"
            svd_beamb = find_modes.get_freq_svd_modes(corr_beamb, 30)
            sr.add_svd('/beamb/sim_fg_cln/', svd_beamb, name_id=jj)
            for i in range(mode_num):
                sub_modes(sim_map_beamb, svd_beamb[1][i:i+1])
                ps_result_beamb_auto = est_power(sim_map_beamb)
                ps_result_beamb_cros = est_power(sim_map_beamb, sim_map_raw)
                if rank == 0:
                    print "output sim_map_beamb + fg: %02d modes map and ps_result"%i
                sr.add_map('/beamb/sim_fg_cln/', sim_map_beamb, name_id=jj, suffix='_%02dmode'%i)
                sr.add_ps('/beamb/sim_fg_cln/', ps_result_beamb_auto, name_id=jj, suffix='_auto_%02dmode'%i)
                sr.add_ps('/beamb/sim_fg_cln/', ps_result_beamb_cros, name_id=jj, suffix='_cros_%02dmode'%i)

            if rank == 0:
                print 50*'-'
            sys.stdout.flush()
            comm.barrier()

        if rank == 0:
            print 50*'='

def check_result(input_root, type="cros"):

    file_list = os.listdir(input_root)

    path_list = [
            '/raw/sim_fg_cln/',
            '/beam/sim_fg_cln/',
            '/beamf/sim_fg_cln/',
            '/beamb/sim_fg_cln/',
            ]
    name_list = [
            '_%03d_' + type + '_02mode',
            '_%03d_' + type + '_02mode',
            '_%03d_' + type + '_02mode',
            '_%03d_' + type + '_02mode',
            ]

    ps1d_plotlist = []
    kbin_plotlist = []
    label_plotlist = []
    error_plotlist = []

    for i in range(len(path_list)):

        path = path_list[i]
        name = name_list[i]

        label_plotlist.append(path)

        ps1d_list = []
        kbin_list = []

        for file_name in file_list:
            file = tables.open_file(input_root + file_name)
            for ii in range(10):
                ps1d_list.append(file.getNode(path, 'ps1d' + name%ii).read())
                kbin_list.append(file.getNode(path, 'kbin' + name%ii).read())
            file.close()

        ps1d_list = np.array(ps1d_list)
        kbin_list = np.array(kbin_list)

        #ps1d_list[ps1d_list==0] = np.ma.masked

        ps1d = np.mean(ps1d_list, axis=0)
        ps1d_error = np.std(ps1d_list, axis=0)
        kbin = kbin_list[0]

        ps1d_plotlist.append(ps1d)
        kbin_plotlist.append(kbin)
        error_plotlist.append(ps1d_error)

    ax = plot_power(ps1d_plotlist, kbin_plotlist, label_plotlist, error_plotlist)

    file = tables.open_file(input_root + file_list[0])
    plot_power( [file.getNode('/raw/sim/',   'ps1d_000').read(), 
                 file.getNode('/beam/sim/',  'ps1d_000_' + type).read(),
                 file.getNode('/beamf/sim/', 'ps1d_000_' + type).read(),
                 file.getNode('/beamb/sim/', 'ps1d_000_' + type).read(),],
                [file.getNode('/raw/sim/',   'kbin_000').read(),
                 file.getNode('/beam/sim/',  'kbin_000_' + type).read(),
                 file.getNode('/beamf/sim/', 'kbin_000_' + type).read(),
                 file.getNode('/beamb/sim/', 'kbin_000_' + type).read(),],
                ['/raw/sim/', '/beam/sim/', '/beamf/sim/', '/beamb/sim/',],
                ax = ax)
    plt.show()

def test():

    #sim_map = algebra.make_vect(algebra.load( './maps/pks/sim_raw_000.npy'))
    sim_map_raw = algebra.make_vect(algebra.load( './maps/gbt/sim_raw_000.npy'))
    #sim_map_wb = convolve_beam(sim_map, apply_bandpass=False, 
    #                           frequency_depended=False, plot_beam=True)
    #exit()
    sim_map = copy.deepcopy(sim_map_raw)
    
    freq = sim_map.get_axis('freq')
    ra = sim_map.get_axis('ra')
    dec = sim_map.get_axis('dec')
    
    print freq.max(), freq.min()
    
    #plot_map(sim_map, filename='sim_raw.png')
    ps_2d, kn_2d, ps_1d, kn_1d, k_bin = est_power(sim_map)

    #plot_power([ps_1d,], [k_bin,], label_list=['raw',])

    #exit()

    
    pts_map, spec_list = add_pointsources(sim_map.shape, freq/1.e6, number=50)
    sim_map += pts_map
    pts_map_unresolved = add_pointsources_unresolved(sim_map.shape, freq/1.e6 )
    sim_map += pts_map_unresolved
    
    #plot_map(pts_map, filename='pts_map.png', freq=freq, ra=ra, dec=dec)
    #plot_map(pts_map_unresolved, filename='pts_map_unresolved.png', freq=freq, ra=ra, dec=dec)
    #plot_map(sim_map, filename='sim_plus_ps.png', freq=freq, ra=ra, dec=dec)

    corr, weight = find_modes.freq_covariance( sim_map, sim_map, None, None, 
            range(freq.shape[0]), range(freq.shape[0]), no_weight=True)
    svd = find_modes.get_freq_svd_modes(corr, 30)

    #corr_norm, weight_norm = find_modes.freq_covariance( sim_map, sim_map, None, None, 
    #        range(freq.shape[0]), range(freq.shape[0]), no_weight=True, normalize=True)
    #svd_norm = find_modes.get_freq_svd_modes(corr_norm, 30)
    
    # convolve beam
    #sim_map_wb = convolve_beam(sim_map, apply_bandpass=True, frequency_depended=True)
    #sim_map_wb = convolve_beam(sim_map, apply_bandpass=False, frequency_depended=True)
    sim_map_wb = convolve_beam(copy.deepcopy(sim_map), apply_bandpass=True, frequency_depended=False)
    #sim_map_wb = convolve_beam(sim_map, apply_bandpass=False, frequency_depended=False)
    
    #plot_map(sim_map_wb, filename='sim_plus_ps_cov.png')

    #log_sim_map_wb = np.log(np.abs(copy.deepcopy(sim_map_wb)))
    #log_sim_map_wb[np.isnan(log_sim_map_wb)] = 0
    #log_corr_wb, log_weight_wb = find_modes.freq_covariance( 
    #        log_sim_map_wb, log_sim_map_wb, None, None, 
    #        range(freq.shape[0]), range(freq.shape[0]), no_weight=True )
    #log_svd_wb = find_modes.get_freq_svd_modes(log_corr_wb, 30)

    #for i in range(5):
    #    sub_modes(log_sim_map_wb, log_svd_wb[1][i:i+1])

    #lin_sim_map_wb = np.exp(log_sim_map_wb) * np.abs(sim_map_wb)/sim_map_wb
    #lin_sim_map_wb[np.isnan(lin_sim_map_wb)] = 0.

    #corr_wb, weight_wb = find_modes.freq_covariance( lin_sim_map_wb, lin_sim_map_wb, None, None, 
    corr_wb, weight_wb = find_modes.freq_covariance( sim_map_wb, sim_map_wb, None, None, 
            range(freq.shape[0]), range(freq.shape[0]), no_weight=True )
    
    log_svd_wb = find_modes.get_freq_svd_modes(np.log(corr_wb), 30)
    svd_wb = find_modes.get_freq_svd_modes(corr_wb, 30)

    for i in range(3):
        sub_modes(sim_map, svd[1][i:i+1])
        #plot_map(sim_map, filename='sim_plot_ps_sub%02dmodes.png'%(i+1),
        #        freq=freq, ra=ra, dec=dec)

    for i in range(3):
        sub_modes(sim_map_wb, log_svd_wb[1][i:i+1])
        #plot_map(sim_map_wb, filename='sim_plot_ps_cov_sub%02dmodes.png'%(i+1),
        #        freq=freq, ra=ra, dec=dec)

    #plot_eig([log_svd_wb, svd_wb], mode_n=10)
    #exit()


    ps_2d_cln, kn_2d_cln, ps_1d_cln, kn_1d_cln, k_bin_cln = est_power(sim_map)
    ps_2d_cln_wb, kn_2d_cln_wb, ps_1d_cln_wb, kn_1d_cln_wb, k_bin_cln_wb = est_power(sim_map_wb, sim_map_raw)

    plot_power([ps_1d, ps_1d_cln, ps_1d_cln_wb], 
               [k_bin, k_bin_cln, k_bin_cln_wb], 
               label_list=['raw', 'cleaned', 'cleaned with beam'])
    plt.show()

    #plot_eig([svd, svd_wb], mode_n=10)



if __name__=="__main__":

    #test()
    #exit()

    #input_root = './maps/gbt/'
    input_root = '/home_local/ycli/gbt_bperr/'
    output_root = '/home_local/ycli/gbt_bperr/'
    #output_root = './maps_withfg/gbt_bperr/'
    #output_root = './maps_withfg/gbt/'

    #svd_simulation(input_root, output_root, sim_num=1, fg_num=10, mode_num=5)

    #check_result(output_root, type="cros")
    #check_result(output_root, type="auto")
    
