"""Phase visibility data to a source."""

try:
    import cPickle as pickle
except ImportError:
    import pickle

import os
import numpy as np
from scipy.linalg import eigh
import h5py
import ephem
import aipy as a
import scipy.signal as signal

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tlpipe.utils import mpiutil
from tlpipe.core.base_exe import Base
from tlpipe.core import tldishes
from tlpipe.utils.pickle_util import get_value
from tlpipe.utils.date_util import get_ephdate
from tlpipe.utils.path_util import input_path, output_path


# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
params_init = {
               'nprocs': mpiutil.size, # number of processes to run this module
               'aprocs': range(mpiutil.size), # list of active process rank no.
               'input_file': 'data_cal.hdf5',
               'output_file': 'data_phs2src.hdf5',
               'extra_history': '',
               'cal_phase' : 0,
               'cal_on_time' : 1, # in second
               'cal_off_time' : 4, # in second
               'cal_percent' : [50, 50],
               'average_size' : 1, # in unit of pixel
              }
prefix = 'nc_'


class NoiseCal(Base):
    """Phase visibility data to a source."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(NoiseCal, self).__init__(
                parameter_file_or_dict, params_init, prefix, feedback)

    def execute(self):

        input_file = input_path(self.params['input_file'])
        output_file = output_path(self.params['output_file'])

        params = self.params

        with h5py.File(input_file, 'r') as f:
            dset = f['data']
            data_shp = dset.shape
            data_type = dset.dtype
            ants = dset.attrs['ants']
            ts = f['time']
            freq = dset.attrs['freq']
            tzone = dset.attrs['timezone']

            npol = dset.shape[2]
            nt = len(ts)
            nfreq = len(freq)
            nants = len(ants)
            bls = [(ants[i], ants[j]) for i in range(nants) for j in range(i, nants)] # start from 1
            nbls = len(bls)

            int_time = dset.attrs["int_time"]

            for ii in range(nbls):

                a1 = bls[ii][0]
                a2 = bls[ii][1]

                print '%02d x %02d'%(a1, a2)

                #cal_p = [True,] * int(params['cal_on_time'] / int_time) \
                #        + [False,] * int(params['cal_off_time'] / int_time)
                cal_on_time  = int(params['cal_on_time'] / int_time)
                cal_off_time = int(params['cal_off_time'] / int_time)
                cal_len = cal_on_time + cal_off_time
                if params['cal_phase'] == None:
                    #cal_phase, cal_on_time = \
                    #        get_phase(np.mean(dset[:cal_len,ii,0,:], axis=-1),
                    #            perc=params['cal_percent'])
                    #print "Automatically get the cal phase:", cal_phase
                    #if cal_on_time != int(params['cal_on_time'] / int_time):
                    #    print "Automatically extend cal_on_time to %d"%cal_on_time
                    data_tmp = np.abs(dset[:cal_len,ii,:,:])
                    data_tmp = np.mean(np.sum(data_tmp, axis=-2), axis=-1)
                    cal_on, cal_off = sep_cal(data_tmp, perc=params['cal_percent'])
                else:
                    cal_phase = params['cal_phase']
                    cal_on  = [True, ] * cal_on_time + [False, ] * cal_off_time
                    cal_on  = np.roll(np.array(cal_on), cal_phase)
                    cal_off = np.logical_not(cal_on)

                cal_pn = nt / cal_len

                time_cut = cal_pn * cal_len
                dset_ii = dset[:, ii, ...][:time_cut, ...]
                ts_ii = ts[:time_cut]

                nt = len(ts_ii)
                data_shp = dset_ii.shape
                dset_ii = dset_ii.reshape((-1, cal_len)+data_shp[1:])
                ts_ii = ts_ii.reshape((-1, cal_len))

                dset_calon  = dset_ii[:, cal_on, ...]
                dset_caloff = dset_ii[:, cal_off, ...]

                ts_calon = ts_ii[:, cal_on]
                ts_caloff = ts_ii[:, cal_off]

                dset_diff =\
                        np.mean(dset_calon, axis=1) - np.mean(dset_caloff, axis=1)
                ts_diff = ts_calon[:,0]

                dset_calon  = dset_calon.reshape((-1,)+data_shp[1:])
                dset_caloff = dset_caloff.reshape((-1,)+data_shp[1:])

                if params['average_size'] > 1:

                    N = params['average_size']
                    print "Performing Moving average over each %d pixels"%N
                    dset_calon  = signal.fftconvolve( dset_calon, 
                            np.ones((N,1,1))/N, mode='same')
                    #dset_caloff = signal.fftconvolve( dset_caloff, 
                    #        np.ones((N,1,,1))/N, mode='same')

                    calon_tmp = ts_calon.shape + data_shp[1:]
                    caloff_tmp = ts_caloff.shape + data_shp[1:]
                    dset_diff = np.mean(dset_calon.reshape(calon_tmp), axis=1)\
                            - np.mean(dset_caloff.reshape(caloff_tmp), axis=1)

                ts_calon  = ts_calon.flatten()
                ts_caloff = ts_caloff.flatten()

                output_root = output_file.replace('.hdf5', '') 

                if a1 == a2:
                    #plot_amp(dset_calon, dset_caloff, 
                    #        ts_calon, ts_caloff, a1, output_root, tzone=tzone)
                    pass
                else:
                    #plot_phase(dset_calon, ts_calon, freq, a1, a2, 
                    #        output_root, suffix='_calon', tzone=tzone)
                    #plot_phase(dset_caloff, ts_caloff, freq, a1, a2, 
                    #        output_root, suffix='_caloff', tzone=tzone)
                    #plot_phase(dset_diff, ts_diff, freq, a1, a2, 
                    #        output_root, suffix='_diff', tzone=tzone)
                    plot_phase(dset_diff, ts_diff, freq, a1, a2, 
                            output_root, suffix='_diff', tzone=tzone, residue=True)

def sep_cal(data_tmp, perc):

    perc_on, perc_off = perc

    if perc_off == None:
        perc_off = perc_on

    cal_on  = data_tmp > np.percentile(data_tmp, perc_on)
    cal_off = data_tmp < np.percentile(data_tmp, perc_off)

    print 'Cal On:', cal_on, 'Cal Off', cal_off

    return cal_on, cal_off

def get_phase(data_tmp, perc=50):

    # return the index of the first value over the median

    #p = data_tmp == np.max(data_tmp)
    p = np.where(data_tmp > np.percentile(data_tmp, perc))[0]
    if len(p) > 1:
        diff = np.diff(p)
        if np.all(diff == 1): return p[0], len(p)
        else: 
            diff_great = np.where(diff>1)[0]
            if len(diff_great) > 1:
                print "Error, Noise Cal S/N is not high engouh"
            else: return p[diff_great[0] + 1], len(p)
    else: return p[0], 1

def plot_amp(calon, caloff, ts_calon, ts_caloff, a1, output_root, tzone='UTC+8' ):

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_axes([0.1, 0.52, 0.80, 0.38])
    ax2 = fig.add_axes([0.1, 0.10, 0.80, 0.38])

    time0 = ts_calon[0]
    time = ts_calon - time0
    time *= 24. * 60.

    #start_time = ephem.Date(time0 + ephem.hour * int(tzone[3:]))
    start_time = ephem.Date(0 + time0 - ephem.julian_date(ephem.Date(0))
            + ephem.hour * int(tzone[3:]))

    ax1.plot(time, np.mean(calon[:, 0, :].real, axis=-1), 'r.-', label='cal on')
    ax2.plot(time, np.mean(calon[:, 1, :].real, axis=-1), 'r.-', label='cal on')

    time = ts_caloff - time0
    time *= 24. * 60.

    ax1.plot(time, np.mean(caloff[:, 0, :].real, axis=-1), 'g.-', label='cal off')
    ax2.plot(time, np.mean(caloff[:, 1, :].real, axis=-1), 'g.-', label='cal off')

    ax1.set_title('Amplitude Ant%02d'%a1)
    ax1.set_ylabel('XX')
    ax1.minorticks_on()
    ax1.tick_params(length=4, width=1., direction='out')
    ax1.tick_params(which='minor', length=2, width=1., direction='out')
    ax1.set_xticklabels([])

    ax2.set_ylabel('YY')
    ax2.set_xlabel('time + %s'%start_time + '[minutes]')
    ax2.minorticks_on()
    ax2.tick_params(length=4, width=1., direction='out')
    ax2.tick_params(which='minor', length=2, width=1., direction='out')
    ax1.set_xlim(xmin=time.min(), xmax=time.max())
    ax2.set_xlim(xmin=time.min(), xmax=time.max())

    plt.savefig(output_root + '_ant%02d_amp.png'%a1 )

    plt.show()
    #plt.clf()
    plt.close()

def plot_phase(vis, time, freq, a1, a2, output_root, suffix='', tzone='UTC+8', residue=False):

    phs = np.angle(vis) * 180. / np.pi

    time0 = time[0]
    time = time - time0
    time *= 24. * 60.

    #start_time = ephem.Date(time0 + ephem.hour * int(tzone[3:]))
    start_time = ephem.Date(0 + time0 - ephem.julian_date(ephem.Date(0))
            + ephem.hour * int(tzone[3:]))

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_axes([0.1, 0.52, 0.80, 0.38])
    ax2 = fig.add_axes([0.1, 0.10, 0.80, 0.38])
    cax1 = fig.add_axes([0.91, 0.52, 0.01, 0.38])
    cax2 = fig.add_axes([0.91, 0.10, 0.01, 0.38])

    Y, X = np.meshgrid(freq, time)

    vmax = 180.1
    vmin = -180.1

    if residue:
        suffix += '_residue'
        #phs_ref = np.mean(phs, axis=0)
        ref_idx = time.shape[0] // 2
        phs_ref = phs[ref_idx,...]
        phs = phs - phs_ref[None, ...]

        phs[phs >  180] -= 360
        phs[phs < -180] += 360

        #mean = np.mean(phs[:,0,:])
        #var  = np.std(phs[:,0,:])
        vmax =  20
        vmin = -20
    
    im = ax1.pcolormesh(X, Y, phs[:, 0, :], vmax=vmax, vmin=vmin)
    ax1.set_title('Phase Angle Ant%02d'%a1 + r'$\times$' + 'Ant%02d '%a2 + suffix[1:])
    ax1.set_ylabel('XX Frequency [MHz]')
    ax1.minorticks_on()
    ax1.tick_params(length=4, width=1., direction='out')
    ax1.tick_params(which='minor', length=2, width=1., direction='out')
    ax1.set_xticklabels([])
    
    fig.colorbar(im, ax=ax1, cax=cax1)
    
    if residue:

        pass

        #mean = np.mean(phs[:,1,:])
        #var  = np.std(phs[:,1,:])
        #vmax = mean + 2*var
        #vmin = mean - 2*var
        vmax =  20
        vmin = -20
    
    im = ax2.pcolormesh(X, Y, phs[:, 1, :], vmax=vmax, vmin=vmin)
    ax2.set_ylabel('YY Frequency [MHz]')
    ax2.set_xlabel('time + %s'%start_time + '[minutes]')
    ax2.minorticks_on()
    ax2.tick_params(length=4, width=1., direction='out')
    ax2.tick_params(which='minor', length=2, width=1., direction='out')
    
    fig.colorbar(im, ax=ax2, cax=cax2)

    ax1.set_ylim(ymin=freq.min(), ymax=freq.max())
    ax2.set_ylim(ymin=freq.min(), ymax=freq.max())
    ax1.set_xlim(xmin=time.min(), xmax=time.max())
    ax2.set_xlim(xmin=time.min(), xmax=time.max())

    plt.savefig(output_root + '_ant%02dx%02d%s.png'%(a1, a2, suffix))

    plt.show()
    #plt.clf()
    plt.close()
