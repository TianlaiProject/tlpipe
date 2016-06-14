""" seperate the noise cal signal from the data """

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

from base_operation import BaseOperation

from tlpipe.utils import mpiutil
from tlpipe.core import tldishes
from tlpipe.utils.pickle_util import get_value
from tlpipe.utils.date_util import get_ephdate
from tlpipe.utils.path_util import input_path, output_path


class NoiseCal(BaseOperation):

    # Define a dictionary with keys the names of parameters to be read from
    # file and values the defaults.

    params_init = {
                   'extra_history': '',
                   'cal_phase' : None,
                   'cal_on_time' : 1, # in second
                   'cal_off_time' : 4, # in second
                   'cal_percent' : [50, 50],
                   'average_size' : 1, # in unit of pixel
                   'bl_set' : 'auto',
                  }
    prefix = 'nc_'
    
    def action(self, vis):

        params = self.params

        dset = np.ma.masked_invalid(vis.main_data.local_data)
        data_shp = dset.shape
        data_type = dset.dtype
        ants = vis['feedno']
        tzone = vis.attrs['timezone']
        ts = vis['jul_date'].local_data
        freq = vis['freq']

        nt = dset.shape[0]
        local_offset = vis.main_data.local_offset
        nfreq = dset.shape[1]
        nants = len(ants)
        bls = vis['blorder']
        nbls = len(bls)

        # combine all the data to get the cal_on and cal_off time
        int_time = vis.attrs["inttime"]
        cal_on_time  = int(params['cal_on_time'] / int_time)
        cal_off_time = int(params['cal_off_time'] / int_time)
        cal_len = cal_on_time + cal_off_time
        dset_tmp = np.ma.mean(np.ma.abs(dset[:cal_len,:,0,:]), axis=-1)

        print np.ma.mean(np.ma.abs(dset[:cal_len,:,0,:]), axis=-1)

        if params['cal_phase'] == None:
            cal_phase, cal_on_time = \
                    get_phase(np.mean(dset_tmp, axis=-1), perc=params['cal_percent'])
            print "Automatically get the cal phase:", cal_phase
            if cal_on_time != int(params['cal_on_time'] / int_time):
                print "Automatically extend cal_on_time to %d"%cal_on_time
            cal_on, cal_off = \
                    sep_cal(np.mean(dset_tmp, axis=-1), perc=params['cal_percent'])
            cal_off_time = np.sum(cal_off)
        else:
            cal_phase = params['cal_phase']
            cal_on  = [True, ] * cal_on_time + [False, ] * cal_off_time
            cal_on  = np.roll(np.array(cal_on), cal_phase)
            cal_off = np.logical_not(cal_on)

        print "Cal On %d, Cal Off %d, Phase %d\n"%(cal_on_time, 
                                                   cal_off_time, 
                                                   cal_phase)

        cal_pn = int(np.ceil(float(nt) / cal_len))
        cal_on = np.repeat(cal_on[None, :], cal_pn, axis=0).flatten()
        cal_on = cal_on[:nt]

        dset_calon = vis.main_data.local_data[cal_on, ...]
        ts_calon   = ts[cal_on]
        vis.create_dataset('noisecal', 
                shape=dset_calon.shape, dtype=dset_calon.dtype, data=dset_calon)
        vis.create_dataset('noisecal_jul_date', 
                shape=ts_calon.shape, dtype=ts_calon.dtype, data=ts_calon)
        vis.main_data.local_data[cal_on, ...] = np.NaN

def sep_cal(data_tmp, perc):

    perc_on, perc_off = perc

    if perc_off == None:
        perc_off = perc_on

    cal_on  = data_tmp > np.percentile([data_tmp.min(), data_tmp.max()], perc_on)
    cal_off = data_tmp < np.percentile([data_tmp.min(), data_tmp.max()], perc_off)

    #print 'Cal On:', cal_on, '\nCal Off', cal_off

    return cal_on, cal_off

def get_phase(data_tmp, perc=[50, 50]):

    # return the index of the first value over the median

    #p = data_tmp == np.max(data_tmp)
    p = np.where(data_tmp > np.percentile([data_tmp.min(), data_tmp.max()], perc[0]))[0]
    if len(p) > 1:
        diff = np.diff(p)
        if np.all(diff == 1): return p[0], len(p)
        else: 
            diff_great = np.where(diff>1)[0]
            if len(diff_great) > 1:
                print p
                plt.plot(data_tmp)
                plt.show()
                raise "Error, Noise Cal S/N is not high engouh"
            else: return p[diff_great[0] + 1], len(p)
    else: return p[0], 1

