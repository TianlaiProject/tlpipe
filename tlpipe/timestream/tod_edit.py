import numpy as np
import gc
import timestream_task
import h5py
from tlpipe.utils.path_util import output_path
from caput import mpiutil
from caput import mpiarray
from scipy.signal import medfilt
from scipy.signal import lombscargle

from scipy.interpolate import interp1d

import warnings
warnings.simplefilter('ignore', np.RankWarning)


class DataEdit(timestream_task.TimestreamTask):
    """
    Edit the Tod data observed by MeerKAT for 1/f noise analysis
    """

    params_init = {
            'bad_time_list' : None,
            'bad_freq_list' : None,
            'bandpass_cal'  : True,
            #'init_cal_pha'  : 2,
            #'cal_period'    : 10,
            'duty_frac'     : 0.9, # s
            'noise_flag'    : '/scratch/users/ycli/meerkat/1551037708_cal_flag.npy',
            'cal_data'      : None,
            'rm_ncal'       : False,
            'cal_with_nd'   : True,
            }

    prefix = 'de_'

    def process(self, ts):

        bad_time_list = self.params['bad_time_list']
        bad_freq_list = self.params['bad_freq_list']

        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']

        if bad_time_list is not None:
            #num_infiles = len(self.input_files)
            #name = ts.main_data_name
            #outfiles_map = ts._get_output_info(name, num_infiles)[-1]
            #st = 0
            #for fi, start, stop in outfiles_map:
            #    et = st + (stop - start)
            #    print "Mask bad time"
            #    for bad_time in bad_time_list:
            #        print bad_time
            #        ts.vis_mask[st:et, ...][slice(*bad_time), ...] = True
            #    st = et
            for bad_time in bad_time_list:
                print bad_time
                ts.vis_mask[slice(*bad_time), ...] = True


        if bad_freq_list is not None:
            print "Mask bad freq"
            for bad_freq in bad_freq_list:
                print bad_freq
                ts.vis_mask[:, slice(*bad_freq), ...] = True

        vis_abs = mpiarray.MPIArray.wrap(ts.vis[:].real, 0)
        ts.create_main_data(vis_abs, recreate=True, copy_attrs=True)

        ra  = ts['ra'][:]
        dec = ts['dec'][:]
        ts.create_time_and_bl_ordered_dataset('ra',  ra, recreate=True, copy_attrs=True)
        ts.create_time_and_bl_ordered_dataset('dec', dec, recreate=True, copy_attrs=True)

        func = ts.bl_data_operate
        if self.params['cal_with_nd']:
            print "Noise Diode calibration "
            func(self.data_edit, full_data=True, copy_data=False, 
                    show_progress=show_progress, 
                    progress_step=progress_step, keep_dist_axis=False)
        else:
            print "Directely calibration "
            if self.params['cal_data'] is None:
                print "Directely calibration needs gain solution"
            raise
            func(self.data_edit_direct_cal, full_data=True, copy_data=False, 
                    show_progress=show_progress, 
                    progress_step=progress_step, keep_dist_axis=False)

        #print ts.vis.shape
        #print np.var(ts.vis[:, :, 0, 0], axis=0)

        return super(DataEdit, self).process(ts)

    def data_edit_direct_cal(self, vis, vis_mask, li, gi, bl, ts, **kwargs):

        cal_on  = np.load(self.params['noise_flag'])
        vis_mask[cal_on, ...] = True
        cal_spec = read_noise_spec(self.params['cal_data'], ts.freq, gi)
        cal_spec[cal_spec==0] = np.inf
        vis /= cal_spec[None, :, :]

    def data_edit(self, vis, vis_mask, li, gi, bl, ts, **kwargs):

        _time = ts['sec1970'][:]

        ant = bl[0] - 1
        print "global index %3d: m%03d"%(gi, ant)
        cal_on  = np.load(self.params['noise_flag'])
        noise_cal = get_noise_cal(vis, _time, noise_flag = cal_on,
                duty_frac = self.params['duty_frac'],
                #cal_phase = self.params['init_cal_pha'],
                #period=self.params['cal_period'],
                )

        vis = cal_to_noise_cal(vis, noise_cal, cal_on, self.params['duty_frac'])

        if self.params['rm_ncal']:
            vis_mask[cal_on, ...] = True

        if self.params['cal_data'] is not None:
            #print ts.freq[0], ts.freq[1]
            cal_spec = read_noise_spec(self.params['cal_data'], ts.freq, ant)
            #cal_spec = polyfit_noise_spec(self.params['cal_data'], ts.freq[:], ant, 20)
            vis *= cal_spec[None, :, :]

def cal_to_noise_cal(vis, noise_cal, cal_on, duty_frac=0.9, sub_cal=True):

    vis_cal = vis[cal_on, ...].copy()
    vis_cal_shp = vis_cal.shape[1:]
    vis_cal.shape  = (-1, 2) + vis_cal_shp
    vis_cal += np.roll(vis_cal, 1, axis=1)
    vis_cal.shape = (-1, ) + vis_cal_shp
    if sub_cal:
        vis_cal -= noise_cal[cal_on] * duty_frac
    vis_cal /= 2.0
    vis[cal_on, ...] = vis_cal

    noise_cal[noise_cal == 0] = np.inf 
    vis /= noise_cal

    return vis

def read_noise_spec(cal_data, freq_in, ant, kind='linear'):

    with h5py.File(cal_data) as data:
        ant_names = np.array(data['ants'][:])
        ant_names = (ant_names[:, 0] - 1) == ant
        if not np.any(ant_names): 
            print "ant m%03d not used"%ant
            raise
        ant_index = (ant_names).nonzero()[0][0]
        freq = data['freq'][:]
        spec = data['noise_T'][...,ant_index]
        mask = data['spec_mask'][...,ant_index]
        spec = np.ma.array(spec)
        spec.mask = mask

    good = ~mask[:,0]
    intf = interp1d(freq[good], spec[good, 0], kind=kind, fill_value='extrapolate')
    spec_int_0 = intf(freq_in)[:, None]

    good = ~mask[:,1]
    intf = interp1d(freq[good], spec[good, 1], kind=kind, fill_value='extrapolate')
    spec_int_1 = intf(freq_in)[:, None]

    spec_int = np.concatenate([spec_int_0, spec_int_1], axis=1)
    return spec_int

def polyfit_noise_spec(cal_data, freq_fit, ant, deg=12):

    freq_fit = freq_fit.astype('float64')

    with h5py.File(cal_data, 'r') as data:
        ant_names = np.array(data['ants'][:])
        ant_names = (ant_names[:, 0] - 1) == ant
        if not np.any(ant_names): 
            print "ant m%03d not used"%ant
            raise
        ant_index = (ant_names).nonzero()[0][0]
        freq = data['freq'][:]
        spec = data['noise_T'][...,ant_index]
        mask = data['spec_mask'][...,ant_index]
        spec = np.ma.array(spec)
        spec.mask = mask

    spec_fit = np.zeros((freq_fit.shape[0], 2), dtype='float32')

    good = ~mask[:, 0] * (freq > 900.)
    if np.any(good):
        _fit = np.polyfit(freq[good], spec[good, 0], deg) #, full=True)[0]
        for ii in range(deg + 1):
            spec_fit[:, 0] += freq_fit ** (deg - ii) * _fit[ii]

    good = ~mask[:, 1] * (freq > 900.)
    if np.any(good):
        _fit = np.polyfit(freq[good], spec[good, 1], deg) #, full=True)[0]
        for ii in range(deg + 1):
            spec_fit[:, 1] += freq_fit ** (deg - ii) * _fit[ii]

    return spec_fit

def get_noise_cal(vis, time, noise_flag, duty_frac=0.9, smooth_cal=True,
        roll_for_off = True):

    _time = time
    vis_shp = vis.shape

    cal_on  = noise_flag[:vis_shp[0]]
    if roll_for_off:
        cal_off = cal_on.copy()
        cal_off = np.roll(cal_off, 2)
        vis_cal = vis[cal_on, ...] - vis[cal_off, ...]
    else:
        #cal_off = np.mean(vis[~cal_on, ...], axis=0)
        #cal_off = np.median(vis[~cal_on, ...], axis=0)
        deg = 4
        time0 = time[0]
        cal_off       = vis[~cal_on, ...]
        cal_off_shp   = cal_off.shape
        cal_off.shape = (cal_off_shp[0], -1)
        cal_off_time  = time[~cal_on] - time0
        cal_off_poly  = np.polyfit(cal_off_time, cal_off, deg=deg)

        vis_cal = vis[cal_on, ...]
        cal_on_time   = time[cal_on] - time0
        cal_off_fit   = np.zeros((vis_cal.shape[0], cal_off.shape[1]))
        for i in range(deg + 1):
            cal_off_fit += cal_on_time[:, None] ** (deg - i) * cal_off_poly[i][None, :]
        cal_off_fit.shape = vis_cal.shape
        vis_cal -= cal_off_fit
    time_cal = time[cal_on]

    time_cal.shape = (-1, 2)

    vis_cal_shp = vis_cal.shape
    vis_cal.shape = time_cal.shape + vis_cal_shp[1:]

    vis_cal = np.sum(vis_cal, axis=1) / duty_frac 
    if smooth_cal:
        kernal = [1, 21] + [1, ] * (vis_cal.ndim - 2)
        vis_cal = medfilt(vis_cal, kernal)
    time_cal = np.mean(time_cal, axis=1)

    interp_fillvalue = (vis_cal[0], vis_cal[-1])
    cal_intf = interp1d(time_cal, vis_cal, axis=0, 
            kind='linear', 
            #kind='nearest', 
            bounds_error = False,
            fill_value=interp_fillvalue)
    noise_cal = cal_intf(time)

    #noise_cal[noise_cal == 0] = np.inf
    #vis = vis/noise_cal
    #vis[cal_on, ...] -= 1.

    if roll_for_off:
        return noise_cal
    else:
        return noise_cal, cal_off_fit


#def cal_to_noise(vis, time, cal_phase=None, period=10, duty_time=1.8):
#    _time = time
#    vis_shp = vis.shape
#
#    if cal_phase is None:
#        cal_phase = np.argmax( np.sum(vis[:period], axis=tuple(range(1, len(vis_shp)))))
#    
#    cal_time0 = _time[cal_phase]
#    if cal_phase > 0.5 * period:
#        off_time0 = _time[cal_phase - 2]
#    else:
#        off_time0 = _time[cal_phase + 2]
#    cal_perio = _time[period] - _time[0]
#    print cal_phase, cal_time0, cal_perio
#
#    cal_stamp = (_time - cal_time0) / cal_perio
#    cal_stamp = cal_stamp - cal_stamp.astype('int')
#    cal_stamp = np.round(10 * cal_stamp, 0)
#    cal_on  = (cal_stamp == 0.) + (cal_stamp == 1.)
#
#    off_stamp = (_time - off_time0) / cal_perio
#    off_stamp = off_stamp - off_stamp.astype('int')
#    off_stamp = np.round(10 * off_stamp, 0)
#    cal_off  = (off_stamp == 0.) + (off_stamp == 1.)
#
#    vis_cal = vis[cal_on, ...] - vis[cal_off, ...]
#    time_cal = time[cal_on]
#
#    time_cal.shape = (-1, 2)
#
#    vis_cal_shp = vis_cal.shape
#    vis_cal.shape = time_cal.shape + vis_cal_shp[1:]
#
#    vis_cal = np.sum(vis_cal, axis=1) / duty_time
#    time_cal = np.mean(time_cal, axis=1)
#
#    cal_intf = interp1d(time_cal, vis_cal, axis=0, kind='nearest', 
#            fill_value='extrapolate')
#    noise_cal = cal_intf(time)
#
#    noise_cal[noise_cal == 0] = np.inf
#    vis = vis/noise_cal
#    vis[cal_on, ...] -= 1.
#
#    return vis, cal_on
