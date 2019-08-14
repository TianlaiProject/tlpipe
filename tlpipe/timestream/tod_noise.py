#import matplotlib.pyplot as plt
import numpy as np
import gc
import timestream_task
import h5py
from astropy.time import Time
from tlpipe.utils.path_util import output_path
from caput import mpiutil
from caput import mpiarray
from scipy.signal import medfilt
from scipy.signal import lombscargle

class DataEdit(timestream_task.TimestreamTask):
    """
    Edit the Tod data observed by MeerKAT for 1/f noise analysis
    """

    params_init = {
            'bad_time_list' : None,
            'bad_freq_list' : None,
            'bandpass_cal'  : True,
            'fill_noise_diode' : False,
            'noise_cal_init_time'  : None,
            'noise_cal_period' : 19.9915424299, # in unit of second
            'noise_cal_length' : 1.8, # in unit of second
            'noise_cal_ext' : [0, 0], # in unit of int time, extending cal_on
            'noise_cal_delayed_ant' : [],
            }

    prefix = 'pned_'

    def process(self, ts):

        print ts.vis.shape
        bad_time_list = self.params['bad_time_list']
        bad_freq_list = self.params['bad_freq_list']

        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']

        if bad_time_list is not None:
            num_infiles = len(self.input_files)
            name = ts.main_data_name
            outfiles_map = ts._get_output_info(name, num_infiles)[-1]
            st = 0
            for fi, start, stop in outfiles_map:
                et = st + (stop - start)
                print "Mask bad time"
                for bad_time in bad_time_list:
                    print bad_time
                    ts.vis_mask[st:et, ...][slice(*bad_time), ...] = True
                st = et

        if bad_freq_list is not None:
            print "Mask bad freq"
            for bad_freq in bad_freq_list:
                print bad_freq
                ts.vis_mask[:, slice(*bad_freq), ...] = True

        old_dtype = ts.vis.dtype
        if self.params['corr'] == 'auto' and old_dtype == np.complex64:
            vis_abs = mpiarray.MPIArray.wrap(np.abs(ts.vis[:]), 0)
            ts.create_main_data(vis_abs, recreate=True, copy_attrs=True)
            new_dtype = ts.vis.dtype
            print "Use Auto only, Convert from %s to %s by abs"%(old_dtype, new_dtype)

        if self.params['noise_cal_init_time'] is not None:
            noise_on = np.zeros((ts.vis.shape[0], ts.vis.shape[3]), dtype='bool')
            noise_on = mpiarray.MPIArray.wrap(noise_on, axis=0)
            ts.create_dataset('ns_on', data=noise_on, dtype=noise_on.dtype)
            ts['ns_on'].attrs['dimname'] = 'Time, Baseline'

            #noise_cal_delayed = False
            #if bl[0] - 1 in self.params['noise_cal_delayed_ant']:
            #    print "Cal delayed, t0 plus 1 s"
            #    noise_cal_delayed = True

            #noise_on = get_cal_stamps(self.params['noise_cal_init_time'],
            #                          self.params['noise_cal_period'],
            #                          self.params['noise_cal_length'],
            #                          ts['sec1970'][:],
            #                          self.params['noise_cal_ext'],
            #                          noise_cal_delayed)
            #noise_on = mpiarray.MPIArray.wrap(noise_on, axis=0)
            #ts.create_time_ordered_dataset('ns_on', data=noise_on, recreate=True)
            #ts['ns_on'].attrs['dimname'] = 'Time'

        func = ts.bl_data_operate
        func(self.data_edit, full_data=True, copy_data=False, 
                show_progress=show_progress, 
                progress_step=progress_step, keep_dist_axis=False)

        #print ts.vis.shape
        #print np.var(ts.vis[:, :, 0, 0], axis=0)

        return super(DataEdit, self).process(ts)

    def data_edit(self, vis, vis_mask, li, gi, bl, ts, **kwargs):

        if vis.dtype == np.complex or vis.dtype == np.complex64:
            vis_abs = np.abs(vis)
        else:
            vis_abs = vis

        if self.params['noise_cal_init_time'] is not None:
            noise_cal_delayed = False
            print "Mask Noise Diode for Ant. M%03d"%(bl[0] - 1)
            if bl[0] - 1 in self.params['noise_cal_delayed_ant']:
                print "Cal delayed, t0 plus 1 s"
                noise_cal_delayed = True

            noise_on = get_cal_stamps(self.params['noise_cal_init_time'],
                                      self.params['noise_cal_period'],
                                      self.params['noise_cal_length'],
                                      ts['sec1970'][:],
                                      self.params['noise_cal_ext'],
                                      noise_cal_delayed)
            ts['ns_on'][:, gi] = noise_on


        bad_time = np.all(vis_mask, axis=(1, 2))
        if 'ns_on' in ts.iterkeys() :
            bad_time = bad_time + ts['ns_on'][:, gi]
        bad_freq = np.all(vis_mask, axis=(0, 2))
        good = (~bad_time)[:, None] * (~bad_freq)[None, :]
        good = good[:, :, None] * np.ones_like(vis_abs).astype('bool')

        bandpass_cal = self.params['bandpass_cal']
        if bandpass_cal:
            "Bandpass cal with time median value"
            bandpass = np.median(vis_abs[tuple(~bad_time), ...], axis=0)
            bandpass[:,0] = medfilt(bandpass[:,0], kernel_size=11)
            bandpass[:,1] = medfilt(bandpass[:,1], kernel_size=11)
            bandpass[bandpass==0] = np.inf
            vis_abs /= bandpass[None, ...]

        if self.params['fill_noise_diode']:
            if 'ns_on' in ts.iterkeys() :
                print 'Fill noise diode on time with nearby time'
                noise_on = ts['ns_on'][:, gi]
                vis_abs = fill_noise_diode_nearby(vis_abs, noise_on)
            else:
                print "No noise diode info. not able to fill"


        if vis.dtype == np.complex or vis.dtype == np.complex64:
            vis.real = vis_abs
            vis.imag = 0.
        else:
            vis = vis_abs

def fill_noise_diode_nearby(vis, noise_on):


    # take one time stamp ahead and one after the flagged time stamps 
    # as noise diode off time
    noise_off = (np.roll(noise_on, 1)  + np.roll(noise_on, -1)) ^ noise_on


    # pick the noise off time value
    vis_fillvalue = vis[noise_off, ...]

    # if the first two time stamps are both noise diode on,
    # vis_fillvalue will pick the last time stamp value. 
    # we fix it by rolling vis_fillvalue one stamp, and fill the first
    # value equal to the second. 
    if noise_on[0] and noise_on[1]:
        vis_fillvalue[-1, ...] = vis_fillvalue[0, ...]
        vis_fillvalue = np.roll(vis_fillvalue, 1, axis=0)
    # fill the noise_on with noise_off values
    vis[noise_on, ...] = vis_fillvalue

    return vis

def get_cal_stamps(t0, p, l, time, ext=[0, 0], noise_cal_delayed=False):

    t0 = Time(t0).unix #+ 5.
    if noise_cal_delayed:
        t0 += 1

    dt = time[1] - time[0]
    t_bins = np.append(time, time[-1] + dt)
    t_bins -= 0.5 * dt

    #noise_st = np.arange(t0,     t_bins[-1], p * dt)
    #noise_ed = np.arange(t0 + l, t_bins[-1], p * dt)
    noise_st = np.arange(t0,     t_bins[-1], p)
    noise_ed = np.arange(t0 + l, t_bins[-1], p)

    noise_on  = np.histogram(noise_st, t_bins)[0]
    noise_on += np.histogram(noise_ed, t_bins)[0]
    noise_on  = noise_on.astype('bool')

    noise_on_p = noise_on.copy()
    if ext[0] != 0:
        for ii in range(ext[0]):
            noise_on_p += np.roll(noise_on.copy(), ii + 1)

    noise_on_n = noise_on.copy()
    if ext[1] != 0:
        for ii in range(ext[0]):
            noise_on_n += np.roll(noise_on.copy(), -(ii + 1))

    noise_on = noise_on_p + noise_on_n

    return noise_on


class PinkNoisePS(timestream_task.TimestreamTask):
    """
    Estimate 1/f noise power spectrum
    """

    params_init = {
            'data_sets' : 'cleaned_vis', # cleaned_vis, vis, svdmodes
            #'n_bins'    : 20,
            #'f_min'     : None,
            #'f_max'     : None,
            'avg_len'   : 100,
            'method'    : 'fft', # lombscargle
            'fill_noise_diode' : False,
            }

    prefix = 'pnps_'

    def process(self, ts):

        data_sets = self.params['data_sets']

        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']

        func = ts.bl_data_operate
        if data_sets == 'vis':
            func(self.est_ps, full_data=True, copy_data=True, 
                show_progress=show_progress, 
                progress_step=progress_step, keep_dist_axis=False)
            #tcorr_ps_list = np.array(self.tcorr_ps_list)
            #tcorr_bc_list = np.array(self.tcorr_bc_list)
            #label_list    = np.array(self.label_list)
        elif data_sets == 'cleaned_vis':
            mode_list = ts['mode_list'][:]
            for m in mode_list:
                ts.main_data_name = 'vis_sub%02dmodes'%m
                func(self.est_ps, full_data=True, copy_data=True, 
                        show_progress=show_progress, 
                        progress_step=progress_step, keep_dist_axis=False)
        elif data_sets == 'modes':
            mode_list = ts['mode_list'][:]
            for m in np.arange(mode_list.max()):
                ts.main_data_name = 'modes%02d'%(m + 1)
                func(self.est_ps, full_data=True, copy_data=True, 
                        show_progress=show_progress, 
                        progress_step=progress_step, keep_dist_axis=False)

        del ts
        gc.collect()

        return None

        #return super(PinkNoisePS, self).process(ts)

    def est_ps(self, vis, vis_mask, li, gi, bl, ts, **kwargs):

        #print vis.dtype
        #vis = np.abs(vis)
        vis = vis.copy()

        method = self.params['method']

        bad_time = np.all(vis_mask, axis=(1, 2))

        if 'ns_on' in ts.iterkeys():
            if self.params['fill_noise_diode']:
                print 'Fill noise diode with nearby time stamps'
                noise_on = ts['ns_on'][:, gi]
                vis = fill_noise_diode_nearby(vis, noise_on)
            else:
                print "Mask noise diode as bad time"
                bad_time = bad_time + ts['ns_on'][:, gi]

        bad_freq = np.all(vis_mask, axis=(0, 2))
        #print np.all(bad_freq)
        #print np.all(bad_time)
        #self.bad_time = bad_time
        self.bad_freq = bad_freq

        num_infiles = len(self.input_files)
        name = ts.main_data_name
        outfiles_map = ts._get_output_info(name, num_infiles)[-1]
        st = 0
        for fi, start, stop in outfiles_map:
            et = st + (stop - start)

            name = ts.main_data_name

            self.bad_time = bad_time[st:et]

            _vis = vis[st:et, ...].copy()
            #_vis = _vis[~bad_time[st:et], ...][:, ~bad_freq, ...]
            #print np.mean(_vis, axis=0)
            #_vis = _vis - np.mean(_vis, axis=0)[None, :, :]
            time = ts['sec1970'][st:et]
            #time = time[~bad_time[st:et]]
            freq = ts['freq'][:]
            #freq = freq[~bad_freq]

            #print "Normalize with dt %f df %f"%((time[1]-time[0]), (freq[1] - freq[0]))
            #_vis = _vis / (time[1]-time[0]) / (freq[1] - freq[0])

            _vis, name, freq, time = self.avg_vis(_vis, name, freq, time)

            file_name = None
            if len(self.output_files) == num_infiles:
                output_file = self.output_files[fi]
                file_suffix = '_%s_tcorrps_%s_m%03d_x_m%03d.h5'%(name, method, bl[0]-1, bl[1]-1)
                file_name = output_file.replace('.h5', file_suffix)
                file_name = output_path(file_name, relative= not file_name.startswith('/'))
            self.ps_process_write(_vis, time, freq, ts, file_name)

            del _vis
            gc.collect()

            st = et

    def avg_vis(self, vis, name, freq, time):

        return vis, name, freq, time

    def ps_process_write(self, vis, time, freq, ts):

        pass

class PinkNoisePS_1DTC(PinkNoisePS):

    params_init = {
            'n_bins'    : 20,
            'f_min'     : None,
            'f_max'     : None,
            }

    prefix = 'pnps1dtc_'

    def avg_vis(self, vis, name, freq, time):

        freq = freq[~self.bad_freq]
        vis  = vis[:, ~self.bad_freq, ...]

        avg_len = self.params['avg_len']

        if avg_len == 0:
            print "average all frequencies before ps. est."
            vis = np.mean(vis, axis=1)
            vis = vis[:, None, :]
            name += '_avgEachF'
            freq = None
        elif avg_len is None:
            print "no frequency average before ps. est."
            name += '_avgNoneF'
        else:
            print "average every %d frequencis before ps. est."%avg_len
            time_n, freq_n, pol_n = vis.shape
            split_n = freq_n / avg_len
            if split_n == 0:
                msg = "Only %d freq channels, avg_len should be less than it"%freq_n
                raise ValueError(msg)
            print "%d/%d freq channels are using"%(split_n * avg_len, freq_n)
            vis = vis[:, :split_n * avg_len, :]
            vis.shape = (time_n, split_n, avg_len, pol_n)
            vis = np.mean(vis, axis=2)
            name += '_avg%04dF'%avg_len
        
            freq = freq[:split_n * avg_len]
            freq.shape = (split_n, avg_len)
            freq = np.median(freq, axis=1)

        return vis, name, freq, time


    def ps_process_write(self, vis, time, freq, ts, file_name = None):

        method = self.params['method']
        n_bins = self.params['n_bins']
        f_min  = self.params['f_min']
        f_max  = self.params['f_max']
        inttime = ts.attrs['inttime']

        if method == 'fft':
            psd_func = est_tcorr_psd1d_fft
        elif method == 'lombscargle':
            psd_func = est_tcorr_psd1d_lombscargle
        tcorr_ps, tcorr_bc = psd_func(vis, time, self.bad_time,
                n_bins=n_bins, f_min=f_min, f_max=f_max, inttime=inttime)

        if file_name is not None:
            #output_files = output_path(self.output_files[fi], relative=False,
            #        iteration=self.iteration)
            #output_file = self.output_files[fi]
            #file_suffix = '_%s_tcorrps_%s_m%03d_x_m%03d.h5'%(name, method, bl[0]-1, bl[1]-1)
            #file_name = output_file.replace('.h5', file_suffix)
            #file_name = output_path(file_name, relative= not file_name.startswith('/'))
            print file_name
            with h5py.File(file_name, 'w') as f:
                f['tcorr_ps'] = tcorr_ps
                f['tcorr_bc'] = tcorr_bc
                if self.bad_freq is not None:
                    f['bad_freq'] = self.bad_freq
                if self.bad_time is not None:
                    f['bad_time'] = self.bad_time
            print '-' * 20
            print

class PinkNoisePS_1DFC(PinkNoisePS):

    params_init = {
            'n_bins'    : 20,
            'w_min'     : None,
            'w_max'     : None,
            }

    prefix = 'pnps1dfc_'

    def avg_vis(self, vis, name, freq, time):

        time = time[~self.bad_time]
        vis  = vis[~self.bad_time, ...]

        avg_len = self.params['avg_len']

        if avg_len == 0:
            print "average all time before ps. est."
            vis = np.mean(vis, axis=0)
            vis = vis[None, :, :]
            name += '_avgEachT'
            time = None
        elif avg_len is None:
            print "no time average before ps. est."
            name += '_avgNoneT'
        else:
            print "average every %d time before ps. est."%avg_len
            time_n, freq_n, pol_n = vis.shape
            split_n = time_n / avg_len
            if split_n == 0:
                msg = "Only %d time steps, avg_len should be less than it"%time_n
                raise ValueError(msg)
            print "%d/%d time steps  are using"%(split_n * avg_len, time_n)
            vis = vis[:split_n * avg_len, :, :]
            vis.shape = (split_n, avg_len, freq_n, pol_n)
            vis = np.mean(vis, axis=1)
            name += '_avg%04dT'%avg_len
        
            time = time[:split_n * avg_len]
            time.shape = (split_n, avg_len)
            time = np.median(time, axis=1)

        return vis, name, freq, time


    def ps_process_write(self, vis, time, freq, ts, file_name = None):

        method = self.params['method']
        n_bins = self.params['n_bins']
        w_min  = self.params['w_min']
        w_max  = self.params['w_max']
        df     = ts.attrs['freqstep']

        if method == 'fft':
            psd_func = est_tcorr_psd1d_fft
        elif method == 'lombscargle':
            psd_func = est_tcorr_psd1d_lombscargle
        vis = np.swapaxes(vis, 0, 1)
        tcorr_ps, tcorr_bc = psd_func(vis, freq, self.bad_freq,
                n_bins=n_bins, f_min=w_min, f_max=w_max, inttime=df)

        if file_name is not None:
            #output_files = output_path(self.output_files[fi], relative=False,
            #        iteration=self.iteration)
            #output_file = self.output_files[fi]
            #file_suffix = '_%s_tcorrps_%s_m%03d_x_m%03d.h5'%(name, method, bl[0]-1, bl[1]-1)
            #file_name = output_file.replace('.h5', file_suffix)
            #file_name = output_path(file_name, relative= not file_name.startswith('/'))
            print file_name
            with h5py.File(file_name, 'w') as f:
                f['tcorr_ps'] = tcorr_ps
                f['tcorr_bc'] = tcorr_bc
                if self.bad_freq is not None:
                    f['bad_freq'] = self.bad_freq
                if self.bad_time is not None:
                    f['bad_time'] = self.bad_time

            print '-' * 20
            print

class PinkNoiseCF_1DTC(PinkNoisePS):

    params_init = {
            'n_bins'    : 20,
            't_min'     : None,
            't_max'     : None,
            'linear_bins': False,
            }

    prefix = 'pntf1dtc_'

    def avg_vis(self, vis, name, freq, time):

        # for time corr, we average over frequencies.
        # just pickup the good frequency channels.
        freq = freq[~self.bad_freq]
        vis  = vis[:, ~self.bad_freq]

        avg_len = self.params['avg_len']

        if avg_len == 0:
            print "average all frequencies before ps. est."
            vis = np.mean(vis, axis=1)
            vis = vis[:, None, :]
            name += '_avgEachF'
            freq = None
        elif avg_len is None:
            print "no frequency average before ps. est."
            name += '_avgNoneF'
        else:
            print "average every %d frequencis before ps. est."%avg_len
            time_n, freq_n, pol_n = vis.shape
            split_n = freq_n / avg_len
            if split_n == 0:
                msg = "Only %d freq channels, avg_len should be less than it"%freq_n
                raise ValueError(msg)
            print "%d/%d freq channels are using"%(split_n * avg_len, freq_n)
            vis = vis[:, :split_n * avg_len, :]
            vis.shape = (time_n, split_n, avg_len, pol_n)
            vis = np.mean(vis, axis=2)
            name += '_avg%04dF'%avg_len
        
            freq = freq[:split_n * avg_len]
            freq.shape = (split_n, avg_len)
            freq = np.median(freq, axis=1)

        return vis, name, freq, time


    def ps_process_write(self, vis, time, freq, ts, file_name = None):

        n_bins = self.params['n_bins']
        t_min  = self.params['t_min']
        t_max  = self.params['t_max']
        linear_bins =self.params['linear_bins']
        inttime = ts.attrs['inttime']

        psd_func = est_tcorr_cor1d
        tcorr_cf, tcorr_bc = psd_func(vis, time, self.bad_time,
                n_bins=n_bins, linear_bins=linear_bins, 
                t_min=t_min, t_max=t_max, inttime=inttime)

        if file_name is not None:
            print file_name
            with h5py.File(file_name, 'w') as f:
                f['tcorr_ps'] = tcorr_cf
                f['tcorr_bc'] = tcorr_bc
                if self.bad_freq is not None:
                    f['bad_freq'] = self.bad_freq
                if self.bad_time is not None:
                    f['bad_time'] = self.bad_time
            print '-' * 20
            print

def est_tcorr_psd1d_fft(data, ax, flag, n_bins=None, inttime=None,
        f_min=None, f_max=None):

    data = data.copy()

    mean = np.mean(data[~flag, ...], axis=0)
    data -= mean[None, :, :]
    data[flag, ...] = 0.

    #windowf = np.blackman(data.shape[0])
    #data *= windowf[:, None, None]

    fftdata = np.fft.fft(data, axis=0, norm='ortho')

    n = ax.shape[0]
    if inttime is None:
        d = ax[1] - ax[0]
    else:
        d = inttime
    freq = np.fft.fftfreq(n, d) #* 2 * np.pi 

    freq_p    = freq[freq>0]
    fftdata_p = fftdata[freq>0, ...]
    fftdata_p = np.abs(fftdata_p) * np.sqrt(float(d))
    fftdata_p = fftdata_p ** 2.

    if n_bins is not None:

        #if avg:
        #    fftdata_p = np.mean(fftdata_p, axis=1)[:, None, :]

        fftdata_bins = np.zeros((n_bins, ) + fftdata_p.shape[1:])

        if f_min is None: f_min = freq_p.min()
        if f_max is None: f_max = freq_p.max()
        freq_bins_c = np.logspace(np.log10(f_min), np.log10(f_max), n_bins)
        freq_bins_d = freq_bins_c[1] / freq_bins_c[0]
        freq_bins_e = freq_bins_c / (freq_bins_d ** 0.5)
        freq_bins_e = np.append(freq_bins_e, freq_bins_e[-1] * freq_bins_d)
        norm = np.histogram(freq_p, bins=freq_bins_e)[0] * 1.
        norm[norm==0] = np.inf

        for i in range(fftdata_p.shape[1]):

            hist_0 = np.histogram(freq_p, bins=freq_bins_e, weights=fftdata_p[:,i,0])[0]
            hist_1 = np.histogram(freq_p, bins=freq_bins_e, weights=fftdata_p[:,i,1])[0]
            hist   = np.concatenate([hist_0[:, None], hist_1[:, None]],axis=1)
            fftdata_bins[:, i, :] = hist / norm[:, None]

        fftdata_bins[freq_bins_c <= freq_p.min()] = 0.
        fftdata_bins[freq_bins_c >= freq_p.max()] = 0.

        return fftdata_bins, freq_bins_c
    else: 
        return fftdata_p, freq_p


def est_tcorr_psd1d_lombscargle(data, ax, flag, n_bins=None, inttime=None, 
        f_min=None, f_max=None):

    #fft_len = data.shape[0]

    #data = np.ma.array(data)
    #data.mask = np.zeros(data.shape)
    #data.mask[flag, ...] = 1
    #data_mean = np.ma.mean(data, axis=0)
    #data_std  = np.ma.std(data, axis=0)
    #print data_mean
    #print data_std
    #data_rand = np.random.standard_normal(data.shape) * data_std + data_mean
    ##data = np.ones_like(data)
    #data = data.data
    #data[flag, ...] = data_rand[flag, ...]
    data = data[~flag, ...]
    ax   = ax[~flag, ...]

    if inttime is None:
        inttime = ax[1] - ax[0]

    #n = ax.shape[0]
    d = inttime
    n = int((ax[-1] - ax[0]) / inttime)
    fft_len = n
    fftfreq = np.fft.fftfreq(n, d) #* 2 * np.pi 
    fftfreq = fftfreq[fftfreq>0]

    print 'int time', inttime

    #windowf = np.blackman(fft_len)

    if n_bins is None: n_bins = 30
    #if f_min  is None: f_min = 1. / ( ax.max()     - ax.min() )
    #if f_max  is None: f_max = 1. / ( np.abs(ax[1] - ax[0]) ) / 2.
    if f_min  is None: f_min = 1. / ( inttime * fft_len )
    if f_max  is None: f_max = 1. / inttime / 2.

    freq_bins_c = np.logspace(np.log10(f_min), np.log10(f_max), n_bins)
    #freq_bins_c = np.linspace(f_min, f_max, n_bins)
    freq_bins_d = freq_bins_c[1] / freq_bins_c[0]
    freq_bins_e = freq_bins_c / (freq_bins_d ** 0.5)
    freq_bins_e = np.append(freq_bins_e, freq_bins_e[-1] * freq_bins_d)

    #fftfreq = freq_bins_c

    #freqs = np.linspace(f_min, f_max, ax.shape[0])
    #freqs = np.linspace(f_min, f_max, 1024)

    n_freq, n_pol = data.shape[1:]

    power = np.zeros([n_bins, n_freq, n_pol])

    norm = np.histogram(fftfreq, bins=freq_bins_e)[0] * 1.
    norm[norm==0] = np.inf
    for i in range(n_freq):
        for j in range(n_pol):
            y = data[:, i, j]
            #y = y - np.mean(y)
            #y = y * windowf
            #power[:, i, j] = lombscargle(ax, y, 2. * np.pi * freq_bins_c,
            #        normalize=False)
            _p = lombscargle(ax, y, 2. * np.pi * fftfreq, normalize=False)
            hist   = np.histogram(fftfreq, bins=freq_bins_e, weights=_p)[0]
            power[:, i, j] = hist / norm
    #power = np.sqrt(4. * power / float(ax.shape[0]) / np.std(y) ** 2.)
    power *= inttime

    power[freq_bins_c <= 1. / ( inttime * fft_len ) ] = 0.
    power[freq_bins_c >= 1. / inttime / 2.] = 0.

    return power, freq_bins_c

def est_tcorr_cor1d(data, ax, n_bins, flag, linear_bins=False, 
        inttime=None, t_min=None, t_max=None):

    ax = ax - ax[0]
    ax = np.float32(ax)

    fft_len = data.shape[0]

    if inttime is None:
        inttime = ax[1] - ax[0]

    n = ax.shape[0]
    d = inttime

    if n_bins is None: n_bins = 30
    if t_min  is None: t_min =  inttime
    if t_max  is None: t_max =  inttime * fft_len / 2.

    if linear_bins:
        tau_bins_c = np.linspace(t_min, t_max, n_bins)
        tau_bins_d = tau_bins_c[1] - tau_bins_c[0]
        tau_bins_e = tau_bins_c - (tau_bins_d * 0.5)
        tau_bins_e = np.append(tau_bins_e, tau_bins_e[-1] + tau_bins_d)
    else:
        tau_bins_c = np.logspace(np.log10(t_min), np.log10(t_max), n_bins)
        tau_bins_d = tau_bins_c[1] / tau_bins_c[0]
        tau_bins_e = tau_bins_c / (tau_bins_d ** 0.5)
        tau_bins_e = np.append(tau_bins_e, tau_bins_e[-1] * tau_bins_d)

    n_freq, n_pol = data.shape[1:]

    power = np.zeros([n_bins, n_freq, n_pol])

    timelag = (ax[:, None] - ax[None, :]).flatten()
    timelag = np.abs(timelag)

    norm = np.histogram(timelag, bins=tau_bins_e)[0] * 1.
    norm[norm==0] = np.inf
    #plt.plot(tau_bins_e[:-1], norm)
    #plt.show()


    for i in range(n_freq):
        for j in range(n_pol):
            y = data[:, i, j]
            #y = y - np.mean(y)
            #y = y.astype('float32')
            _p = (y[:, None] * y[None, :]).flatten()
            #print _p.size * _p.itemsize / ((1024.)**2)
            hist  = np.histogram(timelag, bins=tau_bins_e, weights=_p)[0]
            power[:, i, j] = hist / norm
            del _p
            gc.collect()
    #power = np.sqrt(4. * power / float(ax.shape[0]) / np.std(y) ** 2.)
    #power *= inttime

    power[tau_bins_c <= inttime ] = 0.
    power[tau_bins_c >= inttime * fft_len / 2.] = 0.

    return power, tau_bins_c
