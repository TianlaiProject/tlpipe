import numpy as np
import gc
from tlpipe.timestream import timestream_task
import h5py
from astropy.time import Time
from tlpipe.utils.path_util import output_path
from caput import mpiutil
from caput import mpiarray
from scipy.signal import medfilt
from scipy.signal import lombscargle

class Bandpass_Cal(timestream_task.TimestreamTask):
    """
    Edit the Tod data observed by MeerKAT for 1/f noise analysis
    """

    params_init = {
            'noise_on_time': 2,
            'bandpass_smooth' : 51,
            }

    prefix = 'bpcal_'

    def process(self, ts):

        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']


        func = ts.bl_data_operate
        func(self.cal_data, full_data=True, copy_data=False, 
                show_progress=show_progress, 
                progress_step=progress_step, keep_dist_axis=False)

        return super(Bandpass_Cal, self).process(ts)

    def cal_data(self, vis, vis_mask, li, gi, bl, ts, **kwargs):

        on_t        = self.params['noise_on_time']
        kernel_size = self.params['bandpass_smooth']

        vis1 = np.ma.array(vis.copy())
        vis1.mask = vis_mask

        if 'ns_on' in ts.iterkeys():
            print 'Uisng Noise Diode Mask for Ant. %03d'%(bl[0] - 1)
            if len(ts['ns_on'].shape) == 2:
                on = ts['ns_on'][:, gi].astype('bool')
            else:
                on = ts['ns_on'][:]
            #on = ts['ns_on'][:]
            vis1.mask[~on, ...] = True
        else:
            print "No Noise Diode Mask info"

        vis1, on = get_Ncal(vis1, vis_mask, on, on_t)

        # take the median value of each channel as the bandpass
        bandpass = np.median(vis1, axis=0)
        # smooth the bandpass to remove some RFI
        bandpass[:,0] = medfilt(bandpass[:,0], kernel_size=kernel_size)
        bandpass[:,1] = medfilt(bandpass[:,1], kernel_size=kernel_size)
        bandpass[bandpass==0] = np.inf

        vis /= bandpass[None, ...]

        # get the time var
        time  = ts['sec1970'][:]
        time -= time[0]
        time_on = time[on]
        vis1 /= np.ma.median(vis1, axis=(0,1))[None, None, :]
        vis1 = np.ma.median(vis1, axis=1)
        good = ~vis1.mask
        vis1_poly_xx = np.poly1d(np.polyfit(time_on[good[:,0]], vis1[:, 0][good[:,0]], 4))
        vis1_poly_yy = np.poly1d(np.polyfit(time_on[good[:,1]], vis1[:, 1][good[:,1]], 4))

        vis[..., 0] /= vis1_poly_xx(time)[:, None]
        vis[..., 1] /= vis1_poly_yy(time)[:, None]




def get_Ncal(vis, vis_mask, on, on_t):

    # remove the cal at the beginning/ending
    on[ :on_t] = False
    on[-on_t:] = False
    if on_t == 2:
        # noise cal may have half missing, because of the RFI flagging
        # remove them
        on  = (np.roll(on, 1) * on) + (np.roll(on, -1) * on)
        # use one time stamp before, one after as cal off
        off = (np.roll(on, 1) + np.roll(on, -1)) ^ on
    elif on_t == 1:
        off = np.roll(on, 1)
    else:
        raise
    
    #vis1 = vis1.data
    vis1_on  = vis[on, ...]
    vis1_off = vis[off, ...].data
    vis1 = vis1_on - vis1_off

    if on_t > 1:
        vis_shp = vis1.shape
        vis1 = vis1.reshape((-1, on_t) + vis_shp[1:])
        vis1 = vis1 + vis1[:, ::-1, ...]
        vis1.shape = vis_shp

    return vis1, on

