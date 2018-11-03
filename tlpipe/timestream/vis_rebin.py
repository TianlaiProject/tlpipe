"""Rebin the frequency channels.

Inheritance diagram
-------------------

.. inheritance-diagram:: Rebin
   :parts: 2

"""

import warnings
import numpy as np
import timestream_task
from tlpipe.container.timestream import Timestream
from caput import mpiutil
from caput import mpiarray
from tlpipe.utils.np_util import unique, average


class Rebin(timestream_task.TimestreamTask):
    """Rebin the frequency channels.

    This task rebins the data along the frequency by merging (and average)
    the adjacent frequency channels.

    """

    params_init = {
                    'freq_bin': 2,
                    'time_bin': 2,
                  }

    prefix = 'rb_'

    def process(self, ts):

        assert isinstance(ts, Timestream), '%s only works for Timestream object' % self.__class__.__name__

        MASKNOAVE = 32
        print("vis_rebin")
        freq_bin = self.params['freq_bin']
        time_bin = self.params['time_bin']
        print("bin_numbers",freq_bin,time_bin)
        ts.redistribute('baseline')

        ntin = len(ts.time)
        nfin = len(ts.freq)
        ntout = ntin/time_bin
        nfout = nfin/freq_bin
        
#            warnings.warn('The number of bins can not exceed the number of frequencies, do nothing')
        freq = np.zeros(nfout, dtype=ts.freq.dtype)
        for nf in xrange(nfout):
            bf = nf*freq_bin
            ef = bf + freq_bin
            freq[nf] = np.mean(ts.freq[bf:ef])

        sec1970_in = ts['sec1970']
        sec1970 = np.zeros(ntout, dtype=sec1970_in.dtype)
        jul_date_in = ts['jul_date']
        jul_date = np.zeros(ntout, dtype=jul_date_in.dtype)
        local_hour_in = ts['local_hour']
        local_hour = np.zeros(ntout, dtype=local_hour_in.dtype)
        az_alt_in = ts['az_alt']
        az_alt = np.zeros(ntout, dtype=az_alt_in.dtype)
        ra_dec_in = ts['ra_dec']
        ra_dec = np.zeros(ntout, dtype=ra_dec_in.dtype)
        ns_on_in = ts['ns_on']
        ns_on = np.zeros(ntout, dtype=ns_on_in.dtype)
        print("sec1970",sec1970.dtype,sec1970[0])


        vis = np.zeros((ntout,nfout)+ts.local_vis.shape[2:], dtype=ts.vis.dtype)
        vis_mask= np.zeros((ntout,nfout)+ts.local_vis.shape[2:], dtype=ts.vis_mask.dtype) # all False
        print("shapes",freq.shape,vis.shape,vis_mask.shape)
        
        # average over frequency
        for nt in xrange(ntout):
            bt = nt*time_bin
            et = bt + time_bin
            sec1970[nt] = np.mean(sec1970_in[bt:et])
            jul_date[nt] = np.mean(jul_date_in[bt:et])
            local_hour[nt] = np.mean(local_hour_in[bt:et])
            az_alt[nt] = np.mean(az_alt_in[bt:et])
            ra_dec[nt] = np.mean(ra_dec_in[bt:et])
            ns_on[nt] = np.any(ns_on_in[bt:et])
            for nf in xrange(nfout):
                bf = nf*freq_bin
                ef = bf + freq_bin
                #print(bt,et,bf,ef)
                # rebin freq
                #freq[idx] = average(ts.freq[inds], axis=0, weights=weight)
                # rebin vis
                masked_vis = np.ma.array(ts.local_vis[bt:et,bf:ef], mask=ts.local_vis_mask[bt:et,bf:ef])
                #print("masked_vis.shape",masked_vis.shape)
                vis[nt,nf] = np.mean(masked_vis,axis=(0,1)) 
                good = np.isfinite(vis[nt,nf])
                vis_mask[nt,nf] = np.where(good,0,MASKNOAVE)
                # rebin vis_mask
#                cmask = np.full(vis.shape,-1,dtype=vis_mask.dtype)
#                for i in inds:
#                    cmask[:] &= ts.local_vis_mask[:,i]
#                vis_mask[nt,nf] = np.where(valid_cnt==0,MASKNOAVE|cmask,0)

            # create rebinned datasets
#
#add sec1970,jul_date,local_hour
#
        vis = mpiarray.MPIArray.wrap(vis, axis=3)
        vis_mask= mpiarray.MPIArray.wrap(vis_mask, axis=3)
        ts.create_main_data(vis, recreate=True, copy_attrs=True)
        axis_order = ts.main_axes_ordered_datasets['vis']
        ts.create_main_axis_ordered_dataset(axis_order, 'vis_mask', vis_mask, axis_order, recreate=True, copy_attrs=True)
        ts.create_freq_ordered_dataset('freq', freq, recreate=True, copy_attrs=True, check_align=True)
        ts.create_time_ordered_dataset('sec1970',sec1970,recreate=True,copy_attrs=True)
        ts.create_time_ordered_dataset('jul_date',jul_date,recreate=True,copy_attrs=True)
        ts.create_time_ordered_dataset('local_hour',local_hour,recreate=True,copy_attrs=True)
        ts.create_time_ordered_dataset('az_alt',az_alt,recreate=True,copy_attrs=True)
        ts.create_time_ordered_dataset('ra_dec',ra_dec,recreate=True,copy_attrs=True)
        ts.create_time_ordered_dataset('ns_on',ns_on,recreate=True,copy_attrs=True)

# properties
#        print("ts.attrs=",ts.attrs)
# main_data
#        print("main_data=",ts.main_data)
# main_data_name
#        print("main_data_name=",ts.main_data_name)
# main_data_axes
#        print("main_data_axes=",ts.main_data_axes)
# dist_axis
#        print("dist_axis=",ts.dist_axis)
# dist_axis_name
# main_axes_ordered_datasets
#        print("main_axes_ordered_datasets=",ts.main_axes_ordered_datasets)
# main_time_ordered_datasets
#        print("main_time_ordered_datasets=",ts.main_time_ordered_datasets)
# time_ordered_datasets
#        print("time_ordered_datasets=",ts.time_ordered_datasets)
#        print("freq_ordered_datasets=",ts.freq_ordered_datasets)
# time_ordered_attrs
#        print("time_ordered_datasets=",ts.time_ordered_datasets)

#    def create_time_ordered_dataset(self, name, data, axis_order=(0,), recreate=False, copy_attrs=False):
#    def create_main_axis_ordered_dataset(self, axis, name, data, axis_order, recreate=False, copy_attrs=False, check_align=True):

#    def create_main_data(self, data, recreate=False, copy_attrs=False):

#    def create_main_time_ordered_dataset(self, name, data, axis_order=(0,), recreate=False, copy_attrs=False, check_align=True):

            # for other freq_axis datasets
        for name in ts.freq_ordered_datasets.keys():
            if name in ts.iterkeys() and not name in ('freq', 'vis', 'vis_mask'): # exclude already rebinned datasets
                raise RuntimeError('Should not have other freq_ordered_datasets %s' % name)
#Also nfreq
        ts.attrs['nfreq'] = nfout
#Also inttime
        ts.attrs['inttime'] *= time_bin
        # update freqstep attr
        ts.attrs['freqstep'] *= freq_bin
        period = ts['ns_on'].attrs['period']
        ts['ns_on'].attrs['period'] = np.float32(period)/time_bin
        on_time = ts['ns_on'].attrs['on_time']
        ts['ns_on'].attrs['on_time'] = np.float32(on_time)/time_bin
        off_time = ts['ns_on'].attrs['off_time']
        ts['ns_on'].attrs['off_time'] = np.float32(off_time)/time_bin
        return super(Rebin, self).process(ts)
