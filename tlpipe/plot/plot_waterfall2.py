"""Plot waterfall images.

Inheritance diagram
-------------------

.. inheritance-diagram:: Plot
   :parts: 2

"""

# import pytz
from datetime import datetime
import time
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from tlpipe.timestream import timestream_task
from tlpipe.container.raw_timestream import RawTimestream
from tlpipe.container.timestream import Timestream
from tlpipe.utils.path_util import output_path
from tlpipe.utils import hist_eq
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from tlpipe.utils import date_util
# tz = pytz.timezone('Asia/Shanghai')

class Plot(timestream_task.TimestreamTask):
    """Waterfall plot for Timestream.

    This task plots the waterfall (i.e., visibility as a function of time
    and frequency) of the visibility
    for each baseline (and also each polarization if the input data is a
    :class:`~tlpipe.container.timestream.Timestream` instead of a
    :class:`~tlpipe.container.raw_timestream.RawTimestream`).

    """

    params_init = {
                    'hist_equal': False, # Histogram equalization
                    'fig_name': 'wf/',
                    'feed_no': False, # True to use feed number (true baseline) else use channel no
                    'order_bl': True, # True to make small feed no first
                    'plot_list': ['freq']
                  }

    prefix = 'pwf_'

    def process(self, ts):

        ts.redistribute('baseline')

        if isinstance(ts, RawTimestream):
            func = ts.bl_data_operate
        elif isinstance(ts, Timestream):
            func = ts.pol_and_bl_data_operate

        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']

        func(self.plot, full_data=True, show_progress=show_progress, progress_step=progress_step, keep_dist_axis=False)

        return super(Plot, self).process(ts)

    def plot(self, vis, vis_mask, li, gi, bl, ts, **kwargs):
        """Function that does the actual plot work."""

        hist_equal = self.params['hist_equal']
        fig_prefix = self.params['fig_name']
        feed_no = self.params['feed_no']
        order_bl = self.params['order_bl']
        tag_output_iter = self.params['tag_output_iter']
        plot_list = self.params['plot_list']
        print("plot_list=",plot_list)
        iteration = self.iteration
 #       print("keys=",ts.keys())
 #       print("attrs=",ts.attrs.keys())
#8 hours is the difference between China time & UTC
        sec1970 = ts['sec1970'][0] + 8.0*3600.0 + time.timezone
        tstart = ' Starting ' + time.ctime(sec1970)
#        print("tstart=",tstart)

        if isinstance(ts, Timestream): # for Timestream
            pol = bl[0]
            bl = tuple(bl[1])
            feed_no = True
        elif isinstance(ts, RawTimestream): # for RawTimestream
            pol = None
            bl = tuple(bl)
            if feed_no:
                pol = ts['bl_pol'].local_data[li]
                bl = tuple(ts['true_blorder'].local_data[li])
                if order_bl and (bl[0] > bl[1]):
                    bl = (bl[1], bl[0])
                    vis = vis.conj()
        else:
            raise ValueError('Need either a RawTimestream or Timestream')
        print("Startpol, bl=",pol,bl,vis.shape)
#        print("begin",vis[0,0],vis[3599,0])
#       print("begin",vis[0,1],vis[3599,1])

        ntpt = vis.shape[0]
        freq = ts.freq[:]
        freq_label = r'$Frequency$ (MHz)'
        time_label = r'$Time$ (sec)'


        freq_extent = [freq[0], freq[-1]]
        time_extent = [0.0,np.float32(ntpt-1)]
        extent = time_extent +  freq_extent 
#        plot_list = ['resub','freq']
#        plot_list = ['freq']
        for ptype in plot_list:
            if ptype=='resub':
                paxis = '2D'
                subvis = True
                title = 'Subtracted Real(Vis)' + tstart
                vis1 = np.ma.array(vis.real,mask=vis_mask,copy=True)
            elif ptype=='imsub':
                paxis = '2D'
                subvis = True
                title = 'Subtracted Imag(Vis)' + tstart
                vis1 = np.ma.array(vis.imag, mask=vis_mask,copy=True)
            elif ptype=='abs':
                paxis = '2D'
                subvis = False
                title = 'Abs(Vis)' + tstart
                vis1 = np.ma.array(np.abs(vis), mask=vis_mask,copy=True)
            elif ptype=='freq':
                paxis = 'freq'
                subvis = False
                nl = ntpt/2-100
                if nl<0:
                    nl = 0
                nu = nl + 200
                if nu>ntpt-1:
                    nu = ntpt - 1
                title = 'Abs(Vis) Slice %i-%i' % (nl,nu)
                title = title + tstart
                vis1 = np.ma.array(np.abs(vis[nl:nu,:]), \
                            mask=vis_mask[nl:nu,:],copy=True)
            else:
                continue

            if paxis=='2D':
                plt.figure(figsize=[12,3],dpi=600)         
                axes = plt.subplot(111)
                if subvis:
                    ave = np.mean(vis1,axis=0)
                    vis1[:] = vis1[:] - ave
                im = plt.imshow(vis1.T,aspect=8.0,extent=extent,origin='lower')

                axes.set_xlabel(time_label)
                axes.set_ylabel(freq_label)
                plt.colorbar(im)
                xtext = 0.90
                ytext = 1.05

            elif paxis=='freq':
                plt.figure()
                axes = plt.subplot(111)
                fslice = np.mean(vis1,axis=0)
                axes.plot(freq,fslice)
                axes.set_xlabel(freq_label)
                axes.set_ylabel('Amplitude (Uncalibrated)')
                xtext = 0.85
                ytext = 0.90

            axes.set_title(title)
            if feed_no:
                fig_name = '%s%s_%d_%d_%s.png' % \
                    (fig_prefix, ptype, bl[0], bl[1], ts.pol_dict[pol])
                vislabel = 'V(%d,%d) %s' % (bl[0],bl[1],ts.pol_dict[pol])
            else:
                fig_name = '%s%s_%d_%d.png' % \
                    (fig_prefix, ptype, bl[0], bl[1])
                vislabel = 'V(%d,%d)' % (bl[0],bl[1])
            print("figure=",fig_name)
            if tag_output_iter:
                fig_name = output_path(fig_name, iteration=iteration)
            else:
                fig_name = output_path(fig_name)
            plt.text(xtext,ytext,vislabel,transform=axes.transAxes)
            plt.savefig(fig_name)
            plt.close()
#        print("end",vis[0,0],vis[3599,0])
