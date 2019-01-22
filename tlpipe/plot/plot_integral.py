"""Plot time or frequency integral.

Inheritance diagram
-------------------

.. inheritance-diagram:: Plot
   :parts: 2

"""

from datetime import datetime, timedelta
import numpy as np
from tlpipe.timestream import timestream_task
from tlpipe.container.raw_timestream import RawTimestream
from tlpipe.container.timestream import Timestream
from tlpipe.utils.path_util import output_path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator, AutoMinorLocator


class Plot(timestream_task.TimestreamTask):
    """Plot time or frequency integral.

    This tasks plots the real, imagery part and the absolute value of the
    time or frequency integrated visibility for each baseline (and also each
    polarization if the input data is a
    :class:`~tlpipe.container.timestream.Timestream` instead of a
    :class:`~tlpipe.container.raw_timestream.RawTimestream`).

    """

    params_init = {
                    'integral': 'time', # or 'freq'
                    'bl_incl': 'all', # or a list of include (bl1, bl2)
                    'bl_excl': [],
                    'flag_mask': True,
                    'flag_ns': True,
                    'fig_name': 'int/int',
                    'rotate_xdate': False, # True to rotate xaxis date ticks, else half the number of date ticks
                    'feed_no': False, # True to use feed number (true baseline) else use channel no
                    'order_bl': True, # True to make small feed no first
                  }

    prefix = 'pit_'

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

        integral = self.params['integral']
        bl_incl = self.params['bl_incl']
        bl_excl = self.params['bl_excl']
        flag_mask = self.params['flag_mask']
        flag_ns = self.params['flag_ns']
        fig_prefix = self.params['fig_name']
        rotate_xdate = self.params['rotate_xdate']
        feed_no = self.params['feed_no']
        order_bl = self.params['order_bl']
        tag_output_iter = self.params['tag_output_iter']
        iteration= self.iteration

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

        if bl_incl != 'all':
            bl1 = set(bl)
            bl_incl = [ {f1, f2} for (f1, f2) in bl_incl ]
            bl_excl = [ {f1, f2} for (f1, f2) in bl_excl ]
            if (not bl1 in bl_incl) or (bl1 in bl_excl):
                return

        if flag_mask:
            vis1 = np.ma.array(vis, mask=vis_mask)
        elif flag_ns:
            if 'ns_on' in ts.iterkeys():
                vis1 = vis.copy()
                on = np.where(ts['ns_on'][:])[0]
                vis1[on] = complex(np.nan, np.nan)
            else:
                vis1 = vis
        else:
            vis1 = vis

        if integral == 'time':
            vis1 = np.ma.mean(np.ma.masked_invalid(vis1), axis=0)
            ax_val = ts.freq[:]
            xlabel = r'$\nu$ / MHz'
        elif integral == 'freq':
            vis1 = np.ma.mean(np.ma.masked_invalid(vis1), axis=1)
            # ax_val = ts.time[:]
            # xlabel = r'$t$ / Julian Date'
            ax_val = np.array([ (datetime.utcfromtimestamp(sec) + timedelta(hours=8)) for sec in ts['sec1970'][:] ])
            xlabel = '%s' % ax_val[0].date()
            ax_val = mdates.date2num(ax_val)
        else:
            raise ValueError('Unknown integral type %s' % integral)

        plt.figure()
        f, axarr = plt.subplots(3, sharex=True)
        axarr[0].plot(ax_val, vis1.real, label='real')
        axarr[0].legend()
        axarr[1].plot(ax_val, vis1.imag, label='imag')
        axarr[1].legend()
        axarr[2].plot(ax_val, np.abs(vis1), label='abs')
        axarr[2].legend()
        if integral == 'freq':
            duration = (ax_val[-1] - ax_val[0])
            dt = duration / len(ax_val)
            ext = max(0.05*duration, 5*dt)
            axarr[2].set_xlim([ax_val[0]-ext, ax_val[-1]+ext])
            axarr[2].xaxis_date()
            date_format = mdates.DateFormatter('%H:%M')
            axarr[2].xaxis.set_major_formatter(date_format)
            if rotate_xdate:
                # set the x-axis tick labels to diagonal so it fits better
                f.autofmt_xdate()
            else:
                # reduce the number of tick locators
                locator = MaxNLocator(nbins=6)
                axarr[2].xaxis.set_major_locator(locator)
                axarr[2].xaxis.set_minor_locator(AutoMinorLocator(2))
        elif integral == 'time':
            bw = (ax_val[-1] - ax_val[0])
            df = bw / len(ax_val)
            ext = max(0.05*bw, df)
            axarr[2].set_xlim([ax_val[0]-ext, ax_val[-1]+ext])

        axarr[2].set_xlabel(xlabel)

        if feed_no:
            fig_name = '%s_%s_%d_%d_%s.png' % (fig_prefix, integral, bl[0], bl[1], ts.pol_dict[pol])
        else:
            fig_name = '%s_%s_%d_%d.png' % (fig_prefix, integral, bl[0], bl[1])
        if tag_output_iter:
            fig_name = output_path(fig_name, iteration=iteration)
        else:
            fig_name = output_path(fig_name)
        plt.savefig(fig_name)
        plt.close()
