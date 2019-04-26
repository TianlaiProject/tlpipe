"""Plot time or frequency slices.

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
    """Plot time or frequency slices.

    This task plots a given number of time (or frequency) slice of the visibility
    for each baseline (and also each polarization if the input data is a
    :class:`~tlpipe.container.timestream.Timestream` instead of a
    :class:`~tlpipe.container.raw_timestream.RawTimestream`).

    """

    params_init = {
                    'plot_type': 'time', # or 'freq'
                    'bl_incl': 'all', # or a list of include (bl1, bl2)
                    'bl_excl': [],
                    'flag_mask': True,
                    'flag_ns': True,
                    'slices': 10, # number of slices to plot
                    'fig_name': 'slice/slice',
                    'rotate_xdate': False, # True to rotate xaxis date ticks, else half the number of date ticks
                    'feed_no': False, # True to use feed number (true baseline) else use channel no
                    'order_bl': True, # True to make small feed no first
                  }

    prefix = 'psl_'

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

        plot_type = self.params['plot_type']
        bl_incl = self.params['bl_incl']
        bl_excl = self.params['bl_excl']
        flag_mask = self.params['flag_mask']
        flag_ns = self.params['flag_ns']
        slices = self.params['slices']
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

        if plot_type == 'time':
            nt = vis.shape[0]
            c = nt/2
            s = max(0, c-slices/2)
            e = min(nt, s+slices)
            if flag_mask:
                vis1 = np.ma.array(vis[s:e], mask=vis_mask[s:e])
            elif flag_ns:
                vis1 = vis[s:e].copy()
                if 'ns_on' in ts.iterkeys():
                    ns_on = ts['ns_on'][s:e]
                    on = np.where(ns_on)[0]
                    vis1[on] = complex(np.nan, np.nan)
            else:
                vis1 = vis[s:e]

            o = c - s
            shift = 0.1 * np.ma.max(np.abs(vis1[o]))

            ax_val = ts.freq[:]
            xlabel = r'$\nu$ / MHz'
        elif plot_type == 'freq':
            nfreq = vis.shape[1]
            c = nfreq/2
            s = max(0, c-slices/2)
            e = min(nfreq, s+slices)
            if flag_mask:
                vis1 = np.ma.array(vis[:, s:e], mask=vis_mask[:, s:e])
            elif flag_ns:
                vis1 = vis[:, s:e].copy()
                if 'ns_on' in ts.iterkeys():
                    ns_on = ts['ns_on'][:]
                    on = np.where(ns_on)[0]
                    vis1[on] = complex(np.nan, np.nan)
            else:
                vis1 = vis[:, s:e]

            o = c - s
            shift = 0.1 * np.ma.max(np.abs(vis1[:, o]))

            # ax_val = ts.time[:]
            # xlabel = r'$t$ / Julian Date'
            ax_val = np.array([ (datetime.utcfromtimestamp(sec) + timedelta(hours=8)) for sec in ts['sec1970'][:] ])
            xlabel = '%s' % ax_val[0].date()
            ax_val = mdates.date2num(ax_val)
        else:
            raise ValueError('Unknown plot_type %s, must be either time or freq' % plot_type)

        plt.figure()
        f, axarr = plt.subplots(3, sharex=True)
        for i in range(e - s):
            if plot_type == 'time':
                axarr[0].plot(ax_val, vis1[i].real + (i - o)*shift, label='real')
            elif plot_type == 'freq':
                axarr[0].plot(ax_val, vis1[:, i].real + (i - o)*shift, label='real')
            if i == 0:
                axarr[0].legend()

            if plot_type == 'time':
                axarr[1].plot(ax_val, vis1[i].imag + (i - o)*shift, label='imag')
            elif plot_type == 'freq':
                axarr[1].plot(ax_val, vis1[:, i].imag + (i - o)*shift, label='imag')
            if i == 0:
                axarr[1].legend()

            if plot_type == 'time':
                axarr[2].plot(ax_val, np.abs(vis1[i]) + (i - o)*shift, label='abs')
            elif plot_type == 'freq':
                axarr[2].plot(ax_val, np.abs(vis1[:, i]) + (i - o)*shift, label='abs')
            if i == 0:
                axarr[2].legend()

        if plot_type == 'freq':
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
                # half the number of date ticks so they do not overlap
                # axarr[2].set_xticks(axarr[2].get_xticks()[::2])
                # reduce the number of tick locators
                locator = MaxNLocator(nbins=6)
                axarr[2].xaxis.set_major_locator(locator)
                axarr[2].xaxis.set_minor_locator(AutoMinorLocator(2))
        elif plot_type == 'time':
            bw = (ax_val[-1] - ax_val[0])
            df = bw / len(ax_val)
            ext = max(0.05*bw, df)
            axarr[2].set_xlim([ax_val[0]-ext, ax_val[-1]+ext])

        axarr[2].set_xlabel(xlabel)

        if feed_no:
            fig_name = '%s_%s_%d_%d_%s.png' % (fig_prefix, plot_type, bl[0], bl[1], ts.pol_dict[pol])
        else:
            fig_name = '%s_%s_%d_%d.png' % (fig_prefix, plot_type, bl[0], bl[1])
        if tag_output_iter:
            fig_name = output_path(fig_name, iteration=iteration)
        else:
            fig_name = output_path(fig_name)
        plt.savefig(fig_name)
        plt.close()
