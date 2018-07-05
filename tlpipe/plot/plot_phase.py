"""Plot phase of a time-frequency slice.

Inheritance diagram
-------------------

.. inheritance-diagram:: Plot
   :parts: 2

"""

from datetime import datetime, timedelta
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from tlpipe.timestream import timestream_task
from tlpipe.container.raw_timestream import RawTimestream
from tlpipe.container.timestream import Timestream
from tlpipe.utils.path_util import output_path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator, AutoMinorLocator


class Plot(timestream_task.TimestreamTask):
    """Plot phase of a time-frequency slice.

    This task plots the phase of a time-frequency visibility slice
    for each baseline (and also each polarization if the input data is a
    :class:`~tlpipe.container.timestream.Timestream` instead of a
    :class:`~tlpipe.container.raw_timestream.RawTimestream`).

    """

    params_init = {
                    'bl_incl': 'all', # or a list of include (bl1, bl2)
                    'bl_excl': [],
                    'flag_mask': False,
                    'flag_ns': False,
                    'interpolate_ns': False,
                    'y_axis': 'time', # or 'jul_date', or 'ra'
                    'transpose': False, # if True, x axis will be time
                    'plot_abs': False,
                    'gray_color': False,
                    'color_flag': False,
                    'flag_color': 'yellow',
                    'fig_name': 'phase/phs',
                    'rotate_xdate': False, # True to rotate xaxis date ticks, else half the number of date ticks
                    'feed_no': False, # True to use feed number (true baseline) else use channel no
                    'order_bl': True, # True to make small feed no first
                  }

    prefix = 'pph_'

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

        bl_incl = self.params['bl_incl']
        bl_excl = self.params['bl_excl']
        flag_mask = self.params['flag_mask']
        flag_ns = self.params['flag_ns']
        interpolate_ns = self.params['interpolate_ns']
        y_axis = self.params['y_axis']
        transpose = self.params['transpose']
        plot_abs = self.params['plot_abs']
        gray_color = self.params['gray_color']
        color_flag = self.params['color_flag']
        flag_color = self.params['flag_color']
        fig_prefix = self.params['fig_name']
        rotate_xdate = self.params['rotate_xdate']
        feed_no = self.params['feed_no']
        order_bl = self.params['order_bl']
        tag_output_iter = self.params['tag_output_iter']
        iteration = self.iteration

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
                if not interpolate_ns:
                    vis1[on] = complex(np.nan, np.nan)
                else:
                    off = np.where(np.logical_not(ts['ns_on'][:]))[0]
                    for fi in xrange(vis1.shape[1]):
                        itp_real = InterpolatedUnivariateSpline(off, vis1[off, fi].real)
                        itp_imag= InterpolatedUnivariateSpline(off, vis1[off, fi].imag)
                        vis1[on, fi] = itp_real(on) + 1.0J * itp_imag(on)
            else:
                vis1 = vis
        else:
            vis1 = vis

        freq = ts.freq[:]
        x_label = r'$\nu$ / MHz'
        if y_axis == 'jul_date':
            y_aixs = ts.time[:]
            y_label = r'$t$ / Julian Date'
        elif y_axis == 'ra':
            y_aixs = ts['ra_dec'][:, 0]
            y_label = r'RA / radian'
        elif y_axis == 'time':
            y_aixs = [ (datetime.utcfromtimestamp(s) + timedelta(hours=8)) for s in (ts['sec1970'][0], ts['sec1970'][-1]) ]
            y_label = '%s' % y_aixs[0].date()
            # convert datetime objects to the correct format for matplotlib to work with
            y_aixs = mdates.date2num(y_aixs)
        else:
            raise ValueError('Invalid y_axis %s, can only be "time", "jul_data" or "ra"' % y_axis)

        freq_extent = [freq[0], freq[-1]]
        time_extent = [y_aixs[0], y_aixs[-1]]
        extent = freq_extent + time_extent

        if transpose:
            vis1 = vis1.T
            x_label, y_label = y_label, x_label
            extent = time_extent + freq_extent

        plt.figure()

        if gray_color:
            # cmap = 'gray'
            cmap = plt.cm.gray
            if color_flag:
                cmap.set_bad(flag_color)
        else:
            cmap = None

        if plot_abs:
            if not transpose:
                fig, axarr = plt.subplots(1, 2, sharey=True)
                axarr[0].set_xlabel(x_label)
                axarr[1].set_xlabel(x_label)
            else:
                fig, axarr = plt.subplots(2, 1, sharex=True)
                axarr[0].set_ylabel(y_label)
                axarr[1].set_ylabel(y_label)
            im = axarr[1].imshow(np.abs(vis1), extent=extent, origin='lower', aspect='auto', cmap=cmap)
            plt.colorbar(im, ax=axarr[1])
            ax = axarr[0]
        else:
            fig, ax = plt.subplots()
            if not transpose:
                ax.set_xlabel(x_label)
            else:
                ax.set_ylabel(y_label)

        im = ax.imshow(np.ma.angle(vis1), extent=extent, origin='lower', aspect='auto', cmap=cmap)
        plt.colorbar(im, ax=ax)
        if plot_abs:
            if not transpose:
                ax = axarr[0]
            else:
                ax = axarr[1]
        # format datetime string
        date_format = mdates.DateFormatter('%H:%M')
        if not transpose:
            ax.yaxis_date()
            ax.yaxis.set_major_formatter(date_format)
            ax.set_ylabel(y_label)
        else:
            ax.xaxis_date()
            ax.xaxis.set_major_formatter(date_format)
            if rotate_xdate:
                # set the x-axis tick labels to diagonal so it fits better
                fig.autofmt_xdate()
            else:
                # reduce the number of tick locators
                locator = MaxNLocator(nbins=6)
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.set_xlabel(x_label)

        if feed_no:
            fig_name = '%s_%d_%d_%s.png' % (fig_prefix, bl[0], bl[1], ts.pol_dict[pol])
        else:
            fig_name = '%s_%d_%d.png' % (fig_prefix, bl[0], bl[1])

        if tag_output_iter:
            fig_name = output_path(fig_name, iteration=iteration)
        else:
            fig_name = output_path(fig_name)
        plt.savefig(fig_name)
        plt.close()
