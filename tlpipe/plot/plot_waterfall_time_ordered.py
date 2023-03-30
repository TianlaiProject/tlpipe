"""Plot waterfall images.

Inheritance diagram
-------------------

.. inheritance-diagram:: Plot
   :parts: 2

"""

# import pytz
from datetime import datetime, timedelta
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from caput import mpiutil
from caput import mpiarray
from tlpipe.timestream import timestream_task
from tlpipe.container.raw_timestream import RawTimestream
from tlpipe.container.timestream import Timestream
from tlpipe.utils.path_util import output_path
from tlpipe.utils import hist_eq
from tlpipe.utils import progress
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

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
                    'bl_incl': 'all', # or a list of include (bl1, bl2)
                    'bl_excl': [],
                    'flag_mask': False,
                    'flag_ns': False,
                    'interpolate_ns': False,
                    'y_axis': 'time', # or 'jul_date', or 'ra'
                    'use_utc': False, # True to use UTC time, else Beijing time
                    'plot_abs': False,
                    'abs_only': False,
                    'gray_color': False,
                    'color_flag': False,
                    'flag_color': 'yellow',
                    'transpose': False, # now only for abs plot
                    'hist_equal': False, # Histogram equalization
                    'fig_name': 'wf/vis',
                    'rotate_xdate': False, # True to rotate xaxis date ticks, else half the number of date ticks
                    'feed_no': False, # True to use feed number (true baseline) else use channel no
                    'order_bl': True, # True to make small feed no first
                  }

    prefix = 'pwf_'

    def process(self, ts):
        via_memmap = self.params['via_memmap']
        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']

        ts.redistribute('time', via_memmap=via_memmap)

        if isinstance(ts, RawTimestream):
            redistribute_axis = 2
        elif isinstance(ts, Timestream):
            redistribute_axis = 3

        if 'ns_on' in ts.keys():
            ns_on = mpiutil.gather_array(ts['ns_on'].local_data, root=None, comm=ts.comm)
        else:
            ns_on = None
        sec1970 = mpiutil.gather_array(ts['sec1970'].local_data, root=None, comm=ts.comm)
        nbl = len(ts.bl) # number of baselines
        n, r = nbl // mpiutil.size, nbl % mpiutil.size # number of iterations
        if r != 0:
            n = n + 1
        if show_progress and mpiutil.rank0:
            pg = progress.Progress(n, step=progress_step)
        for i in range(n):
            if show_progress and mpiutil.rank0:
                pg.show(i)

            this_vis = ts.local_vis[..., i*mpiutil.size:(i+1)*mpiutil.size].copy()
            this_vis = mpiarray.MPIArray.wrap(this_vis, axis=0, comm=ts.comm).redistribute(axis=redistribute_axis)
            this_vis_mask = ts.local_vis_mask[..., i*mpiutil.size:(i+1)*mpiutil.size].copy()
            this_vis_mask = mpiarray.MPIArray.wrap(this_vis_mask, axis=0, comm=ts.comm).redistribute(axis=redistribute_axis)
            if this_vis.local_array.shape[-1] != 0:
                if isinstance(ts, RawTimestream):
                    self.plot(this_vis.local_array[..., 0], this_vis_mask.local_array[..., 0], 0, 0, 0, ts, ns_on=ns_on, sec1970=sec1970, pol=ts['bl_pol'][i*mpiutil.size+mpiutil.rank], bl=ts['true_blorder'][i*mpiutil.size+mpiutil.rank], chpair=ts['blorder'][i*mpiutil.size+mpiutil.rank])
                elif isinstance(ts, Timestream):
                    for pi in range(this_vis.local_array.shape[2]):
                        self.plot(this_vis.local_array[..., pi, 0], this_vis_mask.local_array[..., pi, 0], 0, 0, 0, ts, ns_on=ns_on, sec1970=sec1970, pol=pi, bl=ts['blorder'][i*mpiutil.size+mpiutil.rank], chpair=None)

        return super(Plot, self).process(ts)

    def plot(self, vis, vis_mask, li, gi, tf, ts, **kwargs):
        """Function that does the actual plot work."""

        bl_incl = self.params['bl_incl']
        bl_excl = self.params['bl_excl']
        flag_mask = self.params['flag_mask']
        flag_ns = self.params['flag_ns']
        interpolate_ns = self.params['interpolate_ns']
        y_axis = self.params['y_axis']
        use_utc = self.params['use_utc']
        plot_abs = self.params['plot_abs']
        abs_only = self.params['abs_only']
        gray_color = self.params['gray_color']
        color_flag = self.params['color_flag']
        flag_color = self.params['flag_color']
        transpose = self.params['transpose']
        hist_equal = self.params['hist_equal']
        fig_prefix = self.params['fig_name']
        rotate_xdate = self.params['rotate_xdate']
        feed_no = self.params['feed_no']
        order_bl = self.params['order_bl']
        tag_output_iter = self.params['tag_output_iter']
        iteration = self.iteration
        ns_on = kwargs['ns_on']
        sec1970 = kwargs['sec1970']
        pol = kwargs['pol']
        bl = kwargs['bl']
        chpair = kwargs['chpair']

        if isinstance(ts, Timestream): # for Timestream
            pol = pol
            bl = tuple(bl)
            feed_no = True
        elif isinstance(ts, RawTimestream): # for RawTimestream
            if feed_no:
                pol = pol
                bl = tuple(bl)
                if order_bl and (bl[0] > bl[1]):
                    bl = (bl[1], bl[0])
                    vis = vis.conj()
            else:
                pol = None
                bl = tuple(chpair)
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
            if 'ns_on' in ts.keys():
                vis1 = vis.copy()
                on = np.where(ns_on)[0]
                if not interpolate_ns:
                    vis1[on] = complex(np.nan, np.nan)
                else:
                    off = np.where(np.logical_not(ns_on))[0]
                    for fi in range(vis1.shape[1]):
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
            if use_utc:
                y_aixs = [ (datetime.utcfromtimestamp(s)) for s in (sec1970[0], sec1970[-1]) ]
                y_label = 'UT+0h: %s' % y_aixs[0].date()
            else:
                y_aixs = [ (datetime.utcfromtimestamp(s) + timedelta(hours=8)) for s in (sec1970[0], sec1970[-1]) ]
                y_label = '%s' % y_aixs[0].date()
            # convert datetime objects to the correct format for matplotlib to work with
            y_aixs = mdates.date2num(y_aixs)
        else:
            raise ValueError('Invalid y_axis %s, can only be "time", "jul_data" or "ra"' % y_axis)

        freq_extent = [freq[0], freq[-1]]
        time_extent = [y_aixs[0], y_aixs[-1]]
        extent = freq_extent + time_extent

        plt.figure()

        if gray_color:
            # cmap = 'gray'
            cmap = plt.cm.gray
            if color_flag:
                cmap.set_bad(flag_color)
        else:
            cmap = None

        if abs_only:
            if transpose:
                vis1 = vis1.T
                x_label, y_label = y_label, x_label
                extent = time_extent + freq_extent

            fig, ax = plt.subplots()
            vis_abs = np.abs(vis1)
            if hist_equal:
                if isinstance(vis_abs, np.ma.MaskedArray):
                    vis_hist = hist_eq.hist_eq(vis_abs.filled(0))
                    vis_abs = np.ma.array(vis_hist, mask=np.ma.getmask(vis_abs))
                else:
                    vis_hist = hist_eq.hist_eq(np.where(np.isfinite(vis_abs), vis_abs, 0))
                    mask = np.where(np.isfinite(vis_abs), False, True)
                    vis_abs = np.ma.array(vis_hist, mask=mask)
            im = ax.imshow(vis_abs, extent=extent, origin='lower', aspect='auto', cmap=cmap)
            # convert axis to datetime string
            if transpose:
                ax.xaxis_date()
            else:
                ax.yaxis_date()
            # format datetime string
            # date_format = mdates.DateFormatter('%y/%m/%d %H:%M')
            date_format = mdates.DateFormatter('%H:%M')
            # date_format = mdates.DateFormatter('%H:%M', tz=pytz.timezone('Asia/Shanghai'))
            if transpose:
                ax.xaxis.set_major_formatter(date_format)
            else:
                ax.yaxis.set_major_formatter(date_format)

            if transpose:
                if rotate_xdate:
                    # set the x-axis tick labels to diagonal so it fits better
                    fig.autofmt_xdate()
                else:
                    # reduce the number of tick locators
                    locator = MaxNLocator(nbins=6)
                    ax.xaxis.set_major_locator(locator)
                    ax.xaxis.set_minor_locator(AutoMinorLocator(2))

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            plt.colorbar(im)
        else:
            if plot_abs:
                fig, axarr = plt.subplots(1, 3, sharey=True)
            else:
                fig, axarr = plt.subplots(1, 2, sharey=True)
            im = axarr[0].imshow(vis1.real, extent=extent, origin='lower', aspect='auto', cmap=cmap)
            axarr[0].set_xlabel(x_label)
            axarr[0].yaxis_date()
            # format datetime string
            date_format = mdates.DateFormatter('%H:%M')
            axarr[0].yaxis.set_major_formatter(date_format)
            axarr[0].set_ylabel(y_label)
            plt.colorbar(im, ax=axarr[0])
            im = axarr[1].imshow(vis1.imag, extent=extent, origin='lower', aspect='auto', cmap=cmap)
            axarr[1].set_xlabel(x_label)
            plt.colorbar(im, ax=axarr[1])
            if plot_abs:
                im = axarr[2].imshow(np.abs(vis1), extent=extent, origin='lower', aspect='auto', cmap=cmap)
                axarr[2].set_xlabel(x_label)
                plt.colorbar(im, ax=axarr[2])

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
