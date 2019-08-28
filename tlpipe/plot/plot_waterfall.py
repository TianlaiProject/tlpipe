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
from tlpipe.timestream import timestream_task
from tlpipe.container.raw_timestream import RawTimestream
from tlpipe.container.timestream import Timestream
from tlpipe.utils.path_util import output_path
from tlpipe.utils import hist_eq
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator, AutoMinorLocator, MultipleLocator

from astropy.time import Time

from scipy.signal import medfilt

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

class PlotMeerKAT(timestream_task.TimestreamTask):

    params_init = {
            'main_data' : 'vis',
            'corr' : 'auto',
            'flag_mask' : True,
            'flag_ns'   : False,
            'flag_raw'  : True,
            're_scale'  : None,
            'vmin'      : None,
            'vmax'      : None,
            'xmin'      : None,
            'xmax'      : None,
            'ymin'      : None,
            'ymax'      : None,
            'fig_name'  : 'wf/',
            'bad_freq_list' : None,
            'bad_time_list' : None,
            'show'          : None,
            'plot_index'    : False,
            'plot_ra'       : False,
            'unit' : r'${\rm T}\,[{\rm K}]$', 
            }
    prefix = 'pkat_'

    def process(self, ts):

        ts.main_data_name = self.params['main_data']

        #ts.redistribute('baseline')

        func = ts.bl_data_operate

        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']

        bad_time_list = self.params['bad_time_list']
        bad_freq_list = self.params['bad_freq_list']

        if bad_time_list is not None:
            print "Mask bad time"
            for bad_time in bad_time_list:
                print bad_time
                ts.vis_mask[slice(*bad_time), ...] = True

        if bad_freq_list is not None:
            print "Mask bad freq"
            for bad_freq in bad_freq_list:
                print bad_freq
                ts.vis_mask[:, slice(*bad_freq), ...] = True

        if 'flags' in ts.iterkeys() and self.params['flag_raw']:
            print 'apply raw flags'
            ts.vis_mask[:] += ts['flags'][:]

        func(self.plot, full_data=True, show_progress=show_progress, 
                progress_step=progress_step, keep_dist_axis=False)

        return super(PlotMeerKAT, self).process(ts)


    def plot(self, vis, vis_mask, li, gi, bl, ts, **kwargs):

        if vis.dtype == np.complex or vis.dtype == np.complex64:
            print "take the abs of complex value"
            vis = np.abs(vis)

        flag_mask = self.params['flag_mask']
        flag_ns   = self.params['flag_ns']
        re_scale  = self.params['re_scale']
        vmin      = self.params['vmin']
        vmax      = self.params['vmax']
        xmin      = self.params['xmin']
        xmax      = self.params['xmax']
        ymin      = self.params['ymin']
        ymax      = self.params['ymax']
        fig_prefix = self.params['fig_name']
        main_data = self.params['main_data']

        vis1 = np.ma.array(vis)
        if flag_mask:
            vis1.mask = vis_mask

        #if 'flags' in ts.iterkeys():
        #    print 'apply raw flags'
        #    print ts['flags'].shape
        #    vis1.mask += ts['flags'][:]

        if flag_ns:
            if 'ns_on' in ts.iterkeys():
                print 'Uisng Noise Diode Mask for Ant. %03d'%(bl[0] - 1)
                #vis1 = vis.copy()
                #on = np.where(ts['ns_on'][:])[0]
                #vis1[on] = complex(np.nan, np.nan)
                on = ts['ns_on'][:, gi]
                vis1.mask[on, ...] = True
            else:
                print "No Noise Diode Mask info"

        if self.params['plot_index']:
            y_axis = np.arange(ts.freq.shape[0])
            y_label = r'$\nu$ index'
            x_axis = np.arange(ts['sec1970'].shape[0])
            x_label = 'time index'
        else:
            y_axis = ts.freq[:] * 1.e-3
            y_label = r'$\nu$ / GHz'
            if self.params['plot_ra']:
                #print ts['ra'].shape
                #print gi, li
                x_axis = ts['ra'][:, gi]
                x_label = 'R.A.' 
            else:
                x_axis = [ datetime.fromtimestamp(s) for s in ts['sec1970']]
                x_label = 'UTC %s' % x_axis[0].date()
                # convert datetime objects to the correct format for 
                # matplotlib to work with
                x_axis = mdates.date2num(x_axis)
            if xmin is not None:
                xmin = x_axis[xmin]
            if xmax is not None:
                xmax = x_axis[xmax]

        bad_time = np.all(vis_mask, axis=(1, 2))
        bad_freq = np.all(vis_mask, axis=(0, 2))

        if np.any(bad_time):
            good_time_st = np.argwhere(~bad_time)[ 0, 0]
            good_time_ed = np.argwhere(~bad_time)[-1, 0]
            vis1 = vis1[good_time_st:good_time_ed, ...]
            x_axis = x_axis[good_time_st:good_time_ed]

        if np.any(bad_freq):
            good_freq_st = np.argwhere(~bad_freq)[ 0, 0]
            good_freq_ed = np.argwhere(~bad_freq)[-1, 0]
            vis1 = vis1[:, good_freq_st:good_freq_ed, ...]
            y_axis = y_axis[good_freq_st:good_freq_ed]


        if re_scale is not None:
            mean = np.ma.mean(vis1)
            std  = np.ma.std(vis1)
            print mean, std
            vmax = mean + re_scale * std
            vmin = mean - re_scale * std
        else:
            vmax = self.params['vmax']
            vmin = self.params['vmin']

        fig  = plt.figure(figsize=(10, 6))
        axhh = fig.add_axes([0.10, 0.52, 0.75, 0.40])
        axvv = fig.add_axes([0.10, 0.10, 0.75, 0.40])
        cax  = fig.add_axes([0.86, 0.20, 0.02, 0.60])

        im = axhh.pcolormesh(x_axis, y_axis, vis1[:,:,0].T, vmax=vmax, vmin=vmin)
        im = axvv.pcolormesh(x_axis, y_axis, vis1[:,:,1].T, vmax=vmax, vmin=vmin)

        fig.colorbar(im, cax=cax, ax=axvv)

        axhh.set_title('Antenna M%03d'%(bl[0] - 1))

        # format datetime string
        # date_format = mdates.DateFormatter('%y/%m/%d %H:%M')
        date_format = mdates.DateFormatter('%H:%M')
        # date_format = mdates.DateFormatter('%H:%M', tz=pytz.timezone('Asia/Shanghai'))

        ## reduce the number of tick locators
        #locator = MaxNLocator(nbins=6)
        #ax.xaxis.set_major_locator(locator)
        #ax.xaxis.set_minor_locator(AutoMinorLocator(2))

        if not self.params['plot_index'] and not self.params['plot_ra']:
            axhh.xaxis.set_major_formatter(date_format)
            #axhh.xaxis.set_major_locator(MultipleLocator(0.5))
        axhh.set_xticklabels([])
        axhh.set_ylabel(r'${\rm frequency\, [GHz]\, HH}$')
        if xmin is None: xmin = x_axis[0]
        if xmax is None: xmax = x_axis[-1]
        if ymin is None: ymin = y_axis[0]
        if ymax is None: ymax = y_axis[-1]
        axhh.set_xlim(xmin=xmin, xmax=xmax)
        axhh.set_ylim(ymin=ymin, ymax=ymax)
        axhh.minorticks_on()
        axhh.tick_params(length=4, width=1, direction='in')
        axhh.tick_params(which='minor', length=2, width=1, direction='in')

        if not self.params['plot_index'] and not self.params['plot_ra']:
            axvv.xaxis.set_major_formatter(date_format)
        #axvv.set_xlabel(r'$({\rm time} - {\rm UT}\quad %s\,) [{\rm hour}]$'%t_start)
        axvv.set_xlabel(x_label)
        axvv.set_ylabel(r'${\rm frequency\, [GHz]\, VV}$')
        axvv.set_xlim(xmin=xmin, xmax=xmax)
        axvv.set_ylim(ymin=ymin, ymax=ymax)
        axvv.minorticks_on()
        axvv.tick_params(length=4, width=1, direction='in')
        axvv.tick_params(which='minor', length=2, width=1, direction='in')

        if not self.params['plot_index'] and not self.params['plot_ra']:
            fig.autofmt_xdate()

        #cax.set_ylabel(r'${\rm V}/{\rm V}_{\rm time median}$')
        #cax.set_ylabel(r'${\rm V}/{\rm V}_{\rm noise\, cal}$')
        cax.set_ylabel(self.params['unit'])

        if fig_prefix is not None:
            fig_name = '%s_%s_m%03d_x_m%03d.png' % (fig_prefix, main_data,
                                                    bl[0]-1,    bl[1]-1)
            fig_name = output_path(fig_name)
            plt.savefig(fig_name, formate='png') #, dpi=100)
        if self.params['show'] is not None:
            if self.params['show'] == bl[0]-1:
                plt.show()
        plt.close()

class PlotTimeStream(timestream_task.TimestreamTask):

    params_init = {
            'main_data' : 'vis',
            'corr' : 'auto',
            'flag_mask' : True,
            'flag_ns'   : False,
            're_scale'  : None,
            'vmin'      : None,
            'vmax'      : None,
            'xmin'      : None,
            'xmax'      : None,
            'ymin'      : None,
            'ymax'      : None,
            'fig_name'  : 'wf/',
            'bad_freq_list' : None,
            'bad_time_list' : None,
            'show'          : None,
            'plot_index'    : False,
            'plot_ra'       : False,
            'legend_title' : '', 
            'nvss_cat' : None,
            }
    prefix = 'ptsbase_'

    def process(self, ts):

        fig  = plt.figure(figsize=(8, 6))
        self.axhh = fig.add_axes([0.11, 0.52, 0.83, 0.40])
        self.axvv = fig.add_axes([0.11, 0.10, 0.83, 0.40])
        self.fig  = fig
        self.xmin =  1.e19
        self.xmax = -1.e19

        ts.main_data_name = self.params['main_data']

        ts.redistribute('baseline')

        func = ts.bl_data_operate

        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']

        bad_time_list = self.params['bad_time_list']
        bad_freq_list = self.params['bad_freq_list']

        if bad_time_list is not None:
            print "Mask bad time"
            for bad_time in bad_time_list:
                print bad_time
                ts.vis_mask[slice(*bad_time), ...] = True

        if bad_freq_list is not None:
            print "Mask bad freq"
            for bad_freq in bad_freq_list:
                print bad_freq
                ts.vis_mask[:, slice(*bad_freq), ...] = True

        if self.params['nvss_cat'] is not None:
            self.nvss_range = []


        func(self.plot, full_data=True, show_progress=show_progress, 
                progress_step=progress_step, keep_dist_axis=False)

        self.write_output(None)
        #return super(PlotTimeStream, self).process(ts)

    def plot(self, vis, vis_mask, li, gi, bl, ts, **kwargs):

        pass

    def write_output(self, output):

        fig_prefix = self.params['output_files'][0]
        ymin      = self.params['ymin']
        ymax      = self.params['ymax']
        main_data = self.params['main_data']

        axhh = self.axhh
        axvv = self.axvv
        fig  = self.fig

        if self.params['nvss_cat'] is not None:
            nvss_cat = get_nvss_radec(self.params['nvss_cat'] , self.nvss_range)
            _ymin, _ymax = axhh.get_ylim()
            for ii in range(nvss_cat.shape[0]):
                axhh.axvline(nvss_cat['RA'][ii], 0, 1, 
                        color='k', linestyle='--', linewidth=0.8)
                axvv.axvline(nvss_cat['RA'][ii], 0, 1, 
                        color='k', linestyle='--', linewidth=0.8)

        x_label = self.x_label

        date_format = mdates.DateFormatter('%H:%M')

        if not self.params['plot_index'] and not self.params['plot_ra']:
            axhh.xaxis.set_major_formatter(date_format)
            #axhh.xaxis.set_major_locator(MultipleLocator(1./6.))
            axhh.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            axhh.xaxis.set_minor_locator(mdates.MinuteLocator(interval=10))
        axhh.set_xticklabels([])
        #axhh.set_ylabel(r'${\rm frequency\, [GHz]\, HH}$')
        #axhh.set_ylabel('HH Polarization')
        #if xmin is None: xmin = x_axis[0]
        #if xmax is None: xmax = x_axis[-1]
        xmin = self.xmin
        xmax = self.xmax
        axhh.set_xlim(xmin=xmin, xmax=xmax)
        axhh.set_ylim(ymin=ymin, ymax=ymax)
        axhh.minorticks_on()
        axhh.tick_params(length=4, width=1, direction='in')
        axhh.tick_params(which='minor', length=2, width=1, direction='in')
        axhh.legend(title=self.params['legend_title'])

        if not self.params['plot_index'] and not self.params['plot_ra']:
            axvv.xaxis.set_major_formatter(date_format)
            #axvv.xaxis.set_major_locator(MultipleLocator(1./6.))
            axvv.xaxis.set_major_locator(mdates.HourLocator(interval=1))
            axvv.xaxis.set_minor_locator(mdates.MinuteLocator(interval=10))
        axhh.set_xticklabels([])
        #axvv.set_xlabel(r'$({\rm time} - {\rm UT}\quad %s\,) [{\rm hour}]$'%t_start)
        axvv.set_xlabel(x_label)
        #axvv.set_ylabel(r'${\rm frequency\, [GHz]\, VV}$')
        #axvv.set_ylabel('VV Polarization')
        axvv.set_xlim(xmin=xmin, xmax=xmax)
        axvv.set_ylim(ymin=ymin, ymax=ymax)
        axvv.minorticks_on()
        axvv.tick_params(length=4, width=1, direction='in')
        axvv.tick_params(which='minor', length=2, width=1, direction='in')

        if not self.params['plot_index'] and not self.params['plot_ra']:
            fig.autofmt_xdate()

        #cax.set_ylabel(r'${\rm V}/{\rm V}_{\rm time median}$')
        #cax.set_ylabel(r'${\rm V}/{\rm V}_{\rm noise\, cal}$')
        #cax.set_ylabel(self.params['unit'])

def get_nvss_radec(nvss_path, nvss_range):

    import astropy.io.fits as pyfits

    hdulist = pyfits.open(nvss_path)
    data = hdulist[1].data

    _sel = np.zeros(data['RA'].shape[0], dtype ='bool')
    for _nvss_range in nvss_range:
        print _nvss_range

        ra_min, ra_max, dec_min, dec_max = _nvss_range

        sel  = data['RA'] > ra_min
        sel *= data['RA'] < ra_max
        sel *= data['DEC'] > dec_min
        sel *= data['DEC'] < dec_max

        print np.sum(sel)

        _sel = _sel + sel

    hdulist.close()

    return data[_sel]


class PlotVvsTime(PlotTimeStream):

    prefix = 'pts_'


    def plot(self, vis, vis_mask, li, gi, bl, ts, **kwargs):

        #vis = np.abs(vis)
        if vis.dtype == np.complex or vis.dtype == np.complex64:
            print "take the abs of complex value"
            vis = np.abs(vis)

        flag_mask = self.params['flag_mask']
        flag_ns   = self.params['flag_ns']
        re_scale  = self.params['re_scale']
        vmin      = self.params['vmin']
        vmax      = self.params['vmax']
        xmin      = self.params['xmin']
        xmax      = self.params['xmax']
        ymin      = self.params['ymin']
        ymax      = self.params['ymax']

        vis1 = np.ma.array(vis)
        if flag_mask:
            vis1.mask = vis_mask

        if flag_ns:
            if 'ns_on' in ts.iterkeys():
                print 'Uisng Noise Diode Mask for Ant. %03d'%(bl[0] - 1)
                #vis1 = vis.copy()
                #on = np.where(ts['ns_on'][:])[0]
                #vis1[on] = complex(np.nan, np.nan)
                if len(ts['ns_on'].shape) == 2:
                    on = ts['ns_on'][:, gi].astype('bool')
                else:
                    on = ts['ns_on'][:]
                #on = ts['ns_on'][:]
                vis1.mask[on, ...] = True
            else:
                print "No Noise Diode Mask info"

        if self.params['plot_index']:
            y_label = r'$\nu$ index'
            x_axis = np.arange(ts['sec1970'].shape[0])
            self.x_label = 'time index'
        else:
            y_label = r'$\nu$ / GHz'
            #x_axis = [ datetime.fromtimestamp(s) for s in ts['sec1970']]
            #self.x_label = '%s' % x_axis[0].date()
            #x_axis = mdates.date2num(x_axis)
            if self.params['plot_ra']:
                #print ts['ra'].shape
                #print gi, li
                x_axis = ts['ra'][:, gi]
                self.x_label = 'R.A.' 
            else:
                x_axis = [ datetime.fromtimestamp(s) for s in ts['sec1970']]
                self.x_label = '%s UTC' % x_axis[0].date()
                x_axis = mdates.date2num(x_axis)
            if xmin is not None:
                xmin = x_axis[xmin]
            if xmax is not None:
                xmax = x_axis[xmax]

        bad_time = np.all(vis_mask, axis=(1, 2))
        bad_freq = np.all(vis_mask, axis=(0, 2))

        good_time_st = np.argwhere(~bad_time)[ 0, 0]
        good_time_ed = np.argwhere(~bad_time)[-1, 0]
        vis1 = vis1[good_time_st:good_time_ed, ...]
        x_axis = x_axis[good_time_st:good_time_ed]

        good_freq_st = np.argwhere(~bad_freq)[ 0, 0]
        good_freq_ed = np.argwhere(~bad_freq)[-1, 0]
        vis1 = vis1[:, good_freq_st:good_freq_ed, ...]

        self._plot(vis1, x_axis, xmin, xmax, gi, bl, ts)

    def _plot(self, vis1, x_axis, xmin, xmax, gi, bl, ts):

        axhh = self.axhh
        axvv = self.axvv

        label = 'M%03d'%(bl[0] - 1)
        #axhh.plot(x_axis, np.ma.mean(vis1[:,:,0], axis=1), '.', label = label)
        #axvv.plot(x_axis, np.ma.mean(vis1[:,:,1], axis=1), '.')
        _l = axhh.plot(x_axis, np.ma.mean(vis1[:,:,0], axis=1), '-', 
                label = label, drawstyle='steps-mid')
        axhh.plot(x_axis, np.ma.mean(vis1[:,:,0], axis=1), '.', c=_l[0].get_color())

        axvv.plot(x_axis, np.ma.mean(vis1[:,:,1], axis=1), '-',
                drawstyle='steps-mid')
        axvv.plot(x_axis, np.ma.mean(vis1[:,:,1], axis=1), '.', c=_l[0].get_color())


        if xmin is None: xmin = x_axis[0]
        if xmax is None: xmax = x_axis[-1]
        self.xmin = min([xmin, self.xmin])
        self.xmax = max([xmax, self.xmax])

        if self.params['nvss_cat'] is not None:
            dec_min = np.min(ts['dec'][:, gi])
            dec_max = np.max(ts['dec'][:, gi])
            #dec_min = 25.65294 #15.15294
            #dec_max = 25.65294 #15.15294
            #if dec_min == dec_max:
            dec_min -= 2./60.
            dec_max += 2./60.
            #dec_min = 25.541339365641274
            #dec_max = 25.61497357686361
            #dec_min = 25.458541361490884
            #dec_max = 25.532173665364585
            #dec_min = 25.789624659220376
            #dec_max = 25.863256963094077
            self.nvss_range.append([xmin, xmax, dec_min, dec_max])

    def write_output(self, output):

        super(PlotVvsTime, self).write_output(output)

        fig_prefix = self.params['output_files'][0]
        main_data = self.params['main_data']

        axhh = self.axhh
        axvv = self.axvv
        fig  = self.fig

        axhh.set_ylabel('HH Polarization')
        axvv.set_ylabel('VV Polarization')

        if fig_prefix is not None:
            fig_name = '%s_%s_TS.png' % (fig_prefix, main_data)
            fig_name = output_path(fig_name)
            plt.savefig(fig_name, formate='png') #, dpi=100)
        #if self.params['show'] is not None:
        #    if self.params['show'] == bl[0]-1:
        #        plt.show()
        #plt.close()

class PlotNcalVSTime(PlotVvsTime):


    params_init = {
            'noise_on_time' : 2,
            }

    prefix = 'pnt_'

    def plot(self, vis, vis_mask, li, gi, bl, ts, **kwargs):

        #vis = np.abs(vis)
        if vis.dtype == np.complex or vis.dtype == np.complex64:
            print "take the abs of complex value"
            vis = np.abs(vis)

        flag_mask = self.params['flag_mask']
        flag_ns   = self.params['flag_ns']
        re_scale  = self.params['re_scale']
        vmin      = self.params['vmin']
        vmax      = self.params['vmax']
        xmin      = self.params['xmin']
        xmax      = self.params['xmax']
        ymin      = self.params['ymin']
        ymax      = self.params['ymax']

        on_t = self.params['noise_on_time']

        vis1 = np.ma.array(vis)
        if flag_mask:
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

        if self.params['plot_index']:
            y_label = r'$\nu$ index'
            x_axis = np.arange(ts['sec1970'].shape[0])
            self.x_label = 'time index'
        else:
            y_label = r'$\nu$ / GHz'
            #x_axis = [ datetime.fromtimestamp(s) for s in ts['sec1970']]
            #self.x_label = '%s' % x_axis[0].date()
            #x_axis = mdates.date2num(x_axis)
            if self.params['plot_ra']:
                #print ts['ra'].shape
                #print gi, li
                x_axis = ts['ra'][:, gi]
                self.x_label = 'R.A.' 
            else:
                x_axis = [ datetime.fromtimestamp(s) for s in ts['sec1970']]
                self.x_label = '%s UTC' % x_axis[0].date()
                x_axis = mdates.date2num(x_axis)
            if xmin is not None:
                xmin = x_axis[xmin]
            if xmax is not None:
                xmax = x_axis[xmax]

        bad_time = np.all(vis_mask, axis=(1, 2))
        bad_freq = np.all(vis_mask, axis=(0, 2))

        good_time_st = np.argwhere(~bad_time)[ 0, 0]
        good_time_ed = np.argwhere(~bad_time)[-1, 0]
        vis1 = vis1[good_time_st:good_time_ed, ...]
        x_axis = x_axis[good_time_st:good_time_ed]

        on  = on[good_time_st:good_time_ed]
        #if on[0] and not on[1]: on[0] = False
        #if on[-1] and not on[-2]: on[-1] = False
        on[ :on_t] = False
        on[-on_t:] = False
        # rm noise cal with half missing
        #on  = (np.roll(on, 1) * on) + (np.roll(on, -1) * on)
        #off = (np.roll(on, 1) + np.roll(on, -1)) ^ on
        off = np.roll(on, 1)

        good_freq_st = np.argwhere(~bad_freq)[ 0, 0]
        good_freq_ed = np.argwhere(~bad_freq)[-1, 0]
        vis1 = vis1[:, good_freq_st:good_freq_ed, ...]

        #vis1 = vis1.data
        vis1_on  = vis1[on, ...]
        vis1_off = vis1[off, ...].data
        vis1 = vis1_on - vis1_off

        if on_t > 1:
            vis_shp = vis1.shape
            vis1 = vis1.reshape((-1, on_t) + vis_shp[1:])
            vis1 = vis1 + vis1[:, ::-1, ...]
            vis1.shape = vis_shp

        vis1 /= np.ma.median(vis1, axis=(0,1))[None, None, :]
        #vis1 /= np.ma.mean(vis1, axis=(0,1))[None, None, :]

        x_axis   = x_axis[on]

        axhh = self.axhh
        axvv = self.axvv

        label = 'M%03d'%(bl[0] - 1)
        #axhh.plot(x_axis, np.ma.mean(vis1[:,:,0], axis=1), '.', label = label)
        #axvv.plot(x_axis, np.ma.mean(vis1[:,:,1], axis=1), '.')

        vis1 = np.ma.median(vis1, axis=1)
        good = ~vis1.mask
        vis1_poly_xx = np.poly1d(np.polyfit(x_axis[good[:,0]], vis1[:, 0][good[:,0]], 4))
        vis1_poly_yy = np.poly1d(np.polyfit(x_axis[good[:,1]], vis1[:, 1][good[:,1]], 4))

        #vis1 = np.ma.array(medfilt(vis1.data, kernel_size=(11, 1)), mask = vis1.mask)

        _l = axhh.plot(x_axis, vis1[:,0], '-', lw=1, label = label, drawstyle='steps-mid')
        axhh.plot(x_axis, vis1_poly_xx(x_axis), '-', c=_l[0].get_color(), lw=2.0)
        _l = axvv.plot(x_axis, vis1[:,1], '-', lw=1, drawstyle='steps-mid')
        axvv.plot(x_axis, vis1_poly_yy(x_axis), '-', c=_l[0].get_color(), lw=2.0)

        axhh.axhline(1, 0, 1, c='k', lw=1.5, ls='--')
        axvv.axhline(1, 0, 1, c='k', lw=1.5, ls='--')

        if xmin is None: xmin = x_axis[0]
        if xmax is None: xmax = x_axis[-1]
        self.xmin = min([xmin, self.xmin])
        self.xmax = max([xmax, self.xmax])

class PlotNoiseCal(PlotVvsTime):

    prefix = 'pcal_'

    params_init = {
            'noise_cal_init_time' : None,
            'noise_cal_period' : 19.9915424299,
            'noise_cal_length' : 1.8,
            'noise_cal_delayed_ant' : [],
            }

    def plot(self, vis, vis_mask, li, gi, bl, ts, **kwargs):

        super(PlotNoiseCal, self).plot(vis, vis_mask, li, gi, bl, ts, **kwargs)

        axhh = self.axhh
        axvv = self.axvv
        fig  = self.fig

        if self.params['noise_cal_init_time'] is not None:
            print "plot noise diode"
            if self.params['plot_index']:
                msg = 'noise diode need to plot in time, not index'
            else:
                t0 = Time(self.params['noise_cal_init_time']).unix
                t1 = ts['sec1970'][-1]
                tl = self.params['noise_cal_length']
                tp = self.params['noise_cal_period']
                #tp = 19.9915424299

                if bl[0] - 1 in self.params['noise_cal_delayed_ant']:
                    print "Cal delayed, t0 plus 1 s"
                    t0 += 1

                noise_st = np.arange(t0,      t1, tp)
                noise_ed = np.arange(t0 + tl, t1, tp)

                for _st in noise_st:
                    _st = datetime.fromtimestamp(_st) 
                    _st = mdates.date2num(_st)
                    axhh.axvline(_st, color='k', linestyle='-', linewidth=1)
                    axvv.axvline(_st, color='k', linestyle='-', linewidth=1)

                for _ed in noise_ed:
                    _ed = datetime.fromtimestamp(_ed) 
                    _ed = mdates.date2num(_ed)
                    axhh.axvline(_ed, color='k', linestyle='--', linewidth=1)
                    axvv.axvline(_ed, color='k', linestyle='--', linewidth=1)

class PlotPointingvsTime(PlotTimeStream):

    prefix = 'ppt_'

    def plot(self, vis, vis_mask, li, gi, bl, ts, **kwargs):

        az  = ts['az'][:, gi]
        el = ts['el'][:, gi]

        vmin      = self.params['vmin']
        vmax      = self.params['vmax']
        xmin      = self.params['xmin']
        xmax      = self.params['xmax']
        ymin      = self.params['ymin']
        ymax      = self.params['ymax']

        if self.params['plot_index']:
            y_label = r'$\nu$ index'
            x_axis = np.arange(ts['sec1970'].shape[0])
            self.x_label = 'time index'
        else:
            y_label = r'$\nu$ / GHz'
            x_axis = [ datetime.fromtimestamp(s) for s in ts['sec1970']]
            self.x_label = '%s' % x_axis[0].date()
            x_axis = mdates.date2num(x_axis)

        bad_time = np.all(vis_mask, axis=(1, 2))

        good_time_st = np.argwhere(~bad_time)[ 0, 0]
        good_time_ed = np.argwhere(~bad_time)[-1, 0]
        x_axis = x_axis[good_time_st:good_time_ed]
        az = az[good_time_st:good_time_ed]
        el = el[good_time_st:good_time_ed]

        az[az < 0] = az[az < 0] + 360.
        az = (az - 180.) * 60.
        el = (el - np.mean(el)) * 60

        az_slope = np.poly1d(np.polyfit(x_axis, az, 2))
        az -= az_slope(x_axis)

        axhh = self.axhh
        axvv = self.axvv

        label = 'M%03d'%(bl[0] - 1)
        axhh.plot(x_axis, az, label = label)
        axvv.plot(x_axis, el)

        if xmin is None: xmin = x_axis[0]
        if xmax is None: xmax = x_axis[-1]
        self.xmin = min([xmin, self.xmin])
        self.xmax = max([xmax, self.xmax])

    def write_output(self, output):

        super(PlotPointingvsTime, self).write_output(output)

        fig_prefix = self.params['output_files'][0]
        main_data = self.params['main_data']

        axhh = self.axhh
        axvv = self.axvv
        fig  = self.fig

        axhh.set_ylabel('Azimuth [arcmin]')
        axvv.set_ylabel('Elevation [arcmin]')

        if fig_prefix is not None:
            fig_name = '%s_%s_AzEl.png' % (fig_prefix, main_data)
            fig_name = output_path(fig_name)
            plt.savefig(fig_name, formate='png') #, dpi=100)


class PlotSpectrum(PlotTimeStream):

    prefix = 'psp_'

    def plot(self, vis, vis_mask, li, gi, bl, ts, **kwargs):

        xmin      = self.params['xmin']
        xmax      = self.params['xmax']
        ymin      = self.params['ymin']
        ymax      = self.params['ymax']

        print "global index %2d [m%03d]"%(gi, bl[0]-1)
        freq_indx = np.arange(vis.shape[1])
        freq = ts['freq'][:] * 1.e-3
        self.x_label = r'$\nu$ / GHz'

        bad_freq = np.all(vis_mask, axis=(0, 2))
        bad_time = np.all(vis_mask, axis=(1, 2))

        if vis.dtype == 'complex' or vis.dtype == 'complex64':
            print "Convert complex to float by abs"
            vis = np.abs(vis.copy())

        vis = np.ma.array(vis)
        vis.mask = vis_mask.copy()
        
        if 'ns_on' in ts.iterkeys():
            ns_on = ts['ns_on'][:, gi]
        else:
            ns_on = np.zeros(bad_time.shape).astype('bool')

        vis.mask[ns_on, ...] = True
        spec_HH = np.ma.median(vis[..., 0], axis=0)
        spec_VV = np.ma.median(vis[..., 1], axis=0)
        
        #bp_HH = np.ma.array(medfilt(spec_HH, 11))
        #bp_HH.mask = bad_freq
        #spec_HH /= bp_HH
        
        #bp_VV = np.ma.array(medfilt(spec_VV, 11))
        #bp_VV.mask = bad_freq
        #spec_VV /= bp_VV

        axhh = self.axhh
        axvv = self.axvv
        
        label = 'M%03d'%(bl[0] - 1)
        axhh.plot(freq, spec_HH , label=label, linewidth=1.5)
        #ax.plot(freq, bp_HH , 'y-', linewidth=1.0)
        
        axvv.plot(freq, spec_VV , linewidth=1.5)
        #ax.plot(freq, bp_VV , 'y-', linewidth=1.0)

        #if 'ns_on' in ts.iterkeys():

        #    vis.mask = vis_mask.copy()
        #    vis.mask[~ns_on] = True
        #    ns_HH = np.ma.median(vis[..., 0], axis=0)
        #    ns_VV = np.ma.median(vis[..., 1], axis=0)

        #    ns_HH = np.ma.array(medfilt(ns_HH, 11))
        #    ns_HH.mask = bad_freq

        #    ns_VV = np.ma.array(medfilt(ns_VV, 11))
        #    ns_VV.mask = bad_freq

        #    axhh.plot(freq, ns_HH , 'g--', label='noise HH', linewidth=1.5)
        #    axvv.plot(freq, ns_VV , 'r--', label='noise VV', linewidth=1.5)

        if xmin is None: xmin = freq[0]
        if xmax is None: xmax = freq[-1]
        self.xmin = min([xmin, self.xmin])
        self.xmax = max([xmax, self.xmax])

    def write_output(self, output):

        #super(PlotSpectrum, self).write_output(output)

        fig_prefix = self.params['output_files'][0]
        main_data = self.params['main_data']
        ymin      = self.params['ymin']
        ymax      = self.params['ymax']

        axhh = self.axhh
        axvv = self.axvv
        fig  = self.fig

        x_label = self.x_label

        axhh.set_xticklabels([])
        #axhh.set_ylabel(r'${\rm frequency\, [GHz]\, HH}$')
        #axhh.set_ylabel('HH Polarization')
        #if xmin is None: xmin = x_axis[0]
        #if xmax is None: xmax = x_axis[-1]
        xmin = self.xmin
        xmax = self.xmax
        axhh.set_xlim(xmin=self.xmin, xmax=self.xmax)
        axhh.set_ylim(ymin=ymin, ymax=ymax)
        axhh.minorticks_on()
        axhh.tick_params(length=4, width=1, direction='in')
        axhh.tick_params(which='minor', length=2, width=1, direction='in')
        axhh.legend(title=self.params['legend_title'])
        axhh.set_ylabel('HH Polarization')

        #axvv.set_xlabel(r'$({\rm time} - {\rm UT}\quad %s\,) [{\rm hour}]$'%t_start)
        axvv.set_xlabel(x_label)
        #axvv.set_ylabel(r'${\rm frequency\, [GHz]\, VV}$')
        #axvv.set_ylabel('VV Polarization')
        axvv.set_xlim(xmin=self.xmin, xmax=self.xmax)
        axvv.set_ylim(ymin=ymin, ymax=ymax)
        axvv.minorticks_on()
        axvv.tick_params(length=4, width=1, direction='in')
        axvv.tick_params(which='minor', length=2, width=1, direction='in')
        axvv.set_ylabel('VV Polarization')

        if fig_prefix is not None:
            fig_name = '%s_%s_Spec.png' % (fig_prefix, main_data)
            fig_name = output_path(fig_name)
            plt.savefig(fig_name, formate='png') #, dpi=100)

class CheckSpec(timestream_task.TimestreamTask):
    
    prefix = 'csp_'
    
    params_init = {
        'corr' : 'auto',
        'bad_freq_list' : [],
        'bad_time_list' : [],
        'legend_title' : '',
        'show' : None,
        'ymin' : None,
        'ymax' : None,
        'xmin' : None,
        'xmax' : None,
    }
    
    
    def process(self, ts):
        
        bad_time_list = self.params['bad_time_list']
        bad_freq_list = self.params['bad_freq_list']
        
        if bad_time_list is not None:
            for bad_time in bad_time_list:
                print bad_time
                ts.vis_mask[slice(*bad_time), ...] = True

        if bad_freq_list is not None:
            print "Mask bad freq"
            for bad_freq in bad_freq_list:
                print bad_freq
                ts.vis_mask[:, slice(*bad_freq), ...] = True

        ts.redistribute('baseline')
        

        func = ts.bl_data_operate

        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']
        
        func(self.plot, full_data=False, show_progress=show_progress, 
             progress_step=progress_step, keep_dist_axis=False)

        return super(CheckSpec, self).process(ts)

    
    def plot(self, vis, vis_mask, li, gi, bl, ts, **kwargs):

        ymin = self.params['ymin']
        ymax = self.params['ymax']
        xmin = self.params['xmin']
        xmax = self.params['xmax']
        
        
        bad_freq = np.all(vis_mask, axis=(0, 2))
        bad_time = np.all(vis_mask, axis=(1, 2))
        
        print "global index %2d [m%03d]"%(gi, bl[0]-1)
        freq_indx = np.arange(vis.shape[1])
        freq = ts['freq'][:]
        
        fig = plt.figure(figsize=(8, 4))
        ax  = fig.add_axes([0.1, 0.1, 0.85, 0.85])
        
        if vis.dtype == 'complex' or vis.dtype == 'complex64':
            vis = np.abs(vis)
        vis = np.ma.array(vis)
        vis.mask = vis_mask.copy()
        
        #spec_HH = np.ma.mean(vis[:, :, 0], axis=0)
        #spec_VV = np.ma.mean(vis[:, :, 1], axis=0)

        if 'ns_on' in ts.iterkeys():
            ns_on = ts['ns_on'][:, gi]
        else:
            ns_on = np.zeros(bad_time.shape).astype('bool')

        vis.mask[ns_on, ...] = True
        spec_HH = np.ma.median(vis[..., 0], axis=0)
        spec_VV = np.ma.median(vis[..., 1], axis=0)
        
        #bp_HH = np.ma.array(medfilt(spec_HH, 11))
        #bp_HH.mask = bad_freq
        #spec_HH /= bp_HH
        
        #bp_VV = np.ma.array(medfilt(spec_VV, 11))
        #bp_VV.mask = bad_freq
        #spec_VV /= bp_VV

        
        ax.plot(freq, spec_HH , 'g-', label='HH', linewidth=1.5)
        #ax.plot(freq, bp_HH , 'y-', linewidth=1.0)
        
        ax.plot(freq, spec_VV , 'r-', label='VV', linewidth=1.5)
        #ax.plot(freq, bp_VV , 'y-', linewidth=1.0)

        if 'ns_on' in ts.iterkeys():

            vis.mask = vis_mask.copy()
            vis.mask[~ns_on] = True
            ns_HH = np.ma.median(vis[..., 0], axis=0)
            ns_VV = np.ma.median(vis[..., 1], axis=0)

            ns_HH = np.ma.array(medfilt(ns_HH, 11))
            ns_HH.mask = bad_freq

            ns_VV = np.ma.array(medfilt(ns_VV, 11))
            ns_VV.mask = bad_freq

            ax.plot(freq, ns_HH , 'g--', label='noise HH', linewidth=1.5)
            ax.plot(freq, ns_VV , 'r--', label='noise VV', linewidth=1.5)
        


        ax.set_xlabel('Frequency [MHz]')
        ax.set_ylabel('Power')
        #ax.semilogy()
        ax.legend(title=self.params['legend_title'])
        ax.tick_params(length=4, width=0.8, direction='in')
        ax.tick_params(which='minor', length=2, width=0.8, direction='in')
        
        ax.set_ylim(ymin, ymax)
        ax.set_xlim(xmin, xmax)

        if self.params['show'] is not None:
            if self.params['show'] == bl[0]-1:
                plt.show()
        plt.close()
        #plt.show()
