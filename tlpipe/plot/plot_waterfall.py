"""Plot waterfall images."""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from tlpipe.timestream import tod_task
from tlpipe.timestream.raw_timestream import RawTimestream
from tlpipe.timestream.timestream import Timestream
from tlpipe.utils.path_util import output_path
import matplotlib.pyplot as plt

def plot(vis, vis_mask, li, gi, bl, ts, **kwargs):

    bl_incl = kwargs.get('bl_incl', 'all')
    bl_excl = kwargs.get('bl_excl', [])
    flag_mask = kwargs.get('flag_mask', False)
    flag_ns = kwargs.get('flag_ns', False)
    interpolate_ns = kwargs.get('interpolate_ns', False)
    y_axis = kwargs.get('y_axis', 'jul_date')
    plot_abs = kwargs.get('plot_abs', False)
    fig_prefix = kwargs.get('fig_name', 'vis')
    tag_output_iter= kwargs.get('tag_output_iter', True)
    iteration= kwargs.get('iteration', None)

    if isinstance(ts, Timestream): # for Timestream
        pol = bl[0]
        bl = tuple(bl[1])
    elif isinstance(ts, RawTimestream): # for RawTimestream
        pol = None
        bl = tuple(bl)
    else:
        raise ValueError('Need either a RawTimestream or Timestream')

    if bl_incl != 'all':
        bl1 = set(bl)
        bl_incl = [ {f1, f2} for (f1, f2) in bl_incl ]
        bl_excl = [ {f1, f2} for (f1, f2) in bl_excl ]
        if (not bl1 in bl_incl) or (bl1 in bl_excl):
            return vis, vis_mask

    if flag_mask:
        vis1 = np.ma.array(vis, mask=vis_mask)
    elif flag_ns:
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

    freq = ts.freq[:]
    if y_axis == 'jul_date':
        y_aixs = ts.time[:]
        y_label = r'$t$ / Julian Date'
    elif y_axis == 'ra':
        y_aixs = ts['ra_dec'][:, 0]
        y_label = r'RA / radian'
    extent = [freq[0], freq[-1], y_aixs[0], y_aixs[-1]]

    plt.figure()
    if plot_abs:
        fig, axarr = plt.subplots(1, 3, sharey=True)
    else:
        fig, axarr = plt.subplots(1, 2, sharey=True)
    im = axarr[0].imshow(vis1.real, extent=extent, origin='lower', aspect='auto')
    axarr[0].set_xlabel(r'$\nu$ / MHz')
    axarr[0].set_ylabel(y_label)
    plt.colorbar(im, ax=axarr[0])
    im = axarr[1].imshow(vis1.imag, extent=extent, origin='lower', aspect='auto')
    axarr[1].set_xlabel(r'$\nu$ / MHz')
    plt.colorbar(im, ax=axarr[1])
    if plot_abs:
        im = axarr[2].imshow(np.abs(vis1), extent=extent, origin='lower', aspect='auto')
        axarr[2].set_xlabel(r'$\nu$ / MHz')
        plt.colorbar(im, ax=axarr[2])

    if pol is None:
        fig_name = '%s_%d_%d.png' % (fig_prefix, bl[0], bl[1])
    else:
        fig_name = '%s_%d_%d_%s.png' % (fig_prefix, bl[0], bl[1], pol)

    if tag_output_iter:
        fig_name = output_path(fig_name, iteration=iteration)
    else:
        fig_name = output_path(fig_name)
    plt.savefig(fig_name)
    plt.close()

    return vis, vis_mask

class Plot(tod_task.TaskTimestream):
    """Waterfall plot for Timestream."""

    params_init = {
                    'bl_incl': 'all', # or a list of include (bl1, bl2)
                    'bl_excl': [],
                    'flag_mask': False,
                    'flag_ns': False,
                    'interpolate_ns': False,
                    'y_axis': 'jul_date', # or 'ra'
                    'plot_abs': False,
                    'fig_name': 'vis',
                  }

    prefix = 'pwf_'

    def process(self, ts):
        bl_incl = self.params['bl_incl']
        bl_excl = self.params['bl_excl']
        flag_mask = self.params['flag_mask']
        flag_ns = self.params['flag_ns']
        interpolate_ns = self.params['interpolate_ns']
        y_axis = self.params['y_axis']
        plot_abs = self.params['plot_abs']
        fig_name = self.params['fig_name']
        tag_output_iter = self.params['tag_output_iter']

        ts.redistribute('baseline')

        if isinstance(ts, RawTimestream):
            func = ts.bl_data_operate
        elif isinstance(ts, Timestream):
            func = ts.pol_and_bl_data_operate

        func(plot, full_data=True, keep_dist_axis=False, bl_incl=bl_incl, bl_excl=bl_excl, fig_name=fig_name, tag_output_iter=tag_output_iter, iteration=self.iteration, flag_mask=flag_mask, flag_ns=flag_ns, interpolate_ns=interpolate_ns, y_axis=y_axis, plot_abs=plot_abs)

        ts.add_history(self.history)

        return ts
