"""Plot waterfall images."""

import os
import numpy as np
from tlpipe.timestream import tod_task
from tlpipe.utils.path_util import output_path
import matplotlib.pyplot as plt

def plot(vis, li, gi, bl, obj, **kwargs):

    if isinstance(bl, tuple): # for Timestream
        pol = bl[0]
        bl = tuple(bl[1])
    else: # for RawTimestream
        pol = None
        bl = tuple(bl)
    bl_incl = kwargs.get('bl_incl', 'all')
    bl_excl = kwargs.get('bl_excl', [])
    flag_ns = kwargs.get('flag_ns', False)
    y_axis = kwargs.get('y_axis', 'jul_date')
    plot_abs = kwargs.get('plot_abs', False)
    fig_prefix = kwargs.get('fig_name', 'vis')

    if bl_incl != 'all':
        bl1 = set(bl)
        bl_incl = [ {f1, f2} for (f1, f2) in bl_incl ]
        bl_excl = [ {f1, f2} for (f1, f2) in bl_excl ]
        if (not bl1 in bl_incl) or (bl1 in bl_excl):
            return vis

    if flag_ns:
        vis1 = vis.copy()
        on = np.where(obj['ns_on'][:])[0]
        vis1[on] = complex(np.nan, np.nan)
    else:
        vis1 = vis

    freq = obj.freq[:]
    if y_axis == 'jul_date':
        y_aixs = obj.time[:]
        y_label = r'$t$ / Julian Date'
    elif y_axis == 'ra':
        y_aixs = obj['ra_dec'][:, 0]
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
    fig_name = output_path(fig_name)
    plt.savefig(fig_name)
    plt.clf()

    return vis


class PlotRawTimestream(tod_task.SingleRawTimestream):
    """Waterfall plot for RawTimestream."""

    params_init = {
                    'bl_incl': 'all', # or a list of include (bl1, bl2)
                    'bl_excl': [],
                    'flag_ns': False,
                    'y_axis': 'jul_date', # or 'ra'
                    'plot_abs': False,
                    'fig_name': 'vis',
                  }

    prefix = 'prt_'

    def process(self, rt):
        bl_incl = self.params['bl_incl']
        bl_excl = self.params['bl_excl']
        flag_ns = self.params['flag_ns']
        y_axis = self.params['y_axis']
        plot_abs = self.params['plot_abs']
        fig_name = self.params['fig_name']

        rt.bl_data_operate(plot, full_data=True, keep_dist_axis=False, bl_incl=bl_incl, bl_excl=bl_excl, fig_name=fig_name, flag_ns=flag_ns, y_axis=y_axis, plot_abs=plot_abs)
        rt.add_history(self.history)

        return rt


class PlotTimestream(tod_task.SingleTimestream):
    """Waterfall plot for Timestream."""

    params_init = {
                    'bl_incl': 'all', # or a list of include (bl1, bl2)
                    'bl_excl': [],
                    'flag_ns': False,
                    'y_axis': 'jul_date', # or 'ra'
                    'plot_abs': False,
                    'fig_name': 'vis',
                  }

    prefix = 'pts_'

    def process(self, ts):
        bl_incl = self.params['bl_incl']
        bl_excl = self.params['bl_excl']
        flag_ns = self.params['flag_ns']
        y_axis = self.params['y_axis']
        plot_abs = self.params['plot_abs']
        fig_name = self.params['fig_name']

        ts.pol_and_bl_data_operate(plot, full_data=True, keep_dist_axis=False, bl_incl=bl_incl, bl_excl=bl_excl, fig_name=fig_name, flag_ns=flag_ns, y_axis=y_axis, plot_abs=plot_abs)
        ts.add_history(self.history)

        return ts
