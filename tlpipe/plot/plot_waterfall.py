"""Plot waterfall images."""

import os
import numpy as np
from tlpipe.timestream import tod_task
from tlpipe.utils.path_util import output_path


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

    import matplotlib.pyplot as plt

    plt.figure()

    freq = obj.freq[:]
    jul_date = obj.time[:]
    extent = [freq[0], freq[-1], jul_date[0], jul_date[-1]]

    plt.subplot(121)
    plt.imshow(vis1.real, extent=extent, origin='lower', aspect='auto')
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(vis1.imag, extent=extent, origin='lower', aspect='auto')
    plt.colorbar()
    if pol is None:
        fig_name = '%s_%d_%d.png' % (fig_prefix, bl[0], bl[1])
    else:
        fig_name = '%s_%d_%d_%s.png' % (fig_prefix, bl[0], bl[1], pol)
    fig_name = output_path(fig_name)
    plt.savefig(fig_name)

    return vis


class PlotRawTimestream(tod_task.SingleRawTimestream):
    """Waterfall plot for RawTimestream."""

    params_init = {
                    'bl_incl': 'all', # or a list of include (bl1, bl2)
                    'bl_excl': [],
                    'flag_ns': False,
                    'fig_name': 'vis',
                  }

    prefix = 'prt_'

    def process(self, rt):
        bl_incl = self.params['bl_incl']
        bl_excl = self.params['bl_excl']
        flag_ns = self.params['flag_ns']
        fig_name = self.params['fig_name']
        rt.bl_data_operate(plot, full_data=True, keep_dist_axis=False, bl_incl=bl_incl, bl_excl=bl_excl, fig_name=fig_name, flag_ns=flag_ns)
        rt.add_history(self.history)

        return rt


class PlotTimestream(tod_task.SingleTimestream):
    """Waterfall plot for Timestream."""

    params_init = {
                    'bl_incl': 'all', # or a list of include (bl1, bl2)
                    'bl_excl': [],
                    'flag_ns': False,
                    'fig_name': 'vis',
                  }

    prefix = 'pts_'

    def process(self, ts):
        bl_incl = self.params['bl_incl']
        bl_excl = self.params['bl_excl']
        flag_ns = self.params['flag_ns']
        fig_name = self.params['fig_name']
        ts.pol_and_bl_data_operate(plot, full_data=True, keep_dist_axis=False, bl_incl=bl_incl, bl_excl=bl_excl, fig_name=fig_name, flag_ns=flag_ns)
        ts.add_history(self.history)

        return ts
