"""Plot time or frequency integral."""

import numpy as np
from tlpipe.timestream import tod_task
from tlpipe.utils.path_util import output_path
import matplotlib.pyplot as plt


def plot(vis, vis_mask, li, gi, bl, obj, **kwargs):

    if isinstance(bl, tuple): # for Timestream
        pol = bl[0]
        bl = tuple(bl[1])
    else: # for RawTimestream
        pol = None
        bl = tuple(bl)
    integral = kwargs.get('integral', 'time')
    bl_incl = kwargs.get('bl_incl', 'all')
    bl_excl = kwargs.get('bl_excl', [])
    flag_mask = kwargs.get('flag_mask', False)
    flag_ns = kwargs.get('flag_ns', True)
    fig_prefix = kwargs.get('fig_name', 'int')
    iteration= kwargs.get('iteration', 0)

    if bl_incl != 'all':
        bl1 = set(bl)
        bl_incl = [ {f1, f2} for (f1, f2) in bl_incl ]
        bl_excl = [ {f1, f2} for (f1, f2) in bl_excl ]
        if (not bl1 in bl_incl) or (bl1 in bl_excl):
            return vis

    if flag_mask:
        vis1 = np.ma.array(vis, mask=vis_mask)
    elif flag_ns:
        vis1 = vis.copy()
        on = np.where(obj['ns_on'][:])[0]
        vis1[on] = complex(np.nan, np.nan)
    else:
        vis1 = vis

    if integral == 'time':
        ax_val = obj.freq[:]
        vis1 = np.ma.mean(np.ma.masked_invalid(vis1), axis=0)
        xlabel = r'$\nu$ / MHz'
    elif integral == 'freq':
        ax_val = obj.time[:]
        vis1 = np.ma.mean(np.ma.masked_invalid(vis1), axis=1)
        xlabel = r'$t$ / Julian Date'
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
    axarr[2].set_xlabel(xlabel)

    fig_name = '%s_%s_%d_%d.png' % (fig_prefix, integral, bl[0], bl[1])
    fig_name = output_path(fig_name, iteration=iteration)
    plt.savefig(fig_name)
    plt.close()

    return vis, vis_mask


class Plot(tod_task.IterRawTimestream):
    """Plot time or frequency integral."""

    params_init = {
                    'integral': 'time', # or 'freq'
                    'bl_incl': 'all', # or a list of include (bl1, bl2)
                    'bl_excl': [],
                    'flag_mask': True,
                    'flag_ns': True,
                    'fig_name': 'int',
                  }

    prefix = 'pit_'

    def process(self, rt):
        integral = self.params['integral']
        bl_incl = self.params['bl_incl']
        bl_excl = self.params['bl_excl']
        flag_mask = self.params['flag_mask']
        flag_ns = self.params['flag_ns']
        fig_name = self.params['fig_name']

        rt.bl_data_operate(plot, full_data=True, keep_dist_axis=False, integral=integral, bl_incl=bl_incl, bl_excl=bl_excl, fig_name=fig_name, iteration=self.iteration, flag_mask=flag_mask, flag_ns=flag_ns)

        rt.add_history(self.history)

        return rt
