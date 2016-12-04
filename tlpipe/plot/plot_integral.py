"""Plot time or frequency integral.

Inheritance diagram
-------------------

.. inheritance-diagram:: Plot
   :parts: 2

"""

import numpy as np
from tlpipe.timestream import tod_task
from tlpipe.timestream.raw_timestream import RawTimestream
from tlpipe.timestream.timestream import Timestream
from tlpipe.utils.path_util import output_path
import matplotlib.pyplot as plt


def plot(vis, vis_mask, li, gi, bl, ts, **kwargs):

    integral = kwargs.get('integral', 'time')
    bl_incl = kwargs.get('bl_incl', 'all')
    bl_excl = kwargs.get('bl_excl', [])
    flag_mask = kwargs.get('flag_mask', False)
    flag_ns = kwargs.get('flag_ns', True)
    fig_prefix = kwargs.get('fig_name', 'int')
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
        vis1[on] = complex(np.nan, np.nan)
    else:
        vis1 = vis

    if integral == 'time':
        ax_val = ts.freq[:]
        vis1 = np.ma.mean(np.ma.masked_invalid(vis1), axis=0)
        xlabel = r'$\nu$ / MHz'
    elif integral == 'freq':
        ax_val = ts.time[:]
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

    if pol is None:
        fig_name = '%s_%s_%d_%d.png' % (fig_prefix, integral, bl[0], bl[1])
    else:
        fig_name = '%s_%s_%d_%d_%s.png' % (fig_prefix, integral, bl[0], bl[1], pol)
    if tag_output_iter:
        fig_name = output_path(fig_name, iteration=iteration)
    else:
        fig_name = output_path(fig_name)
    plt.savefig(fig_name)
    plt.close()

    return vis, vis_mask


class Plot(tod_task.TaskTimestream):
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

    def process(self, ts):

        integral = self.params['integral']
        bl_incl = self.params['bl_incl']
        bl_excl = self.params['bl_excl']
        flag_mask = self.params['flag_mask']
        flag_ns = self.params['flag_ns']
        fig_name = self.params['fig_name']
        tag_output_iter = self.params['tag_output_iter']

        ts.redistribute('baseline')

        if isinstance(ts, RawTimestream):
            func = ts.bl_data_operate
        elif isinstance(ts, Timestream):
            func = ts.pol_and_bl_data_operate

        func(plot, full_data=True, keep_dist_axis=False, integral=integral, bl_incl=bl_incl, bl_excl=bl_excl, fig_name=fig_name, iteration=self.iteration, tag_output_iter=tag_output_iter, flag_mask=flag_mask, flag_ns=flag_ns)

        ts.add_history(self.history)

        return ts
