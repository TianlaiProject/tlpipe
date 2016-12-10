"""Plot waterfall images.

Inheritance diagram
-------------------

.. inheritance-diagram:: Plot
   :parts: 2

"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from tlpipe.timestream import tod_task
from tlpipe.timestream.raw_timestream import RawTimestream
from tlpipe.timestream.timestream import Timestream
from tlpipe.utils.path_util import output_path
import matplotlib.pyplot as plt


class Plot(tod_task.TaskTimestream):
    """Waterfall plot for Timestream.

    This task plots the waterfall (i.e., visibility as a function of time
    and frequency) of the visibility
    for each baseline (and also each polarization if the input data is a
    :class:`~tlpipe.timestream.timestream.Timestream` instead of a
    :class:`~tlpipe.timestream.raw_timestream.RawTimestream`).

    """

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

        ts.redistribute('baseline')

        if isinstance(ts, RawTimestream):
            func = ts.bl_data_operate
        elif isinstance(ts, Timestream):
            func = ts.pol_and_bl_data_operate

        func(self.plot, full_data=True, keep_dist_axis=False)

        ts.add_history(self.history)

        return ts

    def plot(self, vis, vis_mask, li, gi, bl, ts, **kwargs):
        """Function that does the actual plot work."""

        bl_incl = self.params['bl_incl']
        bl_excl = self.params['bl_excl']
        flag_mask = self.params['flag_mask']
        flag_ns = self.params['flag_ns']
        interpolate_ns = self.params['interpolate_ns']
        y_axis = self.params['y_axis']
        plot_abs = self.params['plot_abs']
        fig_prefix = self.params['fig_name']
        tag_output_iter = self.params['tag_output_iter']
        iteration = self.iteration

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
