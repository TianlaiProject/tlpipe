"""Relative phase calibration using the noise source signal.

Inheritance diagram
-------------------

.. inheritance-diagram:: NsCal
   :parts: 2

"""

import os
from datetime import datetime
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import tod_task
from raw_timestream import RawTimestream
from tlpipe.utils.path_util import output_path
import tlpipe.plot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator, AutoMinorLocator


class NsCal(tod_task.TaskTimestream):
    """Relative phase calibration using the noise source signal.

    The noise source can be viewed as a near-field source, its visibility
    can be expressed as

    .. math:: V_{ij}^{\\text{ns}} = C \\cdot e^{i k (r_{i} - r_{j})}

    where :math:`C` is a real constant.

    .. math::

        V_{ij}^{\\text{on}} &= G_{ij} (V_{ij}^{\\text{sky}} + V_{ij}^{\\text{ns}} + n_{ij}) \\\\
        V_{ij}^{\\text{off}} &= G_{ij} (V_{ij}^{\\text{sky}} + n_{ij})

    where :math:`G_{ij}` is the gain of baseline :math:`i,j`.

    .. math::

        V_{ij}^{\\text{on}} - V_{ij}^{\\text{off}} &= G_{ij} V_{ij}^{\\text{ns}} \\\\
                                       &=|G_{ij}| e^{i k \\Delta L} C \\cdot e^{i k (r_{i} - r_{j})} \\\\
                                       & = C |G_{ij}| e^{i k (\\Delta L + (r_{i} - r_{j}))}

    where :math:`\\Delta L` is the equivalent cable length.

    .. math:: \\text{Arg}(V_{ij}^{\\text{on}} - V_{ij}^{\\text{off}}) = k (\\Delta L + (r_{i} - r_{j})) = k \\Delta L + const.

    To compensate for the relative phase change (due to :math:`\\Delta L`) of the
    visibility, we can do

    .. math:: V_{ij}^{\\text{rel-cal}} = e^{-i \\; \\text{Arg}(V_{ij}^{\\text{on}} - V_{ij}^{\\text{off}})} \\, V_{ij}

    .. note::
        Note there is still an unknown (constant) phase factor to be determined in
        :math:`V_{ij}^{\\text{rel-cal}}`, which may be done by absolute calibration.

    """

    params_init = {
                    'num_mean': 5, # use the mean of num_mean signals
                    'phs_only': True, # phase cal only
                    'plot_gain': False, # plot the gain change
                    'fig_name': 'ns_cal/gain_change',
                    'bl_incl': 'all', # or a list of include (bl1, bl2)
                    'bl_excl': [],
                    'freq_incl': 'all', # or a list of include freq idx
                    'freq_excl': [],
                    'rotate_xdate': False, # True to rotate xaxis date ticks, else half the number of date ticks
                    'feed_no': False, # True to use feed number (true baseline) else use channel no
                  }

    prefix = 'nc_'

    def process(self, rt):

        assert isinstance(rt, RawTimestream), '%s only works for RawTimestream object currently' % self.__class__.__name__

        if not 'ns_on' in rt.iterkeys():
            raise RuntimeError('No noise source info, can not do noise source calibration')

        rt.redistribute('time')

        bl_incl = self.params['bl_incl']
        bl_excl = self.params['bl_excl']
        freq_incl = self.params['freq_incl']
        freq_excl = self.params['freq_excl']

        if bl_incl == 'all':
            bls_plt = [ tuple(bl) for bl in rt.bl ]
        else:
            bls_plt = [ bl for bl in bl_incl if not bl in bl_excl ]

        if freq_incl == 'all':
            freq_plt = range(rt.freq.shape[0])
        else:
            freq_plt = [ fi for fi in freq_incl if not fi in freq_excl ]

        rt.freq_and_bl_data_operate(self.cal, full_data=True, keep_dist_axis=False, bls_plt=bls_plt, freq_plt=freq_plt)

        return super(NsCal, self).process(rt)

    def cal(self, vis, vis_mask, li, gi, fbl, rt, **kwargs):
        """Function that does the actual cal."""

        num_mean = self.params['num_mean']
        phs_only = self.params['phs_only']
        plot_gain = self.params['plot_gain']
        fig_prefix = self.params['fig_name']
        rotate_xdate = self.params['rotate_xdate']
        feed_no = self.params['feed_no']
        tag_output_iter = self.params['tag_output_iter']
        iteration = self.iteration
        bls_plt = kwargs['bls_plt']
        freq_plt = kwargs['freq_plt']

        if np.prod(vis.shape) == 0 :
            return

        fi = gi[0] # freq idx for this cal
        bl = tuple(fbl[1]) # bl for this cal

        nt = vis.shape[0]
        on_time = rt['ns_on'].attrs['on_time']
        off_time = rt['ns_on'].attrs['off_time']
        period = rt['ns_on'].attrs['period']
        num_mean = min(num_mean, on_time-2)
        if num_mean <= 0:
            raise RuntimeError('Do not have enough noise on time samples to do the ns_cal')
        ns_on = rt['ns_on'][:]
        ns_on = np.where(ns_on, 1, 0)
        diff_ns = np.diff(ns_on)
        inds = np.where(diff_ns==1)[0]
        # if inds[0]-num_mean < 0:
        if inds[0]-1 < 0: # no off data in the beginning to use
            inds = inds[1:]
        # if inds[-1]+num_mean+1 > len(ns_on)-1:
        if inds[-1]+2 > nt-1: # no on data in the end to use
            inds = inds[:-1]

        valid_inds = []
        phase = []
        if not phs_only:
            amp = []
        for ind in inds:
            # drop the first and the last ind, as it may lead to exceptional vals
            if ind == inds[0] or ind == inds[-1]:
                continue

            lower = ind - num_mean
            off_sec = np.ma.array(vis[lower:ind], mask=vis_mask[lower:ind])
            # if off_sec.count() > 0: # not all data in this section are masked
            if off_sec.count() >= max(2, num_mean/2): # more valid sample to make stable
                upper = ind + 2 + num_mean
                valid_inds.append(ind)
                diff = np.mean(vis[ind+2:upper]) - np.ma.mean(off_sec)
                phase.append( np.angle(diff) ) # in radians
                if not phs_only:
                    amp.append( np.abs(diff) )


        # not enough valid data to do the ns_cal
        num_valid = len(valid_inds)
        if num_valid <= 3:
            vis_mask[:] = True # mask the vis as no ns_cal has done
            return

        phase = np.unwrap(phase) # unwrap 2pi discontinuity
        if not phs_only:
            amp = np.array(amp) / np.median(amp) # normalize
        # split valid_inds into consecutive chunks
        intervals = [0] + (np.where(np.diff(valid_inds) > 3 * period)[0] + 1).tolist() + [num_valid]
        itp_inds = []
        itp_phase = []
        if not phs_only:
            itp_amp = []
        for i in range(len(intervals) -1):
            this_chunk = valid_inds[intervals[i]:intervals[i+1]]
            if len(this_chunk) > 3:
                itp_inds.append(this_chunk)
                itp_phase.append(phase[intervals[i]:intervals[i+1]])
                if not phs_only:
                    itp_amp.append(amp[intervals[i]:intervals[i+1]])

        # if no such chunk, mask all the data
        num_itp = len(itp_inds)
        if num_itp == 0:
            vis_mask[:] = True

        # get itp pairs
        itp_pairs = []
        for it in itp_inds:
            itp_pairs.append((max(0, it[0]-off_time), min(nt, it[-1]+period)))

        # get mask pairs
        mask_pairs = []
        for i in range(num_itp):
            if i == 0:
                mask_pairs.append((0, itp_pairs[i][0]))
            if i == num_itp - 1:
                mask_pairs.append((itp_pairs[i][-1], nt))
            else:
                mask_pairs.append((itp_pairs[i][-1], itp_pairs[i+1][0]))

        # set maskd for inds in mask_pairs
        for mp1, mp2 in mask_pairs:
            vis_mask[mp1:mp2] = True

        # interpolate for inds in itp_inds
        all_phase = np.array([np.nan]*nt)
        for this_inds, this_phase, (i1, i2) in zip(itp_inds, itp_phase, itp_pairs):
            # no need to interpolate for auto-correlation
            if bl[0] == bl[1]:
                all_phase[i1:i2] = 0
            else:
                f = InterpolatedUnivariateSpline(this_inds, this_phase)
                this_itp_phs = f(np.arange(i1, i2))
                # # make the interpolated values in the appropriate range
                # this_itp_phs = np.where(this_itp_phs>np.pi, np.pi, this_itp_phs)
                # this_itp_phs = np.where(this_itp_phs<-np.pi, np.pi, this_itp_phs)
                all_phase[i1:i2] = this_itp_phs
                # do phase cal for this range of inds
                vis[i1:i2] = vis[i1:i2] * np.exp(-1.0J * this_itp_phs)

        if not phs_only:
            all_amp = np.array([np.nan]*nt)
            for this_inds, this_amp, (i1, i2) in zip(itp_inds, itp_amp, itp_pairs):
                f = InterpolatedUnivariateSpline(this_inds, this_amp)
                this_itp_amp = f(np.arange(i1, i2))
                all_amp[i1:i2] = this_itp_amp
                # do amp cal for this range of inds
                vis[i1:i2] = vis[i1:i2] / this_itp_amp

        if plot_gain and (bl in bls_plt and fi in freq_plt):
            plt.figure()
            if phs_only:
                fig, ax = plt.subplots()
            else:
                fig, ax = plt.subplots(2, sharex=True)
            ax_val = np.array([ datetime.fromtimestamp(sec) for sec in rt['sec1970'][:] ])
            xlabel = '%s' % ax_val[0].date()
            ax_val = mdates.date2num(ax_val)
            if phs_only:
                ax.plot(ax_val, all_phase)
                ax.plot(ax_val[valid_inds], phase, 'ro')
                ax1 = ax
            else:
                ax[0].plot(ax_val, all_amp)
                ax[0].plot(ax_val[valid_inds], amp, 'ro')
                ax[0].set_ylabel(r'$\Delta |g|$')
                ax[1].plot(ax_val, all_phase)
                ax[1].plot(ax_val[valid_inds], phase, 'ro')
                ax1 = ax[1]
            ax1.xaxis_date()
            date_format = mdates.DateFormatter('%H:%M')
            ax1.xaxis.set_major_formatter(date_format)
            if rotate_xdate:
                # set the x-axis tick labels to diagonal so it fits better
                fig.autofmt_xdate()
            else:
                # reduce the number of tick locators
                locator = MaxNLocator(nbins=6)
                ax1.xaxis.set_major_locator(locator)
                ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax1.set_xlabel(xlabel)
            ax1.set_ylabel(r'$\Delta \phi$ / radian')

            if feed_no:
                pol = rt['bl_pol'].local_data[li[1]]
                bl = tuple(rt['true_blorder'].local_data[li[1]])
                fig_name = '%s_%f_%d_%d_%s.png' % (fig_prefix, fbl[0], bl[0], bl[1], rt.pol_dict[pol])
            else:
                fig_name = '%s_%f_%d_%d.png' % (fig_prefix, fbl[0], fbl[1][0], fbl[1][1])
            if tag_output_iter:
                fig_name = output_path(fig_name, iteration=iteration)
            else:
                fig_name = output_path(fig_name)
            plt.savefig(fig_name)
            plt.close()
