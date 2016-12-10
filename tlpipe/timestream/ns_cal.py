"""Relative phase calibration using the noise source signal.

Inheritance diagram
-------------------

.. inheritance-diagram:: NsCal
   :parts: 2

"""

import os
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import tod_task
from raw_timestream import RawTimestream
from tlpipe.utils.path_util import output_path
import tlpipe.plot
import matplotlib.pyplot as plt


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
                    'plot_phs': False, # plot the phase change
                    'fig_name': 'phs_change',
                  }

    prefix = 'nc_'

    def process(self, rt):

        assert isinstance(rt, RawTimestream), '%s only works for RawTimestream object currently' % self.__class__.__name__

        if not 'ns_on' in rt.iterkeys():
            raise RuntimeError('No noise source info, can not do noise source calibration')

        rt.freq_and_bl_data_operate(self.cal, full_data=True, keep_dist_axis=False)

        rt.add_history(self.history)

        return rt

    def cal(self, vis, vis_mask, li, gi, fbl, rt, **kwargs):
        """Function that does the actual cal."""

        num_mean = self.params['num_mean']
        plot_phs = self.params['plot_phs']
        fig_prefix = self.params['fig_name']
        tag_output_iter = self.params['tag_output_iter']
        iteration = self.iteration

        if np.prod(vis.shape) == 0 :
            return vis, vis_mask

        nt = vis.shape[0]
        on_time = rt['ns_on'].attrs['on_time']
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
        for ind in inds:
            if ind == inds[0]: # the first ind
                lower = max(0, ind-num_mean)
            else:
                lower = ind - num_mean
            off_sec = np.ma.array(vis[lower:ind], mask=vis_mask[lower:ind])
            if off_sec.count() > 0: # not all data in this section are masked
                valid_inds.append(ind)
                if ind == inds[-1]: # the last ind
                    upper = min(nt, ind+2+num_mean)
                else:
                    upper = ind + 2 + num_mean
                phase.append( np.angle(np.mean(vis[ind+2:upper]) - np.ma.mean(off_sec)) ) # in radians

        # not enough valid data to do the ns_cal
        if len(phase) <= 3:
            vis_mask[:] = True # mask the vis as no ns_cal has done
            return vis, vis_mask

        phase = np.unwrap(phase) # unwrap 2pi discontinuity

        f = InterpolatedUnivariateSpline(valid_inds, phase)
        all_phase = f(np.arange(nt))

        if plot_phs:
            plt.figure()
            time = rt.time[:]
            plt.plot(time, all_phase)
            plt.plot(time[valid_inds], phase, 'ro')
            plt.xlabel(r'$t$ / Julian Date')
            plt.ylabel(r'$\Delta \phi$ / radian')
            fig_name = '%s_%f_%d_%d.png' % (fig_prefix, fbl[0], fbl[1][0], fbl[1][1])
            if tag_output_iter:
                fig_name = output_path(fig_name, iteration=iteration)
            else:
                fig_name = output_path(fig_name)
            plt.savefig(fig_name)
            plt.close()

        vis = vis * np.exp(-1.0J * all_phase)

        return vis, vis_mask
