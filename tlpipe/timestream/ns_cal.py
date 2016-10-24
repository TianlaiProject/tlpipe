"""Relative phase calibration using the noise source signal."""

import os
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import tod_task
from tlpipe.utils.path_util import output_path
import tlpipe.plot
import matplotlib.pyplot as plt


def cal(vis, vis_mask, li, gi, fbl, rt, **kwargs):

    if np.prod(vis.shape) == 0 :
        return vis, vis_mask

    nt = vis.shape[0]
    on_time = rt['ns_on'].attrs['on_time']
    num_mean = kwargs.get('num_mean', 5)
    num_mean = min(num_mean, on_time-2)
    if num_mean <= 0:
        raise RuntimeError('Do not have enough noise on time samples to do the ns_cal')
    plot_phs = kwargs.get('plot_phs', False)
    fig_prefix = kwargs.get('fig_name', 'phs_changke')
    iteration = kwargs.get('iteration', 0)

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
        fig_name = output_path(fig_name, iteration=iteration)
        plt.savefig(fig_name)
        plt.close()

    vis = vis * np.exp(-1.0J * all_phase)

    return vis, vis_mask


class NsCal(tod_task.IterRawTimestream):
    """Relative phase calibration using the noise source signal."""

    params_init = {
                    'num_mean': 5, # use the mean of num_mean signals
                    'plot_phs': False, # plot the phase change
                    'fig_name': 'phs_change',
                  }

    prefix = 'nc_'

    def process(self, rt):

        num_mean = self.params['num_mean']
        plot_phs = self.params['plot_phs']
        fig_name = self.params['fig_name']

        if not 'ns_on' in rt.iterkeys():
            raise RuntimeError('No noise source info, can not do noise source calibration')

        rt.freq_and_bl_data_operate(cal, full_data=True, num_mean=num_mean, plot_phs=plot_phs, fig_name=fig_name, iteration=self.iteration)

        rt.add_history(self.history)

        return rt
