"""Relative phase calibration using the noise source signal."""

import os
import numpy as np
from scipy import interpolate
import tod_task
from tlpipe.utils.path_util import output_path


def cal(vis, li, gi, fbl, rt, **kwargs):

    if np.prod(vis.shape) == 0 :
        return vis

    num_mean = kwargs.get('num_mean', 5)
    plot_phs = kwargs.get('plot_phs', False)
    fig_prefix = kwargs.get('fig_name', 'phs_changke')

    ns_on = rt['ns_on'][:]
    ns_on = np.where(ns_on, 1, 0)
    diff_ns = np.diff(ns_on)
    inds = np.where(diff_ns==1)[0]
    if inds[0]-num_mean < 0:
        inds = inds[1:]
    if inds[1]+num_mean+1 > len(ns_on)-1:
        inds = inds[:-1]

    phase = []
    for ind in inds:
        phase.append( np.angle(np.mean(vis[ind+2:ind+2+num_mean]) - np.mean(vis[ind-num_mean:ind])) ) # in radians

    phase = np.unwrap(phase) # unwrap 2pi discontinuity

    # f = interpolate.interp1d(inds, phase, kind='cubic', bounds_error=False, assume_sorted=True) # no assume_sorted in older version
    f = interpolate.interp1d(inds, phase, kind='cubic', bounds_error=False)
    all_phase = f(np.arange(vis.shape[0]))
    not_none_inds = np.where(np.logical_not(np.isnan(all_phase)))[0]
    all_phase[:not_none_inds[0]] = all_phase[not_none_inds[0]]
    all_phase[not_none_inds[-1]+1:] = all_phase[not_none_inds[-1]]

    if plot_phs:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(all_phase)
        plt.plot(inds, phase, 'ro')
        fig_name = '%s_%f_%d_%d.png' % (fig_prefix, fbl[0], fbl[1][0], fbl[1][1])
        fig_name = output_path(fig_name)
        fig_dir = os.path.dirname(fig_name)
        try:
            os.makedirs(fig_dir)
        except OSError:
            pass
        plt.savefig(fig_name)

    vis = vis * np.exp(-1.0J * all_phase)

    return vis


class NsCal(tod_task.SingleRawTimestream):
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

        rt.freq_and_bl_data_operate(cal, full_data=True, num_mean=num_mean, plot_phs=plot_phs, fig_name=fig_name)

        rt.add_history(self.history)

        return rt
