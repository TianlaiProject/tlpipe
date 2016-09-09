"""Line RFI flagging."""

import os
import numpy as np
import tod_task
from sg_filter import savitzky_golay
from caput import mpiarray

class Flag(tod_task.IterRawTimestream):
    """Line RFI flagging."""

    params_init = {
                    'window_size': 11,
                    'sigma': 5.0,
                    'plot_fit': False, # plot the smoothing fit
                    'fig_name': 'fit',
                  }

    prefix = 'lf_'

    def process(self, rt):

        window_size = self.params['window_size']
        sigma = self.params['sigma']
        plot_fit = self.params['plot_fit']
        fig_prefix = self.params['fig_name']

        rt.redistribute('baseline')

        # time integration
        tm_vis = np.ma.mean(np.ma.masked_invalid(rt.local_vis[:]), axis=0)
        freq_mask = np.zeros(tm_vis.shape, dtype=bool)
        # iterate over local baselines
        freq = rt.freq[:]
        nfreq = len(freq)
        bl = rt.local_bl[:]
        window_size = min(nfreq/2, window_size)
        # ensure window_size is an odd number
        if window_size % 2 == 0:
            window_size += 1
        for lbi in range(tm_vis.shape[-1]):
            abs_vis = np.abs(tm_vis[:, lbi])
            smooth = savitzky_golay(abs_vis, window_size, 3)

            # flage RFI
            diff = abs_vis - smooth
            mean = np.mean(diff)
            std = np.std(diff)
            inds = np.where(np.abs(diff - mean) > sigma*std)[0]
            freq_mask[inds, lbi] = True

            if plot_fit:
                import tlpipe.plot
                import matplotlib.pyplot as plt
                from tlpipe.utils.path_util import output_path

                plt.figure()
                abs_vis1 = np.where(freq_mask[:, lbi], np.nan, abs_vis)
                plt.plot(freq, abs_vis1)
                plt.plot(freq, smooth)
                plt.xlabel(r'$\nu$ / MHz')
                fig_name = '%s_%d_%d.png' % (fig_prefix, bl[lbi][0], bl[lbi][1])
                fig_name = output_path(fig_name)
                plt.savefig(fig_name)
                plt.clf()

        # create a dataset for freq_mask
        freq_mask = mpiarray.MPIArray.wrap(freq_mask, axis=1)
        rt.create_freq_and_bl_ordered_dataset('freq_mask', freq_mask)

        # rt.info()

        rt.add_history(self.history)

        return rt
