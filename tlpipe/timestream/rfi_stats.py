"""RFI statistics.

Inheritance diagram
-------------------

.. inheritance-diagram:: Stats
   :parts: 2

"""

from datetime import datetime, timedelta
import numpy as np
import timestream_task
from tlpipe.utils.path_util import output_path
import tlpipe.plot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from caput import mpiutil


class Stats(timestream_task.TimestreamTask):
    """RFI statistics.

    Analysis of RFI distributions along time and frequency.

    """

    params_init = {
                    'excl_auto': False, # exclude auto-correclation
                    'plot_stats': True, # plot RFI statistics
                    'fig_name': 'stats/stats',
                    'rotate_xdate': False, # True to rotate xaxis date ticks, else half the number of date ticks
                  }

    prefix = 'rs_'

    def process(self, ts):

        excl_auto = self.params['excl_auto']
        plot_stats = self.params['plot_stats']
        fig_prefix = self.params['fig_name']
        rotate_xdate = self.params['rotate_xdate']
        tag_output_iter = self.params['tag_output_iter']

        ts.redistribute('baseline')

        if ts.local_vis_mask.ndim == 3: # RawTimestream
            if excl_auto:
                bl = ts.local_bl
                vis_mask = ts.local_vis_mask[:, :, bl[:, 0] != bl[:, 1]].copy()
            else:
                vis_mask = ts.local_vis_mask.copy()
            nt, nf, lnb = vis_mask.shape
        elif ts.local_vis_mask.ndim == 4: # Timestream
            # suppose masks are the same for all 4 pols
            if excl_auto:
                bl = ts.local_bl
                vis_mask = ts.local_vis_mask[:, :, 0, bl[:, 0] != bl[:, 1]].copy()
            else:
                vis_mask = ts.local_vis_mask[:, :, 0].copy()
            nt, nf, lnb = vis_mask.shape
        else:
            raise RuntimeError('Incorrect vis_mask shape %s' % ts.local_vis_mask.shape)

        # total number of bl
        nb = mpiutil.allreduce(lnb, comm=ts.comm)

        # un-mask ns-on positions
        if 'ns_on' in ts.iterkeys():
            vis_mask[ts['ns_on'][:]] = False

        # statistics along time axis
        time_mask = np.sum(vis_mask, axis=(1, 2)).reshape(-1, 1)
        # gather local array to rank0
        time_mask = mpiutil.gather_array(time_mask, axis=1, root=0, comm=ts.comm)
        if mpiutil.rank0:
            time_mask = np.sum(time_mask, axis=1)

        # statistics along time axis
        freq_mask = np.sum(vis_mask, axis=(0, 2)).reshape(-1, 1)
        # gather local array to rank0
        freq_mask = mpiutil.gather_array(freq_mask, axis=1, root=0, comm=ts.comm)
        if mpiutil.rank0:
            freq_mask = np.sum(freq_mask, axis=1)

        if plot_stats and mpiutil.rank0:
            time_fig_name = '%s_%s.png' % (fig_prefix, 'time')
            if tag_output_iter:
                time_fig_name = output_path(time_fig_name, iteration=self.iteration)
            else:
                time_fig_name = output_path(time_fig_name)

            # plot time_mask
            plt.figure()
            fig, ax = plt.subplots()
            x_vals = np.array([ (datetime.utcfromtimestamp(s) + timedelta(hours=8)) for s in ts['sec1970'][:] ])
            xlabel = '%s' % x_vals[0].date()
            x_vals = mdates.date2num(x_vals)
            ax.plot(x_vals, 100*time_mask/np.float(nf*nb))
            ax.xaxis_date()
            date_format = mdates.DateFormatter('%H:%M')
            ax.xaxis.set_major_formatter(date_format)
            if rotate_xdate:
                # set the x-axis tick labels to diagonal so it fits better
                fig.autofmt_xdate()
            else:
                # reduce the number of tick locators
                locator = MaxNLocator(nbins=6)
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_minor_locator(AutoMinorLocator(2))

            ax.set_xlabel(xlabel)
            ax.set_ylabel(r'RFI (%)')
            plt.savefig(time_fig_name)
            plt.close()

            freq_fig_name = '%s_%s.png' % (fig_prefix, 'freq')
            if tag_output_iter:
                freq_fig_name = output_path(freq_fig_name, iteration=self.iteration)
            else:
                freq_fig_name = output_path(freq_fig_name)

            # plot freq_mask
            plt.figure()
            plt.plot(ts.freq[:], 100*freq_mask/np.float(nt*nb))
            plt.xlabel(r'$\nu$ / MHz')
            plt.ylabel(r'RFI (%)')
            plt.savefig(freq_fig_name)
            plt.close()

        return super(Stats, self).process(ts)
