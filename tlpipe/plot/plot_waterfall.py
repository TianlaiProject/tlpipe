"""Plot waterfall images."""

import os
from tlpipe.timestream import tod_task
from tlpipe.utils.path_util import output_path


def plot(vis, li, gi, bl, **kwargs):

    bl = tuple(bl)
    bl_incl = kwargs.get('bl_incl', 'all')
    bl_excl = kwargs.get('bl_excl', [])
    fig_name = kwargs.get('fig_name', 'vis')

    if bl_incl != 'all':
        bl1 = set(bl)
        bl_incl = [ {f1, f2} for (f1, f2) in bl_incl ]
        bl_excl = [ {f1, f2} for (f1, f2) in bl_excl ]
        if (not bl1 in bl_incl) or (bl1 in bl_excl):
            return vis

    import matplotlib.pyplot as plt

    plt.figure()
    plt.subplot(121)
    plt.imshow(vis.real, origin='lower', aspect='auto')
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(vis.imag, origin='lower', aspect='auto')
    plt.colorbar()
    fig_name += ('_%d_%d.png' % bl)
    fig_name = output_path(fig_name)
    fig_dir = os.path.dirname(fig_name)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    plt.savefig(fig_name)

    return vis


class PlotRawTimestream(tod_task.SingleRawTimestream):
    """Waterfall plot for RawTimestream."""

    params_init = {
                    'bl_incl': 'all', # or a list of include (bl1, bl2)
                    'bl_excl': [],
                    'fig_name': 'vis',
                  }

    prefix = 'prt_'

    def process(self, rt):
        bl_incl = self.params['bl_incl']
        bl_excl = self.params['bl_excl']
        fig_name = self.params['fig_name']
        rt.bl_data_operate(plot, full_data=True, keep_dist_axis=False, bl_incl=bl_incl, bl_excl=bl_excl, fig_name=fig_name)
        rt.add_history(self.history)

        rt.info()

        return rt
