import h5py
import numpy as np
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tlpipe.utils.path_util import output_path, input_path
from tlpipe.pipeline.pipeline import FileIterBase
from tlpipe.mcmc import get_dist

_c_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
          "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f", 'w']
_l_list = ['o:', 's--', '.--']


class PlotPS(FileIterBase):

    params_init = {
            'labels': [],
            'label_title': '',
            'vmin'  : None,
            'vmax'  : None,
            'fig_name' : 'ps/tcorr',
            'c_indx' : None,
            'l_indx' : None,
            'mfc_indx' : None,
            'fitting_path' : None,
            'fitting_comb' : False,
            }

    prefix = 'pps_'

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        self.fig  = plt.figure(figsize=(8, 4))
        axhh = self.fig.add_axes([0.11, 0.13, 0.41, 0.78])
        axvv = self.fig.add_axes([0.53, 0.13, 0.41, 0.78])
        self.axes = [axhh, axvv]

        self.xmin = 1.e20
        self.xmax = -1.e20
        self.legend_list = []

        super(PlotPS, self).__init__(parameter_file_or_dict, feedback)

    def read_input(self):

        return h5py.File(input_path(self.input_files[self.iteration]))

    def write_output(self, output):

        vmin, vmax = output

        xmax = self.xmax
        xmin = self.xmin

        #fig_name = self.params['fig_name']
        fig_name = self.output_file
        fig_name = output_path(fig_name)
        print '--', fig_name

        axhh, axvv = self.axes
        axhh.set_xlabel(r'$f\,[{\rm Hz}]$')
        axhh.set_ylabel(r'${\rm PSD}(f)$')
        axhh.set_title(r'${\rm HH}$')
        axhh.legend(handles=self.legend_list, frameon=False, 
                title=self.params['label_title'])
        axhh.set_xlim(xmin=xmin, xmax=xmax)
        axhh.set_ylim(ymin=vmin, ymax=vmax)
        axhh.loglog()
        #axhh.semilogx()
        #axhh.set_xticklabels([])
        axhh.minorticks_on()
        axhh.tick_params(length=4, width=1, direction='in')
        axhh.tick_params(which='minor', length=2, width=1, direction='in')

        axvv.set_xlabel(r'$f\,[{\rm Hz}]$')
        #axvv.set_ylabel(r'${\rm frequency\, [GHz]\, VV}$')
        axvv.set_title(r'${\rm VV}$')
        axvv.set_xlim(xmin=xmin, xmax=xmax)
        axvv.set_ylim(ymin=vmin, ymax=vmax)
        axvv.loglog()
        #axvv.semilogx()
        axvv.set_yticklabels([])
        axvv.minorticks_on()
        axvv.tick_params(length=4, width=1, direction='in')
        axvv.tick_params(which='minor', length=2, width=1, direction='in')

        self.fig.savefig(fig_name + '.png', dpi=500)
        self.fig.clf()
        plt.close()


    def process(self, input):

        labels   = self.params['labels']
        vmin     = self.params['vmin']
        vmax     = self.params['vmax']
        #fig_name = self.params['fig_name']
        #fig_name = output_path(fig_name)

        c_indx = self.params['c_indx']
        l_indx = self.params['l_indx']
        mfc_indx = self.params['mfc_indx']
        if c_indx is None:
            c_indx = range(self.iter_num)
        if l_indx is None:
            l_indx = np.zeros(self.iter_num).astype('int')
        if mfc_indx is None:
            mfc_indx = c_indx

        axhh, axvv = self.axes

        #for i in range(len(input)):
        i = self.iteration
        fh = input
        tcorr_ps = fh['tcorr_ps'][:]
        tcorr_bc = fh['tcorr_bc'][:]

        shift = (((tcorr_bc[1] / tcorr_bc[0]) ** 0.5)**(1./float(self.iter_num)))
        shift = shift ** (np.arange(self.iter_num) - (self.iter_num - 1) * 0.5)
        shift = shift[i]
        tcorr_bc *= shift

        t = tcorr_bc

        c = _c_list[c_indx[i]]
        fmt = _l_list[l_indx[i]]
        mfc = _c_list[mfc_indx[i]]

        mean = np.mean(tcorr_ps, axis=1)
        erro = np.std(tcorr_ps, axis=1)
        upper = mean + erro
        lower = mean - erro
        if vmin is None:
            vmin = lower[lower>0].min()
        lower[lower<=0] = vmin

        #upper = upper - mean
        #lower = mean - lower
        #errors = np.concatenate([lower[None, :, :], upper[None, :, :]], axis=0)

        pp = mean[:,0] > 0
        axhh.fill_between(tcorr_bc[pp],upper[pp, 0],lower[pp, 0], facecolor=c, alpha=0.3)
        axhh.plot(tcorr_bc[pp], mean[pp, 0], fmt[0] ,color=c,linewidth=1.5,#label=labels[i],
                mec=c, mfc=mfc, ms=4)
        #axhh.errorbar(tcorr_bc[pp], mean[pp, 0], errors[:, pp, 0],
        #        c=c, marker='o', mfc=mfc, mec=c, ms=4, mew=1, lw=1,
        #        ecolor=c, elinewidth=1.5, capsize=0, capthick=0)
        pp = mean[:,1] > 0
        axvv.fill_between(tcorr_bc[pp],upper[pp, 1],lower[pp, 1], facecolor=c,alpha=0.3)
        axvv.plot(tcorr_bc[pp], mean[pp, 1], fmt[0], color=c, linewidth=1.5,
                mec=c, mfc=mfc, ms=4)
        #axvv.errorbar(tcorr_bc[pp], mean[pp, 1], errors[:, pp, 1],
        #        c=c, marker='o', mfc=mfc, mec=c, ms=4, mew=1, lw=1,
        #        ecolor=c, elinewidth=1.5, capsize=0, capthick=0)

        if not self.params['fitting_path'] is None:

            x = np.logspace(np.log10(tcorr_bc.min()), np.log10(tcorr_bc.max()), 200)

            base_path = self.params['fitting_path']
            base_name, ext = os.path.basename(self.input_files[self.iteration]).split('.')

            # for HH
            like_file = input_path(base_path + base_name + '_HH/plot_data.likestats')
            self._plot_thpsd_(axhh, x, like_file, fmt=fmt[1:], c=c)

            # for VV
            like_file = input_path(base_path + base_name + '_VV/plot_data.likestats')
            self._plot_thpsd_(axvv, x, like_file, fmt=fmt[1:], c=c)

            if (labels[i] is not None) and self.params['fitting_comb']:
                base_name = base_name.replace(base_name.split('_')[0], 'combineded')
                like_file = input_path(base_path + base_name + '_HH/plot_data.likestats')
                self._plot_thpsd_(axhh, x, like_file, fmt='-', c=c, lw=1.5)
                like_file = input_path(base_path + base_name + '_VV/plot_data.likestats')
                self._plot_thpsd_(axvv, x, like_file, fmt='-', c=c, lw=1.5)

        if labels[i] is not None:
            self.legend_list.append(mpatches.Patch(color=c, label=labels[i]))

        if self.xmin > t.min()/1.2: self.xmin = t.min()/1.2
        if self.xmax < t.max()*1.2: self.xmax = t.max()*1.2

        if i == self.iter_num - 1:
            return vmin, vmax

    def _plot_thpsd_(self, ax, x, like_file, fmt='-', c='k', lw=1.0):

        psd_f = lambda f, _amp, _fk, _alpha: _amp * (1. + (_fk/f) ** alpha)

        stats = get_dist.load_param_dict(like_file)
        
        amp   = 10. ** stats['amp'][0]
        fk    = 10. ** stats['fk'][0]
        alpha = stats['alpha'][0]
        
        amp_m   = 10. ** stats['amp'][1]
        fk_m    = 10. ** stats['fk'][1]
        alpha_m = stats['alpha'][1]
        
        amp_l   = amp_m - 10. ** stats['amp'][3]
        fk_l    = fk_m  - 10. ** stats['fk'][3]
        alpha_l = alpha_m - stats['alpha'][3]
        
        amp_u   = 10. ** stats['amp'][4] - amp_m  
        fk_u    = 10. ** stats['fk'][4]  - fk_m
        alpha_u = stats['alpha'][4] - alpha_m
        
        ax.plot(x, psd_f(x, amp, fk, alpha), fmt, color=c, linewidth=lw)





