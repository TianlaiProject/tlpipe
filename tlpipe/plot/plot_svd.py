import h5py
import numpy as np

from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tlpipe.utils.path_util import output_path, input_path
from tlpipe.pipeline.pipeline import FileIterBase

_c_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
          "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]

class PlotSVD(FileIterBase):

    params_init = {
            'mode_n'   : 6,
            'fig_name' : 'svd/svd',
            }

    prefix = 'psvd_'

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        self.fig  = plt.figure(figsize=(10, 6))
        axeg = self.fig.add_axes([0.08, 0.13, 0.15, 0.78])
        axhh = self.fig.add_axes([0.24, 0.53, 0.34, 0.38])
        axvv = self.fig.add_axes([0.24, 0.13, 0.34, 0.38])
        axhh2 = self.fig.add_axes([0.63, 0.53, 0.34, 0.38])
        axvv2 = self.fig.add_axes([0.63, 0.13, 0.34, 0.38])
        self.axes = [axeg, axhh, axvv, axhh2, axvv2]

        super(PlotSVD, self).__init__(parameter_file_or_dict, feedback)

    def read_input(self):

        print self.input_files[self.iteration]
        return h5py.File(input_path(self.input_files[self.iteration]))

    def write_output(self, output):

        mode_n = self.params['mode_n']

        xmin, xmax, fmin, fmax = output

        #fig_name = self.params['fig_name']
        fig_name = self.output_file
        fig_name = output_path(fig_name)
        print '--', fig_name

        axeg, axhh, axvv, axhh2, axvv2 = self.axes
        date_format = mdates.DateFormatter('$%H:%M$')

        axeg.semilogy()
        axeg.legend(frameon=False)
        axeg.set_xlabel('SVD Mode')
        axeg.set_ylabel('Singular Value')
        axeg.minorticks_on()
        axeg.tick_params(length=4, width=1, direction='in')
        axeg.tick_params(which='minor', length=2, width=1, direction='in')

        axhh.set_ylim(ymin=0.0, ymax=mode_n + 0.5)
        axhh.set_xlim(xmin=xmin, xmax=xmax)
        axhh.set_title('Time Modes')
        axhh.xaxis.set_major_formatter(date_format)
        #axhh.semilogy()
        axhh.set_yticklabels([])
        axhh.set_xticklabels([])
        axhh.minorticks_on()
        axhh.tick_params(length=4, width=1, direction='in')
        axhh.tick_params(which='minor', length=2, width=1, direction='in')
        axhh.yaxis.set_tick_params(which='both', left='off', right='off')


        axvv.set_ylim(ymin=0.0, ymax=mode_n + 0.5)
        axvv.set_xlim(xmin=xmin, xmax=xmax)
        axvv.xaxis.set_major_formatter(date_format)
        #axvv.set_xlabel('Time index')
        axvv.set_xlabel(self.x_label)
        #axvv.semilogy()
        axvv.set_yticklabels([])
        axvv.minorticks_on()
        axvv.tick_params(length=4, width=1, direction='in')
        axvv.tick_params(which='minor', length=2, width=1, direction='in')
        axvv.yaxis.set_tick_params(which='both', left='off', right='off')

        axhh2.set_ylim(ymin=0.0, ymax=mode_n + 0.5)
        #axhh2.set_xlim(xmin=f.min(), xmax=f.max())
        axhh2.set_xlim(xmin=fmin, xmax=fmax)
        axhh2.set_title('Frequency Modes')
        axhh2.set_yticklabels([])
        axhh2.set_xticklabels([])
        axhh2.minorticks_on()
        axhh2.tick_params(length=4, width=1, direction='in')
        axhh2.tick_params(which='minor', length=2, width=1, direction='in')
        axhh2.yaxis.set_tick_params(which='both', left='off', right='off')


        axvv2.set_ylim(ymin=0.0, ymax=mode_n + 0.5)
        #axvv2.set_xlim(xmin=f.min(), xmax=f.max())
        axvv2.set_xlim(xmin=fmin, xmax=fmax)
        axvv2.set_xlabel('Frequency GHz')
        axvv2.set_yticklabels([])
        axvv2.minorticks_on()
        axvv2.tick_params(length=4, width=1, direction='in')
        axvv2.tick_params(which='minor', length=2, width=1, direction='in')
        axvv2.yaxis.set_tick_params(which='both', left='off', right='off')

        #self.fig.savefig(fig_name + '.png', dpi=500)
        self.fig.savefig(fig_name + '.eps')
        self.fig.clf()
        plt.close()


    def process(self, input):

        #labels   = self.params['labels']
        #fig_name = self.params['fig_name']
        #fig_name = output_path(fig_name)
        mode_n = self.params['mode_n']

        axeg, axhh, axvv, axhh2, axvv2 = self.axes

        ii = self.iteration
        fh = input

        u = fh['u'][:]
        v = fh['v'][:]
        s = fh['s'][:]

        t = fh['t'][:]
        f = fh['f'][:]
        print f
        print f[1] - f[0]

        bt = fh['bad_time'][:]
        bf = fh['bad_freq'][:]

        gf_st = np.argwhere(~bf)[ 0, 0]
        gf_ed = np.argwhere(~bf)[-1, 0]
        gt_st = np.argwhere(~bt)[ 0, 0]
        gt_ed = np.argwhere(~bt)[-1, 0]

        print u.shape, v.shape
        print bt.shape, bf.shape
        
        x = [datetime.fromtimestamp(ss) for ss in t]
        self.x_label = '$%s$' % x[gt_st].date()
        x = mdates.date2num(x)
        _u = np.ma.zeros(t.shape[0])
        _u.mask = bt
        _v = np.ma.zeros(f.shape[0])
        _v.mask = bf

        for i in range(mode_n):

            rg = 6 * np.std(u[:, i, 0])
            mn = np.mean(u[:, i, 0])
            _u[~bt] = 1 * (u[:, i, 0] - mn) / rg
            axhh.plot(x, (mode_n - i) + _u, 'r-', linewidth = 1.)
            rg = 6 * np.std(u[:, i, 1])
            mn = np.mean(u[:, i, 1])
            _u[~bt] = 1 * (u[:, i, 1] - mn) / rg
            axvv.plot(x, (mode_n - i) + _u, 'g-', linewidth = 1.)

            #axhh.text(x.max() + 50, (mode_n - i), r'${\rm Mode}\, %d$'%i)
            #axvv.text(x.max() + 50, (mode_n - i), r'${\rm Mode}\, %d$'%i)

            rg = 4 * np.std(v[:, i, 0])
            mn = np.mean(v[:, i, 0])
            _v[~bf] = 1. * (v[i, :, 0] - mn) / rg
            axhh2.plot(f, (mode_n - i)+_v, 'r-', linewidth = 1.)
            rg = 4 * np.std(v[:, i, 1])
            mn = np.mean(v[:, i, 0])
            _v[~bf] = 1. * (v[i, :, 1] - mn) / rg
            axvv2.plot(f, (mode_n - i)+_v, 'g-', linewidth = 1.)

        axeg.plot(np.arange(mode_n) + 1, s[:mode_n, 0], 'ro--', linewidth = 1.5, label='HH')
        axeg.plot(np.arange(mode_n) + 1, s[:mode_n, 1], 'g^-', linewidth = 1.5, label='VV')

        if ii == self.iter_num - 1:
            return x[gt_st], x[gt_ed], f[gf_st], f[gf_ed]




