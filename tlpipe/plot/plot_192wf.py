
from tlpipe.utils import mpiutil
from tlpipe.core.base_exe import Base

import h5py
import numpy as np
import scipy as sp
import gc
import matplotlib.pyplot as plt
from tlpipe.utils.path_util import input_path, output_path

from tlpipe.timestream.base_operation import BaseOperation


# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.

class Plot_192WF(BaseOperation):
    """ 
    """
    params_init = {
                   'extra_history' : 'plot',
                   'output_plot'   : './',
                   'time_distributed_dset' : ['vis', ],
                   'amp_phs' : False,
                  }
    prefix = 'p192wf_'
    
    def action(self, vis):

        params = self.params

        if mpiutil.rank0:
            output_plot  = output_path(params['output_plot'] + '/%s')
        mpiutil.barrier()
        output_plot  = output_path(params['output_plot'] + '/%s')

        #output_plot  = output_path(params['output_plot'])
        #output_plot += "/%s"

        vis.redistribute(-1)

        #print mpiutil.rank, vis['blorder'].local_shape
        bl_list = vis['blorder'].local_data
        freq = vis['freq'][:]
        for dset_name in params['time_distributed_dset']:
            time = vis['jul_date'][:]

            for bl_indx in np.arange(bl_list.shape[0]):

                a1 = bl_list[bl_indx][0]
                a2 = bl_list[bl_indx][1]
                ant_pair = "%02dx%02d"%(a1, a2)
                print "%d x %d"%(a1, a2)

                dset = np.ma.masked_invalid(vis[dset_name].local_data[...,bl_indx])
                plot_wf(dset[:,:,0], time, freq, 
                        output_plot%dset_name, ant_pair, 'XX', params['amp_phs'])
                plot_wf(dset[:,:,1], time, freq, 
                        output_plot%dset_name, ant_pair, 'YY', params['amp_phs'])
                plot_wf(dset[:,:,2], time, freq, 
                        output_plot%dset_name, ant_pair, 'XY', params['amp_phs'])
                plot_wf(dset[:,:,3], time, freq, 
                        output_plot%dset_name, ant_pair, 'YX', params['amp_phs'])

class Plot_192NC(BaseOperation):
    """ 
    """
    params_init = {
                   'extra_history' : 'plot',
                   'output_plot'   : './',
                   'amp_phs' : False,
                  }
    prefix = 'p192nc_'
    
    def action(self, vis):

        params = self.params

        if params['bl_set'] != 'all':
            raise "bl_set need to be all"

        if mpiutil.rank0:
            output_plot  = output_path(params['output_plot'] + '/%s')
        mpiutil.barrier()
        output_plot  = output_path(params['output_plot'] + '/%s')

        bl_list = vis['blorder'][:]
        print bl_list
        freq = vis['freq'][:]
        dset_name = 'noisecal'
        time = vis['noisecal_jul_date'][:]

        for bl_indx in np.arange(bl_list.shape[0]):

            a1 = bl_list[bl_indx][0]
            a2 = bl_list[bl_indx][1]
            ant_pair = "%02dx%02d"%(a1, a2)
            print "%d x %d"%(a1, a2)

            dset = vis[dset_name].local_data[...,bl_indx]
            plot_wf(dset[:,:,0], time, freq, 
                    output_plot%("rank%03d"%mpiutil.rank + dset_name), 
                    ant_pair, 'XX', params['amp_phs'])
            plot_wf(dset[:,:,1], time, freq, 
                    output_plot%("rank%03d"%mpiutil.rank + dset_name), 
                    ant_pair, 'YY', params['amp_phs'])
            plot_wf(dset[:,:,2], time, freq, 
                    output_plot%("rank%03d"%mpiutil.rank + dset_name), 
                    ant_pair, 'XY', params['amp_phs'])
            plot_wf(dset[:,:,3], time, freq, 
                    output_plot%("rank%03d"%mpiutil.rank + dset_name), 
                    ant_pair, 'YX', params['amp_phs'])


def plot_wf(vis, time, freq, output_file, ant_pair, pol, amp_phs=False):

    time0 = int(time[0])
    time = time - time0

    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.add_axes([0.12, 0.53, 0.75, 0.40])
    ax2 = fig.add_axes([0.12, 0.10, 0.75, 0.40])
    cax1 = fig.add_axes([0.875, 0.55, 0.02, 0.34])
    cax2 = fig.add_axes([0.875, 0.13, 0.02, 0.34])
    
    
    X, Y = np.meshgrid(time, freq)
    #im1 = ax1.pcolormesh(X, Y, np.abs(vis).T)
    #im2 = ax2.pcolormesh(X, Y, np.angle(vis).T)
    if amp_phs:
        im1 = ax1.pcolormesh(X, Y, np.ma.abs(vis.T))
        im2 = ax2.pcolormesh(X, Y, np.ma.angle(vis.T) * 180./np.pi, vmax=180., vmin=-180)
    else:
        im1 = ax1.pcolormesh(X, Y, vis.real.T)
        im2 = ax2.pcolormesh(X, Y, vis.imag.T)
    
    fig.colorbar(im1, cax=cax1)
    fig.colorbar(im2, cax=cax2)
    cax1.minorticks_on()
    cax1.tick_params(length=2, width=1, direction='out')
    cax1.tick_params(which='minor', length=1, width=1, direction='out')
    cax2.minorticks_on()
    cax2.tick_params(length=2, width=1, direction='out')
    cax2.tick_params(which='minor', length=1, width=1, direction='out')
    
    
    ax1.set_title('Antenna Pair %s %s '%(ant_pair, pol))
    
    ax1.set_xticklabels([])
    if amp_phs:
        ax1.set_ylabel('Freq Amplitude')
    else:
        ax1.set_ylabel('Freq Real')
    #ax1.set_ylabel('Freq Amp')
    ax1.minorticks_on()
    ax1.set_xlim(xmin=time.min(), xmax=time.max())
    ax1.set_ylim(ymin=freq.min(), ymax=freq.max())
    ax1.tick_params(length=4, width=1., direction='out')
    ax1.tick_params(which='minor', length=2, width=1., direction='out')
    
    
    if amp_phs:
        ax2.set_ylabel('Freq Phase')
    else:
        ax2.set_ylabel('Freq Imag')
    #ax2.set_ylabel('Freq Pha')
    ax2.set_xlabel('Time + %f [JD]'%time0)
    ax2.minorticks_on()
    ax2.set_xlim(xmin=time.min(), xmax=time.max())
    ax2.set_ylim(ymin=freq.min(), ymax=freq.max())
    ax2.tick_params(length=4, width=1., direction='out')
    ax2.tick_params(which='minor', length=2, width=1., direction='out')
    
    if amp_phs:
        plt.savefig(output_file + '_AP_AntPir%s_%s.png'%(ant_pair, pol), format='png')
    else:
        plt.savefig(output_file + '_RI_AntPir%s_%s.png'%(ant_pair, pol), format='png')
    #plt.show()

    plt.close()
    gc.collect()

