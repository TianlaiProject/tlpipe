import os
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import numpy as np
import ephem
from tlpipe.utils.date_util import get_ephdate
from tlpipe.kiyopy import parse_ini
from tlpipe.utils import mpiutil
from tlpipe.core.base_exe import Base
from tlpipe.utils.path_util import input_path, output_path

params_init = {
               'nprocs': mpiutil.size, # number of processes to run this module
               'aprocs': range(mpiutil.size), # list of active process rank no.
               'input_file': 'uv_image.hdf5',  # str or a list of str
               'output_file': '', # None, str or a list of str
               'auto' : True,
               'cros' : False,
               'antenna_list' : [],
               'int_time' : 4,
               'obs_starttime': '',
               'tzone' : 'UTC+08',
               'time_range': [],
               'source_cat': {},
              }
prefix = 'plt192_'

class Plot_192(Base):
    """Plot image."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(Plot_192, self).__init__(parameter_file_or_dict, params_init, prefix, feedback)

    def execute(self):

        if mpiutil.rank0:
            input_file = input_path(self.params['input_file'])
            output_root = output_path(self.params['output_file'])
            if not os.path.exists(output_root):
                os.makedirs(output_root)

            # open one file, get data shape
            input_base = os.path.split(input_file)[0]
            data = h5py.File(os.path.join(input_base,os.listdir(input_base)[0]), 'r')
            data_shape = data['data'].shape
            data.close()

            tzone = self.params['tzone']
            obs_starttime = get_ephdate(self.params['obs_starttime'], tzone=tzone)

            time_axis  = np.arange(data_shape[0]).astype('float')
            time_axis *= self.params['int_time'] * ephem.second
            time_axis += obs_starttime 

            st_ind  = get_ephdate(self.params['time_range'][0], tzone=tzone)
            ed_ind  = get_ephdate(self.params['time_range'][1], tzone=tzone)

            st_ind -= obs_starttime 
            ed_ind -= obs_starttime

            st_ind /= data_shape[0] * self.params['int_time'] * ephem.second
            ed_ind /= data_shape[0] * self.params['int_time'] * ephem.second

            st_ind  = int(st_ind)
            ed_ind  = int(ed_ind)

            input_file_list = []
            for i in range(st_ind, ed_ind + 1):
                input_file_list.append(input_file%i)

            if self.params['cros']:
                plot_cros_wf(
                        input_file_list=input_file_list, 
                        output_root=output_root, 
                        ant_list=self.params['antenna_list'], 
                        source_cat=self.params['source_cat'])
            if self.params['auto']:
                plot_auto_sp(
                        input_file_list=input_file_list, 
                        output_root=output_root, 
                        ant_list=self.params['antenna_list'], 
                        source_cat=self.params['source_cat'])
                plot_auto_wf(
                        input_file_list=input_file_list, 
                        output_root=output_root, 
                        ant_list=self.params['antenna_list'], 
                        source_cat=self.params['source_cat'])
        mpiutil.barrier()

def plot_cros_wf(input_file_list, output_root, ant_list=None, source_cat={}):

    if ant_list == None:
        ant_list = range(96)
    color_list = ['r', 'k', 'b', 'g', 'c', 'm']
    
    
    ant_num = len(ant_list)
    bl_num =  np.sum(range(ant_num+1))
    for ai in range(ant_num):
        for aj in range(ai, ant_num):

            a1  = ant_list[ai]
            a2  = ant_list[aj]
            ant = bl_num - np.sum(range(ant_num-ai+1)) + (aj-ai)
    
            fig = plt.figure(figsize=(10, 5))
            ax1 = fig.add_axes([0.1, 0.52, 0.80, 0.38])
            ax2 = fig.add_axes([0.1, 0.10, 0.80, 0.38])
            cax1 = fig.add_axes([0.91, 0.52, 0.01, 0.38])
            cax2 = fig.add_axes([0.91, 0.10, 0.01, 0.38])

    
            ymin = 1.e99
            ymax = -1.e99
            xmin = 1.e99
            xmax = -1.e99
            time0 = None
            mean = None
            std  = None
            #for i in range(13, 26):
            for input_file in input_file_list:
            
                #f = h5py.File(file_root + file_name%i, 'r')
                f = h5py.File(input_file, 'r')
            
                vis = np.ma.array(f['data'].value)
                time = f['time'].value
                freq = f['data'].attrs['freq']

                ymin, ymax = freq.min(), freq.max()
            
                if time0 == None:
                    time0 = int(time[0])
                time -= time0

                if xmin > time.min(): xmin=time.min()
                if xmax < time.max(): xmax=time.max()
            

                vis[np.logical_not(np.isfinite(vis))] = np.ma.masked

                print np.ma.mean(vis[:, ant, 3, :].real, axis=0).shape
                print np.ma.mean(vis[:, ant, 3, :].real, axis=0)

                if mean == None:
                    mean = np.ma.mean(vis[:, ant, 0::3, :].real)
                    std  = np.ma.std(vis[:, ant, 0::3, :].real)
            
                Y, X = np.meshgrid(freq, time)

                im = ax1.pcolormesh(X, Y, 
                        (vis[:, ant, 0, :].real - mean)/std, vmax=3, vmin=-3)
                ax1.set_title('CROSS Real Ant%02d'%a1 + r'$\times$' + 'Ant%02d'%a2)
                ax1.set_ylabel('XX Frequency [MHz]')
                ax1.minorticks_on()
                ax1.tick_params(length=4, width=1., direction='out')
                ax1.tick_params(which='minor', length=2, width=1., direction='out')
                ax1.set_xticklabels([])

                fig.colorbar(im, ax=ax1, cax=cax1)
    
                im = ax2.pcolormesh(X, Y, 
                        (vis[:, ant, 3, :].real - mean)/std, vmax=3, vmin=-3)
                ax2.set_ylabel('YY Frequency [MHz]')
                ax2.set_xlabel('time + %d '%time0)
                ax2.minorticks_on()
                ax2.tick_params(length=4, width=1., direction='out')
                ax2.tick_params(which='minor', length=2, width=1., direction='out')
    
                fig.colorbar(im, ax=ax2, cax=cax2)
            
                f.close()
    
            ax1.set_ylim(ymin=ymin, ymax=ymax)
            ax2.set_ylim(ymin=ymin, ymax=ymax)
            ax1.set_xlim(xmin=xmin, xmax=xmax)
            ax2.set_xlim(xmin=xmin, xmax=xmax)
    
            for j in range(len(source_cat.keys())):
                source = source_cat.keys()[j]
                ax1.vlines( get_ephdate(source_cat[source], tzone='UCT+8') - time0, 
                        ymax=ymax, ymin=ymin, label=source, colors=color_list[j])
                ax2.vlines( get_ephdate(source_cat[source], tzone='UCT+8') - time0, 
                        ymax=ymax, ymin=ymin, label=source, colors=color_list[j])
    
            plt.savefig(os.path.join(output_root, 'cros_ant%02dx%02d_wf.png'%(a1, a2)))
            
            plt.show()
    
            plt.close()
            #plt.clf()

def plot_auto_wf(input_file_list, output_root, ant_list=None, source_cat={}):

    if ant_list == None:
        ant_list = range(96)
    color_list = ['r', 'k', 'b', 'g', 'c', 'm']
    
    
    for ant in ant_list:
    
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_axes([0.1, 0.52, 0.80, 0.38])
        ax2 = fig.add_axes([0.1, 0.10, 0.80, 0.38])
        cax1 = fig.add_axes([0.91, 0.52, 0.01, 0.38])
        cax2 = fig.add_axes([0.91, 0.10, 0.01, 0.38])

        ymin = 1.e99
        ymax = -1.e99
        xmin = 1.e99
        xmax = -1.e99
        time0 = None
        mean = None
        std  = None
        for input_file in input_file_list:
        
            f = h5py.File(input_file, 'r')
        
            vis = np.ma.array(f['data'].value)
            time = f['time'].value
            freq = f['data'].attrs['freq']

            ymin, ymax = freq.min(), freq.max()
        
            if time0 == None:
                time0 = int(time[0])
            time -= time0

            if xmin > time.min(): xmin=time.min()
            if xmax < time.max(): xmax=time.max()
        

            vis[np.logical_not(np.isfinite(vis))] = np.ma.masked

            if mean == None:
                mean = np.ma.mean(vis[:, ant, 0::3, :].real)
                std  = np.ma.std(vis[:, ant, 0::3, :].real)
        
            Y, X = np.meshgrid(freq, time)

            im = ax1.pcolormesh(X, Y, 
                    (vis[:, ant, 0, :].real - mean)/std, vmax=3, vmin=-3)
            ax1.set_title('AUTO Ant No. %02d'%ant)
            ax1.set_ylabel('XX Frequency [MHz]')
            ax1.minorticks_on()
            ax1.tick_params(length=4, width=1., direction='out')
            ax1.tick_params(which='minor', length=2, width=1., direction='out')
            ax1.set_xticklabels([])

            fig.colorbar(im, ax=ax1, cax=cax1)
    
            im = ax2.pcolormesh(X, Y, 
                    (vis[:, ant, 3, :].real - mean)/std, vmax=3, vmin=-3)
            ax2.set_ylabel('YY Frequency [MHz]')
            ax2.set_xlabel('time + %d '%time0)
            ax2.minorticks_on()
            ax2.tick_params(length=4, width=1., direction='out')
            ax2.tick_params(which='minor', length=2, width=1., direction='out')
    
            fig.colorbar(im, ax=ax2, cax=cax2)
        
            f.close()
    
        ax1.set_ylim(ymin=ymin, ymax=ymax)
        ax2.set_ylim(ymin=ymin, ymax=ymax)
        ax1.set_xlim(xmin=xmin, xmax=xmax)
        ax2.set_xlim(xmin=xmin, xmax=xmax)
    
        for j in range(len(source_cat.keys())):
            source = source_cat.keys()[j]
            ax1.vlines( get_ephdate(source_cat[source], tzone='UCT+8') - time0, 
                    ymax=ymax, ymin=ymin, label=source, colors=color_list[j])
            ax2.vlines( get_ephdate(source_cat[source], tzone='UCT+8') - time0, 
                    ymax=ymax, ymin=ymin, label=source, colors=color_list[j])

        plt.savefig(os.path.join(output_root, 'auto_ant%02d_wf.png'%ant))
        
        plt.show()
    
        plt.close()


def plot_auto_sp(input_file_list, output_root, ant_list=None, source_cat={}):

    #file_root = '/project/ycli/data/tianlai/cyl192ch_test/raw/'
    #file_name = '20160127114202_lz1_%05d.hdf5'
    #
    #output_root = '/project/ycli/data/tianlai/cyl192ch_test/plot/'
    
    if ant_list == None:
        ant_list = range(96)
    color_list = ['r', 'k', 'b', 'g', 'c', 'm']
    
    
    for ant in ant_list:
    
        fig = plt.figure(figsize=(7, 6))
        ax1  = fig.add_axes([0.1, 0.51, 0.85, 0.40])
        ax2  = fig.add_axes([0.1, 0.1, 0.85, 0.40])
    
    
        ymin = 1.e99
        ymax = -1.e99
        time0 = None
        for input_file in input_file_list:
        
            f = h5py.File(input_file, 'r')
        
            vis = f['data'].value
            time = f['time'].value
        
            if time0 == None:
                time0 = int(time[0])
            time -= time0
        
            vis[np.logical_not(np.isfinite(vis))] = 0
        
            #x = np.arange(vis.shape[0]) + i * vis.shape[0]
            x = time
    
            y = np.mean(vis[:,ant,0,:].real, axis=-1)
            if ymax < y.max(): ymax = y.max()
            if ymin > y.min(): ymin = y.min()
        
            ax1.plot(x, y, 'k.-' )
            ax1.set_title('AUTO Ant No. %02d'%ant)
            ax1.set_ylabel('XX')
            #ax1.set_xlim(xmin=np.min(time), xmax=np.max(time))
            ax1.minorticks_on()
            ax1.tick_params(length=4, width=1., direction='out')
            ax1.tick_params(which='minor', length=2, width=1., direction='out')
            ax1.set_xticklabels([])
    
            y = np.mean(vis[:,ant,3,:].real, axis=-1)
            if ymax < y.max(): ymax = y.max()
            if ymin > y.min(): ymin = y.min()
    
            ax2.plot(x, y, 'k.-' )
            ax2.set_ylabel('YY')
            #ax2.set_xlim(xmin=np.min(time), xmax=np.max(time))
            ax2.set_xlabel('time + %d '%time0)
            ax2.minorticks_on()
            ax2.tick_params(length=4, width=1., direction='out')
            ax2.tick_params(which='minor', length=2, width=1., direction='out')
    
        
            f.close()
    
        ax1.set_ylim(ymin=ymin, ymax=ymax)
        ax2.set_ylim(ymin=ymin, ymax=ymax)
    
        for j in range(len(source_cat.keys())):
            source = source_cat.keys()[j]
            ax1.vlines( get_ephdate(source_cat[source], tzone='UCT+8') - time0, 
                    ymax=ymax, ymin=ymin, label=source, colors=color_list[j])
            ax2.vlines( get_ephdate(source_cat[source], tzone='UCT+8') - time0, 
                    ymax=ymax, ymin=ymin, label=source, colors=color_list[j])

        ax1.legend(frameon=False)
    
        plt.savefig(os.path.join(output_root, 'auto_ant%02d_sp.png'%ant))
        
        plt.show()
    
        #plt.clf()
        plt.close()


if __name__=="__main__":

    pass
