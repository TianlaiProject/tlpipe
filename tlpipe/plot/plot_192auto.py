import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import numpy as np
from tlpipe.utils.date_util import get_ephdate

def plot_cros_wf(ant_list=None):

    file_root = '/project/ycli/data/tianlai/cyl192ch_test/raw_cross/'
    file_name = '20160127114202_lz1_%05d.hdf5'
    
    output_root = '/project/ycli/data/tianlai/cyl192ch_test/plot/'
    
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
            for i in range(13, 26):
            
                f = h5py.File(file_root + file_name%i, 'r')
            
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
    
            ax1.vlines( get_ephdate('2016/1/27 13:29:15', tzone='UCT+8') - time0, 
                    ymax=ymax, ymin=ymin, label='CygA', colors='g')
            ax2.vlines( get_ephdate('2016/1/27 13:29:15', tzone='UCT+8') - time0, 
                    ymax=ymax, ymin=ymin, label='CygA', colors='g')
            ax1.vlines( get_ephdate('2016/1/27 14:05:36', tzone='UCT+8') - time0, 
                    ymax=ymax, ymin=ymin, label='Sun', colors='r')
            ax2.vlines( get_ephdate('2016/1/27 14:05:36', tzone='UCT+8') - time0, 
                    ymax=ymax, ymin=ymin, label='Sun', colors='r')
    
            #ax1.legend(frameon=False)
    
            plt.savefig(output_root + 'cros_ant%02dx%02d_wf.png'%(a1, a2))
            
            plt.show()
    
            plt.clf()



def plot_all():

    file_root = '/project/ycli/data/tianlai/cyl192ch_test/raw/'
    file_name = '20160127114202_lz1_%05d.hdf5'
    
    output_root = '/project/ycli/data/tianlai/cyl192ch_test/plot/'
    
    #ant_list = range(96)

    fig = plt.figure(figsize=(10, 10))

    l = 0.01
    b = 0.01
    h = 0.03
    w = 0.3

    for i in range(33):

        hs = ((1 - l - b) - 33 * h ) / 32
        ax = fig.add_axes([l, b+(33-i-1)*(hs+h), w, h])

        ax.set_xticklabels([])
        ax.set_yticklabels([])

    plt.show()

def plot_wf(ant_list=None):
    file_root = '/project/ycli/data/tianlai/cyl192ch_test/raw/'
    file_name = '20160127114202_lz1_%05d.hdf5'
    
    output_root = '/project/ycli/data/tianlai/cyl192ch_test/plot/'
    
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
        for i in range(13, 26):
        
            f = h5py.File(file_root + file_name%i, 'r')
        
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
    
        ax1.vlines( get_ephdate('2016/1/27 13:29:15', tzone='UCT+8') - time0, 
                ymax=ymax, ymin=ymin, label='CygA', colors='g')
        ax2.vlines( get_ephdate('2016/1/27 13:29:15', tzone='UCT+8') - time0, 
                ymax=ymax, ymin=ymin, label='CygA', colors='g')
        ax1.vlines( get_ephdate('2016/1/27 14:05:36', tzone='UCT+8') - time0, 
                ymax=ymax, ymin=ymin, label='Sun', colors='r')
        ax2.vlines( get_ephdate('2016/1/27 14:05:36', tzone='UCT+8') - time0, 
                ymax=ymax, ymin=ymin, label='Sun', colors='r')
    
        #ax1.legend(frameon=False)
    
        plt.savefig(output_root + 'auto_ant%02d_wf.png'%ant)
        
        plt.show()
    
        plt.clf()


def plot_each():

    file_root = '/project/ycli/data/tianlai/cyl192ch_test/raw/'
    file_name = '20160127114202_lz1_%05d.hdf5'
    
    output_root = '/project/ycli/data/tianlai/cyl192ch_test/plot/'
    
    ant_list = range(96)
    color_list = ['r', 'k', 'b', 'g', 'c', 'm']
    
    
    for ant in ant_list:
    
        fig = plt.figure(figsize=(7, 6))
        ax1  = fig.add_axes([0.1, 0.51, 0.85, 0.40])
        ax2  = fig.add_axes([0.1, 0.1, 0.85, 0.40])
    
    
        ymin = 1.e99
        ymax = -1.e99
        time0 = None
        for i in range(13, 26):
        
            f = h5py.File(file_root + file_name%i, 'r')
        
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
    
        ax1.vlines( get_ephdate('2016/1/27 13:29:15', tzone='UCT+8') - time0, 
                ymax=ymax, ymin=ymin, label='CygA', colors='g')
        ax2.vlines( get_ephdate('2016/1/27 13:29:15', tzone='UCT+8') - time0, 
                ymax=ymax, ymin=ymin, label='CygA', colors='g')
        ax1.vlines( get_ephdate('2016/1/27 14:05:36', tzone='UCT+8') - time0, 
                ymax=ymax, ymin=ymin, label='Sun', colors='r')
        ax2.vlines( get_ephdate('2016/1/27 14:05:36', tzone='UCT+8') - time0, 
                ymax=ymax, ymin=ymin, label='Sun', colors='r')
    
        ax1.legend(frameon=False)
    
    
        plt.savefig(output_root + 'auto_ant%02d.png'%ant)
        
        plt.show()
    
        plt.clf()


if __name__=="__main__":

    #plot_wf([1, 2, 95])
    plot_cros_wf([1, 2, 95])
