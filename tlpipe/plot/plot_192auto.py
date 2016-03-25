import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import numpy as np
from tlpipe.utils.date_util import get_ephdate


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
        
        #plt.show()
    
        plt.clf()


if __name__=="__main__":

    plot_all()
