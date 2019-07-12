"""Experimental sandbox.

Inheritance diagram
-------------------

.. inheritance-diagram:: Play
   :parts: 2

"""

import os
from datetime import datetime
import numpy as np
import timestream_task
from tlpipe.container.raw_timestream import RawTimestream
from tlpipe.container.timestream_common import TimestreamCommon
from tlpipe.utils.path_util import output_path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
#from mpi4py import MPI

import numpy.ma as ma
from scipy import linalg as la
import scipy.optimize as opt
import scipy.interpolate as interp
import scipy.stats.mstats as mstats

import logging

logger = logging.getLogger(__name__)

 
class Stack(timestream_task.TimestreamTask):


    params_init = {
                    'fig_name': 'plots/',
 #                   'plot': False,
                    'plot_tproj': False,
                   }

    prefix = 'st_'


    def process(self, ts):

        logger.info("Start Stack")

        fig_prefix = self.params['fig_name']
#        plot = self.params['plot']
        plot_tproj = self.params['plot_tproj']
#        print("fig_prefix=",fig_prefix)
#        print("plot_tproj=",plot_tproj)
#        print("plot=",plot)

#Check axis order.  Must be time x frequency x baseline. No reordering
        axes = ts.main_data_axes
        print("axes=",axes)
#        ok = True
#        if axes[0] != 'time':
#            ok = False
#        if axes[1] != 'frequency':
#            ok = False
#        if axes[2] != 'baseline':
#            ok = False
#        if not ok:
#            raise RuntimeError('Axis order must be time,frequency,baseline')

#        print("keys=",ts.keys())
#        print("attrs=",ts.attrs.keys())
#        tzero = ts.attrs['sec1970']
        #print("tzero=",tzero[0],tzero[1])
#        strout = "sec1970=%15.1f" % tzero[0]
#        print(strout)
#        sec1970 = ts['sec1970']
#        strout = "sec1970=%15.3f to %15.3f " % (sec1970[0],sec1970[3599])
#        print(strout)

        blorder = ts['blorder']
        freq = ts['freq'][:]
        print("freq.dim=",freq.shape)
#        print(freq)
#        print("blorder=",blorder)
        ntot = blorder.shape[0]
        for n in range(ntot):
            print("blorder=",n,blorder[n])
 #       exit(0)
        autocorr = list()
        crosscorr = list()
        for n in range(ntot):
            if (blorder[n][0]==blorder[n][1]):
                autocorr.append(n)
            else: 
                crosscorr.append(n)
#        print ("autocorr",autocorr)
#        print ("crosscorr",crosscorr)
#        jul_date = ts['jul_date']
#        print("jul_date=",jul_date)
        vis = ts['vis']
        vis_mask = ts['vis_mask']
#        print("vis_mask.dtype=",vis_mask.dtype,vis_mask.shape)
#        number = np.count_nonzero(vis_mask)
#        print("masked=",number)
#        blorder = ts['blorder']
#        ntpt = vis.shape[0]
#        nfpt = vis.shape[1]
#        nbl = vis.shape[2]
#        print("ntpt=",ntpt,"nfpt=",nfpt,"nbl=",nbl)
#        print("blorder")
#        for n in range(nbl):
#            print(n,blorder[n])

 

#        logger.info("Count of frequencies masked by rfi %i" % fcount)
#        logger.info("Count of times masked by rfi %i" % tcount)
#        logger.info("Count of frequency-time points masked by rfi %i" % count)

        ntpt = vis.shape[0]
        nfpt = vis.shape[1]
        npol = vis.shape[2]
        nbl = vis.shape[3]
#        print("ntpt,nfpt,npol,nbl=",ntpt,nfpt,npol,nbl)

    
        if plot_tproj:
            tproj = np.mean(np.abs(vis[:,:,:,crosscorr]),axis=(1,3))
            for n in range(npol):
                plt.figure()
                plt.plot(range(ntpt),tproj[:,n])
                if n==0:
                    name = fig_prefix + '_cross_xx'
                if n==1:
                    name = fig_prefix + '_cross_yy'
                if n==2:
                    name = fig_prefix + '_cross_xy'
                if n==3:
                    name = fig_prefix + '_cross_yx'
                fig_name = output_path(name)
                print("fig_name=",fig_name)
                plt.savefig(fig_name)
                plt.close()               

            tproj = np.mean(np.abs(vis[:,:,:,autocorr]),axis=(1,3))
            for n in range(npol):
                plt.figure()
                plt.plot(range(ntpt),tproj[:,n])
                if n==0:
                    name = fig_prefix + '_auto_xx'
                if n==1:
                    name = fig_prefix + '_auto_yy'
                if n==2:
                    name = fig_prefix + '_auto_xy'
                if n==3:
                    name = fig_prefix + '_auto_yx'
                fig_name = output_path(name)
                print("fig_name=",fig_name)
                plt.savefig(fig_name)
                plt.close()
        plot_tindv = True
        if plot_tindv:
            for nb in range(nbl):
                for n in range(npol):
                    tproj = np.mean(np.abs(vis[:,:,n,nb]),axis=1)
                    plt.figure()
                    plt.plot(range(ntpt),tproj[:])
                    name = fig_prefix + ('_%i_%i' % (nb,n))
                    fig_name = output_path(name)
                    print("fig_name=",fig_name)
                    plt.savefig(fig_name)
                    plt.close()

        freq_label = r'$Frequency$ (MHz)'
        time_label = r'$Time$ (sec)'
        title = 'Phase(Vis)'
        freq_extent = [freq[0], freq[-1]]
        time_extent = [0.0,np.float32(ntpt-1)]
        extent = time_extent +  freq_extent 
        mvis = np.ma.array(vis,mask=vis_mask)
        xtext = 0.80
        ytext = 1.10
        midt = ntpt/2
        fbin = np.arange(nfpt)

        vislvl = np.zeros([4,nbl],dtype=np.float32)
        for pol in range(4):
            for blord in range(nbl):
                bl = blorder[blord]
 #               print("bl,pol=",bl,pol)

#                plt.figure(figsize=[6,3],dpi=600)
#                axes = plt.subplot(111)
#                vis1 = np.angle(mvis[:,:,pol,blord])
#                im = plt.imshow(vis1.T,aspect=2.0,extent=extent,origin='lower')
#                axes.set_title(title)
#                axes.set_xlabel(time_label)
#                axes.set_ylabel(freq_label)
#                plt.colorbar(im)
#                fig_name = '%s_phase_%d_%d_%s.png' % \
#                    (fig_prefix, bl[0], bl[1], ts.pol_dict[pol])
#                vislabel = 'V(%d,%d) %s' % (bl[0],bl[1],ts.pol_dict[pol])
#                fig_name = output_path(fig_name)
#                plt.text(xtext,ytext,vislabel,transform=axes.transAxes)
#                plt.savefig(fig_name)
#                plt.close()

#                famp = np.abs(vis[midt,:,pol,blord])
#                fphs = (180.0/np.pi)*np.angle(vis[midt,:,pol,blord])
#                plt.figure()
#                ax1 = plt.axes()
#                ax1.set_xlabel("Frequency (MHz)")
#                ax1.set_ylabel("Amplitude")
#                hiamp = 1.5*np.max(famp[50:461])
#                ax1.set_ylim(0.0,hiamp)
#                ax1.plot(freq,famp,'rs',ms=3.0,label='Amplitude')
#                ax1.legend(loc=2)
#                ax2 = ax1.twinx()
#                ax2.set_ylabel("Phase")
#                ax2.set_ylim(-200.0,250.0)
#                ax2.plot(freq,fphs,'bo',ms=3.0,label='Phase')
#                ax2.legend(loc=1)
#                fig_name = '%s_pr_%d_%d_%s.png' % \
#                    (fig_prefix,bl[0],bl[1],ts.pol_dict[pol])
#                fig_name = output_path(fig_name)
#                print("fig_name",fig_name)
#                plt.savefig(fig_name)
#                plt.close()
                vislvl[pol,blord] = np.mean(np.abs(vis[:,100:412,pol,blord]))
                if vislvl[pol,blord]<1.e-6:
                    vislvl[pol,blord]=-100.0
#                print("vislvl=",pol,blord,vislvl[pol,blord])
#            print("auto_corr=",pol,histdat)
            if pol==0:
                pname = 'XX'
            if pol==1:
                pname = 'YY'
            if pol==2:
                pname = 'XY'
            if pol==3:
                pname = 'YX'
            
            histdat = vislvl[pol,autocorr]
            plt.figure()
            ax = plt.axes()
            ax.set_title(pname+' Auto-correlations')
            ave = np.mean(histdat)
            upper = 10.0*ave
#            upper = histdat.max()
            if pol<=1:
                upper = 50000.0
            else:
                upper = 500.0
            
            ax.hist(histdat,bins=20,range=(0.0,upper))
            name = fig_prefix + '_hist_auto_' + pname
            fig_name = output_path(name)
            plt.savefig(fig_name)
            plt.savefig(fig_name)
            plt.close()
            ndata = len(histdat)
            for n in range(ndata):
                if histdat[n]>upper:
                    blord = autocorr[n]
                    print("Overflow=",histdat[n],upper,pol,blorder[blord])

#            print("cross_corr=",pol,histdat)
            histdat = vislvl[pol,crosscorr]
            plt.figure()
            ax = plt.axes()
            ax.set_title(pname+' Cross-correlations')
            ave = np.mean(histdat)
            upper = 10.0*ave
#            upper = histdat.max()
            if pol<=1:
                upper = 200
            else:
                upper = 100
            ax.hist(histdat,bins=50,range=(0.0,upper))
            name = fig_prefix + '_hist_cross_' + pname
            fig_name = output_path(name)
            plt.savefig(fig_name)
            plt.savefig(fig_name)
            plt.close()
            ndata = len(histdat)
            for n in range(ndata):
                if histdat[n]>upper:
                    blord = crosscorr[n]
                    print("Overflow=",histdat[n],upper,pol,blorder[blord])
        
        return super(Stack, self).process(ts)

  


