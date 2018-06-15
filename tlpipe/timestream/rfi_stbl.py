"""rfi Cleaning.

Inheritance diagram
-------------------

.. inheritance-diagram:: RfiStbl
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


 
class RfiStbl(timestream_task.TimestreamTask):


    params_init = {
                    'plot_map': False,
                    'plot_dets': False, # plot rfi detections
                    'plot_sigs': False,
                    'figure_dir':'output/rfi_stbl/',
                    'fract_req': 0.5,
                    'ttol_fact': 2.0,
                    'ftol_fact': 1.0,
                    'tftol_fact': 1.0,
                    'ttol_extra': 1.0,
                    'ftol_extra': 1.0,
                    'tftol_extra': 1.0,
                  }

    prefix = 'lf_'


    def process(self, ts):

        logger.info("Start RfiStbl")


#Check axis order.  Must be time x frequency x baseline. No reordering
        axes = ts.main_data_axes
        ok = True
        if axes[0] != 'time':
            ok = False
        if axes[1] != 'frequency':
            ok = False
        if axes[2] != 'baseline':
            ok = False
        if not ok:
            raise RuntimeError('Axis order must be time,frequency,baseline')

#Check for noise source on/off array
        if not 'ns_on' in ts.iterkeys():
            raise RuntimeError('No noise source info, can not do noise source calibration')
#        print("keys=",ts.keys())
        
        fract_req = self.params['fract_req']
        ttol_fact = self.params['ttol_fact']
        ftol_fact = self.params['ftol_fact']
        tftol_fact = self.params['tftol_fact']
        ttol_extra = self.params['ttol_extra']
        ftol_extra = self.params['ftol_extra']
        tftol_extra = self.params['tftol_extra']
        plot_dets = self.params['plot_dets']
        plot_sigs = self.params['plot_sigs']
        plot_map = self.params['plot_map']
        siglim = (0.1,0.1)
        ns_on = ts['ns_on'][:]
        vis = ts['vis']
        vis_mask = ts['vis_mask']
#        print("vis_mask.dtype=",vis_mask.dtype,vis_mask.shape)
        number = np.count_nonzero(vis_mask)
        print("masked=",number)
        blorder = ts['blorder']
        ntpt = vis.shape[0]
        nfpt = vis.shape[1]
        nbl = vis.shape[2]
#        print("ntpt=",ntpt,"nfpt=",nfpt,"nbl=",nbl)

        nreq = fract_req*nbl
        print("nreq=",nreq)

        #find edges of source source excitation
        edge = []
        edge = edge + [0]  
        for nt in range(ntpt-1):
            if ns_on[nt] != ns_on[nt+1]:
                edge = edge + [nt+1]
        edge = edge + [ntpt]
        edge = np.asanyarray(edge)
        nedge = len(edge)
        print ("nedge=",nedge,"edge=",edge)

        #Define required arrays
        tcen = np.zeros([ntpt,nbl],np.int32)
        twid = np.zeros([ntpt,nbl],np.int32)

        tplt = np.zeros(ntpt,np.int32)
        fplt = np.zeros([2,nfpt],dtype=np.float32)
        tfplt = np.zeros([ntpt,nfpt],dtype=np.int32)

        sigmaf = np.zeros([nfpt-2,nbl],dtype=np.float32)
        sigmaq = np.zeros([nfpt-2,nbl],dtype=np.float32)
        fcen = np.zeros([nfpt,nbl],dtype=np.int32)
        fwid = np.zeros([nfpt,nbl],dtype=np.int32)
#        sigp = tftol_fact
#        sigw = 1.0
        #Count rfi instances found (for diagnostics)
        tcount = 0
        fcount = 0
        count = 0
        nsoncnt = 0
        nsoffcnt = 0

        #Define parameters for histograms
        nsigma = 5
        ntbin = 50
        tbinhioff = nsigma*ttol_fact
        tbinhion = ttol_extra*tbinhioff
        tbinhion = np.ceil(tbinhion)
        tbinhioff = np.ceil(tbinhioff)
        tbinon = tbinhion/ntbin
        tbinoff = tbinhioff/ntbin
        nfbin = 50
        fbinhioff = nsigma*ftol_fact
        fbinhion = ftol_extra*fbinhioff
        fbinhion = np.ceil(fbinhion)
        fbinhioff = np.ceil(fbinhioff)
        fbinon = fbinhion/nfbin
        fbinoff = fbinhioff/nfbin
        ntfbin = 100
        tfbinhioff = nsigma*tftol_fact
        tfbinhion = tftol_extra*tfbinhioff
        tfbinhion = np.ceil(tfbinhion)
        tfbinhioff = np.ceil(tfbinhioff)
        tfbinon = tfbinhion/ntfbin
        tfbinoff = tfbinhioff/ntfbin
        twgton = np.zeros(ntbin,dtype=np.int32)
        twgtoff = np.zeros(ntbin,dtype=np.int32)
        fwgton = np.zeros(nfbin,dtype=np.int32)
        fwgtoff = np.zeros(nfbin,dtype=np.int32)
        tfwgton = np.zeros(ntfbin,dtype=np.int64)
        tfwgtoff = np.zeros(ntfbin,dtype=np.int64)
        #Loop over intervals noise source on / noise source off

#        for ne in range(nedge-1):
        for ne in range(18,20):
            n1 = edge[ne]
            n2 = edge[ne+1]
            noise = ns_on[n1]

            print("Looking in range",n1," to ",n2-1)
            ntmax = n2 - n1
            avef = np.float(ntmax)
            sqavef =np.sqrt(avef)
            sigpm = ftol_fact/sqavef
#            print("sigpm=",sigpm)
            if noise:
                sigpm = sigpm*ftol_extra
#            print("sigpm=",sigpm)
            tcen[:,:] = 0
            twid[:,:] = 0

            for bl in range(nbl):
#
#First look for brief spikes in the time dimension for visibilities 
#summed over all frequencies
#

                #Get absolute value of visibilities for this time x freq
                msk = vis_mask[n1:n2,:,bl] >= 2
                #Copy to masked array temp
                temp = np.ma.array(abs(vis[n1:n2,:,bl]),mask=msk)
                #Average over frequency
                fave = np.mean(temp,axis=1)
                #Get differences between adjacent time bins
                diff = np.diff(fave)
                #Now get second differences in time
                quad = np.diff(diff)

                #sigt is the rms deviation of time differences
                sigt = mstats.trimmed_std(quad,siglim,ddof=1)
                 #Check for valid sigma
                if sigt<=0.0 or sigt==np.ma.core.MaskedConstant:
                    #No.  Just mask everything and continue with next bl
                    vis_mask[n1:n2,:,bl] |= 2
                    continue
                if ntmax<5:
                    vis_mask[n1:n2,:,bl] |= 2
                    continue
                #print("sigt=",sigt,sigtpr,n1,n2,blorder[bl])
                #ttol is the threshhold for rfi detection based of 
                #frequency averaged time differences
                ttol = ttol_fact*sigt
                if noise:
                    ttol = ttol*ttol_extra

                for nt in range(1,ntmax-3):
                    if quad.mask[nt]:
                        vis_mask[nt,:,bl] |= 2
                        continue
                       #Look for maximum slope change=minimum in quad
                    if quad[nt]>quad[nt+1]:
                        continue
                    if quad[nt]>quad[nt-1]:
                        continue
#                    print("nt=",nt,"quad=",quad[nt-1],quad[nt],quad[nt+1],\
#                                  ttol)
#test is True if point does NOT have rfi signature
                    test = -quad[nt]<ttol
                    if plot_sigs:
                        if noise:
                            ib = np.int32(-quad[nt]/(sigt*tbinon))
                        else:
                            ib = np.int32(-quad[nt]/(sigt*tbinoff))

                        if ib>=0 and ib<ntbin:
                            if noise:
                                twgton[ib] += 1
                            else:
                                twgtoff[ib] += 1
#                    print("nt=",nt,"quad=",quad[nt-1],quad[nt],quad[nt+1],\
#                                 test)
#If not significant, try adding adjacent time bin to get more significance
                    if test:
                        add = quad[nt-1]
                        if quad[nt+1]<add:
                            add = quad[nt+1]
                            test = -(quad[nt]+add)<ttol
                    if test:
                        continue

                    tcen[nt+1,bl] = 1
                    for i in range(nt,ntmax-3):
                        if quad[i]<0.0:
                            twid[i+1,bl] = 1
                        else:
                            break
#check n1-1
                    for i in range(nt-1,-1,-1):
                        if quad[i]<0.0:
                            twid[i+1,bl] = 1
                        else:
                            break
#Special for first bin and next to first bin
                if tcen[2,bl] == 1:
                    tcen[1,bl] = 1
                else:
                    if -quad[0]>ttol:
                        tcen[1,bl] = 1
                if tcen[1,bl] == 1:
                    tcen[0,bl] = 1
                else:
                    if -diff[0]>ttol:
                        tcen[0,bl] = 1
#Special for next to last and last bins
                if tcen[ntmax-3,bl] == 1:
                    tcen[ntmax-2,bl] = 1
#                    print("flag n-2")
                else:
                    if -quad[ntmax-3]>ttol:
                        tcen[ntmax-2,bl] = 1
#                       print("flag n-2",quad[ntmax-3],ttol)
                if tcen[ntmax-2,bl] ==1:
                    tcen[ntmax-1,bl] = 1
#                    print("flag n-1")
                else:
                    if diff[ntmax-2]>ttol:
                        tcen[ntmax-1,bl] = 1
#                        print("flag n-1",diff[ntmax-2],ttol)
            sumc = np.sum(tcen[0:ntmax],axis=1)
            sumw = np.sum(twid[0:ntmax],axis=1)
            for nt in range(0,ntmax):
                if sumc[nt]<nreq or sumw[nt]<nreq:
                    continue
                tcount = tcount + 1
                print("time=",n1+nt,"sumc=",sumc[nt],"sumw=",sumw[nt])
            #mask point fi+1 for all baselines
                vis_mask[nt+n1,:,:] |= 2
            tplt[n1:n2] = sumc[0:ntmax]
    
#
#Next look for narrow band spikes in the frequency dimension for visibilities 
#summed over all times in the interval
#       


            fcen[:,:] = 0
            fwid[:,:] = 0
       
            for bl in range(nbl):
                msk = vis_mask[n1:n2,:,bl] >=2
                #Copy to masked array temp
                temp = np.ma.array(abs(vis[n1:n2,:,bl]),mask=msk)
 
                tave  = np.mean(temp,axis=0)
                diff = np.diff(tave)
                quad = np.diff(diff)

                sig1 = 0.0
                sig2 = 0.0
                for fi in range(-2,nfpt-2):
                   #sig3 = np.std(vis[n1:n2,fi+2,bl],ddof=1)
                    sig3 = mstats.trimmed_std(temp[:,fi+2],siglim,ddof=1)
                    if sig3==np.ma.core.MaskedConstant:
                        sig3 = 0.0

                    if sig1>0.0 and sig2>0.0 and sig3>0.0:
                        sigmaq[fi,bl] = np.sqrt(sig1**2+4.0*sig2**2+sig3**2)
                    else:
                        if fi>=0:
                            sigmaq[fi,bl] = 0.0
                    sig1 = sig2
                    sig2 = sig3
                    
                    if fi<1 or fi>=nfpt-3:
                        continue
                #for fi in range(1,nfpt-3):
                       #Look for maximum slope change=minimum in quad
                    if quad[fi]>quad[fi+1]:
                        continue
                    if quad[fi]>quad[fi-1]:
                        continue

#test is True if point does NOT have rfi signature
                    test = -quad[fi]/sigmaq[fi,bl]<sigpm
#                    if fi==2 or fi==194 or fi==354:
#                        print("blorder=",fi,bl,blorder[bl])
#                        print("tave=",tave[fi-2],tave[fi-1],tave[fi],tave[fi+1])
#                        print("cont ",tave[fi+2],tave[fi+3],tave[fi+4])
#                        print("diff=",bl,diff[0],diff[1],diff[2],diff[3])
#                        print("quads",bl,quad[1],quad[2],quad[3])
#                        print("sigma",bl,sigmaq[1,bl],sigmaq[2,bl])
                    if plot_sigs:
                        if noise:
                            ib = np.int32(-sqavef*quad[fi] \
                                               /(sigmaq[fi,bl]*fbinon))
                        else:
                            ib = np.int32(-sqavef*quad[fi] \
                                               /(sigmaq[fi,bl]*fbinoff))

                        if ib>=0 and ib<nfbin:
                            if noise:
                                fwgton[ib] += 1
                            else:
                                fwgtoff[ib] += 1

#If not significant, try adding adjacent frequency bin to get more significance
                    if test:
                        add = quad[fi-1]
                        if quad[fi+1]<add:
                            add = quad[fi+1]
                            test = -(quad[fi]+add)/sigmaq[fi,bl]<sigpm

                    if test:
                        continue
                    fcen[fi+1,bl] = 1
#                    if fi==10 and ne==1:
#                        print(bl,test,fcen[fi+1,bl])
                    for i in range(fi,nfpt-2):
                        if quad[i]<0.0:
                            fwid[i+1,bl] = 1
                        else:
                            break
                    for i in range(fi-1,0,-1):
                        if quad[i]<0.0:
                            fwid[i+1,bl] = 1
                        else:
                            break
            
#                        print("Center=",fi,quad[fi-1],quad[fi],quad[fi+1])
            sumc = np.sum(fcen,axis=1)
            sumw = np.sum(fwid,axis=1)
            if noise:
                nsoncnt += 1
                fplt[0,:] += sumc[:]
            else:
                nsoffcnt += 1
                fplt[1,:] += sumc[:]

            for fi in range(1,nfpt-3):
#                if ne==1:
#                    print("fi=",fi,sumc[fi],nreq)
                if sumc[fi]<nreq or sumw[fi]<nreq:
                    continue
                fcount = fcount + 1
                print("freq=",fi,"sumc=",sumc[fi],"sumw=",sumw[fi])
            #mask point fi+1 for all baselines
                vis_mask[n1:n2,fi,:] = vis_mask[n1:n2,fi,:] | 2

#Last - Look for brief increases in the visibility over a narrow range
#  of time and frequency
            tftol = tftol_fact
#            print("tftol=",tftol)
            if noise:
                tftol = tftol*tftol_extra
 #           print ("tftol=",tftol)
            for nt in range(ntmax):
                fcen[:,:] = 0
                fwid[:,:] = 0
                for bl in range(nbl):
 #                   print("baseline=",bl)
                    temp = abs(vis[nt+n1,:,bl])
                    diff = np.diff(temp)
                    quad = np.diff(diff)
                    signif = -quad/sigmaq[:,bl]
  
                    for fi in range(1,nfpt-3):
                        if vis_mask[nt+n1,fi+1,bl]>=2:
                            continue
                       #Look for maximum slope change=minimum in quad
                        if quad[fi]>quad[fi+1]:
                            continue
                        if quad[fi]>quad[fi-1]:
                            continue
#                        print("fi=",fi,"quad=",quad[fi-1],quad[fi],quad[fi+1],\
#                                  signif[fi])
#test is True if point does NOT have rfi signature
                        test = signif[fi]<tftol

                        if plot_sigs:
                            if noise:
                                ib = np.int32(signif[fi]/tfbinon)
                            else:
                                ib = np.int32(signif[fi]/tfbinoff)

                            if ib>=0 and ib<ntfbin:
                                if noise:
                                    tfwgton[ib] += 1
                                else:
                                    tfwgtoff[ib] += 1

#If not significant, try adding adjacent frequency bin to get more significance
                        if test:
                            add = quad[fi-1]
                            if quad[fi+1]<add:
                                add = quad[fi+1]
#FIX THIS*******************************
                                test = -(quad[fi]+add)/sigmaq[fi,bl]<tftol
#                                if not test:
#                                    print("fi=",fi,"quad=",quad[fi-1], \
#                                              quad[fi],quad[fi+1],add, \
#                                              signif[fi])

                        if test:
                            continue
                    
                        fcen[fi+1,bl] = 1
                        for i in range(fi,nfpt-2):
                            if quad[i]<0.0:
                                fwid[i+1,bl] = 1
                            else:
                                break
                        for i in range(fi-1,0,-1):
                            if quad[i]<0.0:
                                fwid[i+1,bl] = 1
                            else:
                                break
                        
#                        print("Center=",fi,quad[fi-1],quad[fi],quad[fi+1])
                sumc = np.sum(fcen,axis=1)
                sumw = np.sum(fwid,axis=1)
                for fi in range(1,nfpt-3):
                    if sumc[fi]<nreq or sumw[fi]<nreq:
                        continue
                    count = count + 1
                    print("time,freq=",n1+nt,fi,"sumc=",sumc[fi],"sumw=",sumw[fi])
                    #mask point (t,fi) for all baselines
                    vis_mask[nt+n1,fi,:] = vis_mask[nt+n1,fi,:] | 2
                tfplt[nt+n1,:] = sumc[:]

        logger.info("Count of frequencies masked by rfi %i" % fcount)
        logger.info("Count of times masked by rfi %i" % tcount)
        logger.info("Count of frequency-time points masked by rfi %i" % count)

        if plot_dets:
            plt.figure()
            plt.plot(range(ntpt),tplt)
            fig_name = output_path('rfi_stbl/rfi_time.png')
            plt.savefig(fig_name)
            plt.close('all')

            plt.figure()
            fplt[0,:] /= nsoncnt
            plt.plot(range(nfpt),fplt[0,:])
            fig_name = output_path('rfi_stbl/rfi_freq_nson.png')
            plt.savefig(fig_name)
            plt.close('all')

            plt.figure()
            fplt[1,:] /= nsoffcnt
            plt.plot(range(nfpt),fplt[1,:])
            fig_name = output_path('rfi_stbl/rfi_freq_nsoff.png')
            plt.savefig(fig_name)
            plt.close('all')

            plt.figure()
            plt.pcolormesh(tfplt.transpose())
            fig_name = output_path('rfi_stbl/rfi_time_and_freq.png')
            plt.savefig(fig_name)
            plt.close('all')

            if plot_map:
#        vis_mask[0:3600,10:11,0] = vis_mask[0:3600,10:11,0] | 2
#        vis_mask[3200:3210,:,0] = vis_mask[3200:3210,:,0] | 2
                mask_map = vis_mask[:,:,0].transpose(axes=[0,1])
                plt.figure(figsize=[3,12],dpi=600)
                plt.pcolormesh(mask_map)
                fig_name = output_path('rfi_stbl/rfi_map.png')
                plt.savefig(fig_name,bbox_inches='tight',pad_inches=0.5)
                plt.close('all')

            if plot_sigs:
                plt.figure()
                half = 0.5*tbinon
                mid = np.arange(half,tbinhion,tbinon)
                bins = np.arange(0.0,tbinhion-half,tbinon)
                plt.hist(mid,bins=bins,weights=twgton)
                fig_name = output_path('rfi_stbl/time_sigs_nson.png')
                plt.savefig(fig_name)
                plt.close('all')
                
                plt.figure()
                half = 0.5*tbinoff
                mid = np.arange(half,tbinhioff,tbinoff)
                bins = np.arange(0.0,tbinhioff-half,tbinoff)
                plt.hist(mid,bins=bins,weights=twgtoff)
                fig_name = output_path('rfi_stbl/time_sigs_nsoff.png')
                plt.savefig(fig_name)
                plt.close('all')
        
                plt.figure()
                half = 0.5*fbinon
                mid = np.arange(half,fbinhion,fbinon)
                bins = np.arange(0.0,fbinhion-half,fbinon)
                plt.hist(mid,bins=bins,weights=fwgton)
                fig_name = output_path('rfi_stbl/freq_sigs_nson.png')
                plt.savefig(fig_name)
                plt.close('all')
                
                plt.figure()
                half = 0.5*fbinoff
                mid = np.arange(half,fbinhioff,fbinoff)
                bins = np.arange(0.0,fbinhioff-half,fbinoff)
                plt.hist(mid,bins=bins,weights=fwgtoff)
                fig_name = output_path('rfi_stbl/freq_sigs_nsoff.png')
                plt.savefig(fig_name)
                plt.close('all')
                
                plt.figure()
                half = 0.5*tfbinon
                mid = np.arange(half,tfbinhion,tfbinon)
                bins = np.arange(0.0,tfbinhion-half,tfbinon)
                plt.hist(mid,bins=bins,weights=tfwgton)
                fig_name = output_path('rfi_stbl/timefreq_sigs_nson.png')
                plt.savefig(fig_name)
                plt.close('all')
                
                plt.figure()
                half = 0.5*tfbinoff
                mid = np.arange(half,tfbinhioff,tfbinoff)
                bins = np.arange(0.0,tfbinhioff-half,tfbinoff)
                plt.hist(mid,bins=bins,weights=tfwgtoff)
                fig_name = output_path('rfi_stbl/timefreq_sigs_nsoff.png')
                plt.savefig(fig_name)
                plt.close('all')
        
#                plt.figure()
#                half = 0.5*tfbinon
#                mid = np.arange(half,tfbinhion,tfbinon)
#                bins = np.arange(0.0,tfbinhion-half,tfbinon)
#                plt.hist(mid,bins=bins,weights=fwgton)
#                fig_name = output_path('rfi_stbl/freq_sigs_nson.png')
#                plt.savefig(fig_name)
#                plt.close('all')

        print("vis_mask.dtype=",vis_mask.dtype,vis_mask.shape)
        number = np.count_nonzero(vis_mask)
        print("masked=",number)

        bsamp = (4,5,8,33,34,35,36,57,58,73,74,75,76,105,106,107,108, \
                     216,217,218,219,220,221)
        non = np.zeros(10,dtype=np.int32)
        noff = np.zeros(10,dtype=np.int32)
        sumon = np.zeros(10,dtype=np.float32)
        sumvon = np.zeros(10,dtype=np.float32)
        sumoff = np.zeros(10,dtype=np.float32)
        sumvoff = np.zeros(10,dtype=np.float32)
        frange = range(10)
        for fi in frange:
            print("fi=",fi)
            for bl in bsamp:
                print("bl=",bl)
                for ne in range(nedge-1):
                    n1 = edge[ne]
                    n2 = edge[ne+1]
                    noise = ns_on[n1]
                    temp = abs(vis[n1:n2,fi,bl])
#                    print("fi=",fi,"bl=",bl,"ne=",ne,temp[n2-n1-1])
                    tempsq = temp*temp
                    if noise:
                        non[fi] += (n2-n1)
                        sumon[fi] += np.sum(temp)
                        sumvon[fi] += np.sum(tempsq)
                    else:
                        noff[fi] += (n2-n1)
                        sumoff[fi] += np.sum(temp)
                        sumvoff[fi] += np.sum(tempsq)
        print("non=",non)
        print("noff=",noff)

        print("sumon=",sumon)
        print("sumoff",sumoff)
        print("noise on")
        for n in range(10):
            sumon[n] = sumon[n]/non[n]
            sumvon[n] = sumvon[n]/non[n] - sumon[n]*sumon[n]
            sumvon[n] = np.sqrt(sumvon[n]/(non[n]-1))
            line = "%i,%.4f,%.4f" % (n,sumon[n],sumvon[n])
            print(line)
        print("noise off")
        for n in range(10):
            sumoff[n] = sumoff[n]/noff[n]
            sumvoff[n] = sumvoff[n]/noff[n] - sumoff[n]*sumoff[n]
            sumvoff[n] = np.sqrt(sumvoff[n]/(noff[n]-1))
            line = "%i,%.4f,%.4f" % (n,sumoff[n],sumvoff[n])
            print(line)
        
        tbad = np.zeros(9,dtype=np.float32)
        tgood = np.zeros(9,dtype=np.float32)
        numbad = 0
        numgood = 0
        for nt in range(2155,2164):
            for bl in range(nbl):
                x = 6
                temp = abs(vis[nt,0:nfpt,bl])
                base = blorder[bl]
                base0 = base[0]
                base1 = base[1]
                if base0==base1:
                    continue
                good = base0<=16 and base1>16
                if good:
                    tgood[nt-2155] += np.mean(temp)
                    numgood += 1
                else:
                    tbad[nt-2155] += np.mean(temp)
                    numbad += 1
            tgood[nt-2155] /= numgood
            tbad[nt-2155] /= numbad
            print ("time=",nt,tgood[nt-2155],tbad[nt-2155])

        return super(RfiStbl, self).process(ts)

  


