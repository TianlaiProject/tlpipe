"""Relative phase calibration using the noise source signal.

Inheritance diagram
-------------------

.. inheritance-diagram:: NsCal
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
import logging

def nsCalFitFun(x,*args):
    """Function to be minimized by NsCalg least squares fit to 
    noise source visibilities.
    """
    ai = np.complex128(complex(0.0,1.0))
    Vmat,Verr = args
    #Get the number of feeds being fit
    ndim = Vmat.shape[0]
    #Express gains as complex numbers
    g = np.zeros(ndim,dtype=np.complex128)
    #g[0] is taken arbitrarily to have phase=0.  This is allowed because
    #only phase differences matter
    g[0] = complex(x[0],0.0)
    #For other g's real and imaginary parts are stored in x
    for i in range(1,ndim):
        g[i] = complex(x[2*i-1],x[2*i])
    #Number of parameters being fit
    npar = x.shape
    #jac holds derivatives of the function
    jac = np.zeros(npar,dtype=np.float64)
    #The calibrated visibilities should be = 1
    #So the uncalibrated visibilities should be g x g* 
    Vfit = np.outer(g,g.conj())
    #Difference between fitted and measured visibilities
    diff = Vfit - Vmat
    #Initialize chisq for summing
    chisq = np.float64(0.0)
    #Add contribution for valid data
    if Verr[0,0] > 0:
        #Estimated error for visibility (0,0)
        sigsq = Verr[0,0]*Verr[0,0]
        #Visibility (0,0) term in derivative for parameter 0
        jac[0] = 2.0*x[0]*diff[0,0].real/sigsq
    #Now complete derivatives and chi-squared for the remaining parameters
    for i in range(ndim):
        if i > 0 and Verr[i,i]>0:
            sigsq = Verr[i,i]*Verr[i,i]
            jac[2*i-1] = jac[2*i-1] + 2.0*x[2*i-1]*diff[i,i].real/sigsq
            jac[2*i] = jac[2*i] + 2.0*x[2*i]*diff[i,i].real/sigsq
        for j in range(ndim):
            if Verr[i,j] <= 0:
                continue;
            if j>=i:
                dev = abs(diff[i,j])/Verr[i,j]
                chisq = chisq + dev*dev
            if i == j:
               continue
            gv = g[j]*Vmat[i,j]
            vr = Vfit[j,j].real
            
            sigsq = Verr[i,j]*Verr[i,j]
            if i == 0:
                vterm = g[j]*(x[0]*g[j].conj()-Vmat[0,j])
                jac[0] = jac[0] + (x[0]*vr-gv.real)/sigsq
            else:
                vterm = g[j]*(x[2*i-1]*g[j].conj()-Vmat[i,j])
                jac[2*i-1] = jac[2*i-1] + vterm.real/sigsq
                vterm = g[j]*(ai*x[2*i]*g[j].conj()-Vmat[i,j])
                jac[2*i] = jac[2*i] + vterm.imag/sigsq
#    print("fun chisq=",chisq)
#    print("fun jac=",jac[:])
    jac[:] = 2.0*jac[:]
    #return chi-squared and derivatives
    return (chisq, jac)

logger = logging.getLogger(__name__)

 
class NsCalg(timestream_task.TimestreamTask):


    params_init = {
        'min_nsrc': 5, #minimum number of epochs of noise source for calibration
        'min_signif': 10.0, #minimum S/N for valid noise signal
        'max_chisq':1.e+10, #Maximum chi-squared for acceptable fit
        'fit_tol':1.e-2,  #Tolerance on derivatives at chi-squared maximum
        'fig_dir': 'ns_calg', #Directory for plots
        'plot_corr': False, #True to plot visibility correlations
        'plot_fit' : False, #True to plot fit results
        'plot_adj' : False, #True to plot adjacent freqs on fit results
        'plot_resp' : False, #True to plot noise source response vs frequency
        'plot_chisq' : False, #True to plot chi-squared of fits
        'bl_incl': 'all', #baselines to plot (list or 'all')
        'bl_excl': [], # baselines not plotted if bl_incl='all'
        'freq_incl': 'mid', #frequency bins to plot (list,'mid', or 'all')
        'freq_excl': [], #frequency bins not plotted if freq_incl='all'
        'time_incl': 'mid', #noise source epochs to plot (list,'mid', or 'all')
        }

    prefix = 'nc_'

    def process(self, rt):
        logger.info("Start noise source calibration module ns_calg")

        assert isinstance(rt, RawTimestream), '%s only works for RawTimestream object currently' % self.__class__.__name__

        if not 'ns_on' in rt.iterkeys():
            raise RuntimeError('No noise source info, can not do noise source calibration')
        
#Check axis order.  Must be time x frequency x baseline. No reordering
        axes = rt.main_data_axes
        ok = True
        if axes[0] != 'time':
            ok = False
        if axes[1] != 'frequency':
            ok = False
        if axes[2] != 'baseline':
            ok = False
        if not ok:
            raise RuntimeError('Axis order must be time,frequency,baseline')
        #minimum number of required noise source epochs
        min_nsrc = self.params['min_nsrc']
        #list of baselines present in visibility array
        bl_list = rt['blorder'][:].tolist()
        #noise source parameters
        on_time = rt['ns_on'].attrs['on_time']
        off_time = rt['ns_on'].attrs['off_time']
        period = rt['ns_on'].attrs['period']
        #The noise source off time is required to be at least twice the on time
        #This restriction comes from the way the time correlation plots are
        #calculated.  It could be relaxed, but we probably will always satisfy
        #this criteria anyway.
        if off_time<=2*on_time:
            message = 'Noise source 2 X on time (%i) must < off time (%i)' \
                % (2*on_time,off_time)
            raise RuntimeError(message)
        #Get array dimensions
        ntpt = rt.vis.shape[0]
        nfpt = rt.vis.shape[1]
        nbl = rt.vis.shape[2]
        #Get array the specifies noise source on/off
        ns_on = rt['ns_on'][:]
        #Initialize time bins where noise source turns on
        tpts = []
        half = (on_time+1)/2
#Loop for transistions: noise source off->noise source on
#Record first time bin with noise source on
#Useful noise sources have half time bins before and after noise source,
#so restrict the search to that range
        for i in range(half,ntpt-on_time-half):
            if not ns_on[i] and ns_on[i+1]:
                tpts = tpts + [i+1]
        tpts = np.asanyarray(tpts)
#        print ("tpt=",tpts)
        #check that there are enough noise source epochs
        nsrc = len(tpts)
        if nsrc<min_nsrc:
            message = 'Not enough noise source epochs'
            RuntimeError(message)

        vnsrc = np.full([nsrc,nfpt,nbl],(0.0+9.99j),dtype=np.complex64)
        vnsrc_err = np.full([nsrc,nfpt,nbl],-1.0,dtype=np.float32)


#Feeds should be numbered starting with 1.  If not, it is a fatal error
        minbl = min(bl_list)
        if minbl[0]<=0 or minbl[1]<=0:
            raise RuntimeError("Error in baseline numbers")
#Find the largest feed number present.
        maxbl = max(bl_list)
        maxfeed = max(maxbl[0],maxbl[1]) + 1
#Build the ind array which is a lookup for fit parameter ordinal as a 
#function of baseline number.  Missing feeds are indicated by ind = -1
#We need an extra space in the ind array that starts with zero
#So ind[0] is always=-1
        ind = np.zeros(maxfeed,np.int)
        for bl in bl_list:
            n = bl[0]
            ind[n] = 1
            n = bl[1]
            ind[n] = 1

            nfeed = 0
        for n in range(maxfeed):
            if ind[n] <= 0:
                ind[n] = -1
            else:
                ind[n] = nfeed
                nfeed = nfeed + 1
#Reserve storage for chi-squared and fitted gains
        chisq = np.zeros([nsrc,nfpt],dtype=np.float32)
        gfit = np.zeros([nsrc,nfpt,nfeed],dtype=np.complex64)
        fitres = {'bl_list':bl_list,'ind':ind,'chisq':chisq,'gfit':gfit}
#        print("fitres.ind",fitres['ind'])

#       Is this or something like this needed?
#        rt.data_operate(self.cal, op_axis=None, axis_vals=0, full_data=True, copy_data=False, keep_dist_axis=False)

        #Extract noise source visibilities 
        self.__nsCalExtract(tpts,rt,vnsrc,vnsrc_err)
        #Redefine tpts to be floating point and middle of noise source on range
        tpts = tpts + (on_time+1.0)/2.0
#        print ("tpts=",tpts)
        #Do fit for gains
        self.__nsCalFit(tpts, vnsrc, vnsrc_err, rt, fitres)
        #Apply gain correction and make plots, if requested
        self.__nsCalFinish(tpts, vnsrc, vnsrc_err, rt,fitres)

        return super(NsCalg, self).process(rt)

    def __nsCalExtract(self, tpt, rt, vnsrc, vnsrc_err):
        """The values of the noise source response are extracted from the 
        visibility time sequence by the subtraction of the noise on 
        and noise off data 
        """
        nsrc = len(tpt)

# TO DO-->Check for masked visibilities

        on_time = rt['ns_on'].attrs['on_time']
        half = on_time/2
        afact = 1.0/(on_time-1.0)
        bfact = 0.25/(half-1.0)
        nfpt = rt.vis.shape[1]
        nbl = rt.vis.shape[2]
        for n in range(nsrc):

            llo = tpt[n] - half
            lhi = tpt[n]
            slo = lhi
            shi = slo + on_time
            tlo = shi
            thi = shi + half
#        print ("ns l=",rt['ns_on'][llo:lhi])
#        print ("ns s=",rt['ns_on'][slo:shi])
#        print ("ns t=",rt['ns_on'][tlo:thi])
            for fi in range(nfpt):
                for nb in range(nbl):
                    vl = np.mean(rt.vis[llo:lhi,fi,nb])
                    vs = np.mean(rt.vis[slo:shi,fi,nb])
                    vt = np.mean(rt.vis[tlo:thi,fi,nb])
                    vlsig = np.std(rt.vis[llo:lhi,fi,nb],ddof=1)
                    vssig = np.std(rt.vis[slo:shi,fi,nb],ddof=1)
                    vtsig = np.std(rt.vis[tlo:thi,fi,nb],ddof=1)
#                    if fi == 256 and nb == 5:
#                        print("vl=",vl,vlsig,"vs=",vs,vssig,"vt=",vt,vtsig)
                    vnsrc[n,fi,nb] = vs - 0.5 * (vl+vt)
                    vsq = afact*vssig*vssig \
                        + bfact*(vtsig*vtsig + vlsig*vlsig)
                    if vsq<0.001:
                        vsq = 0.001
                    vnsrc_err[n,fi,nb] = np.sqrt(vsq)
        return

    def __nsCalFit(self, tpts, vnsrc, vnsrc_err, rt,fitres):
        """ 
A least squares fit to the noise source visibilities is made to determine
a gain for each feed such that the gain corrected visibilities will be 
(approximately) complex(1,0)
"""
        max_chisq = self.params['max_chisq']
        fit_tol = self.params['fit_tol']
        min_signif = self.params['min_signif']
        bl_list = fitres['bl_list']
        ind = fitres['ind']
        maxfeed = len(ind)
        chisq = fitres['chisq']
        gfit = fitres['gfit']
        nsrc = vnsrc.shape[0]
        nfpt = vnsrc.shape[1]
        nbl = vnsrc.shape[2]
        ntpt = rt.vis.shape[0]
        nfeed = gfit.shape[2]      

        Vmat = np.zeros([nfeed,nfeed],dtype=np.complex128)
        Verr = np.zeros([nfeed,nfeed],dtype=np.float64)
        #Loop over frequency bins
        for fi in range(nfpt):
            #Count of bad time epochs for this frequency
            nbadnt = 0
            #List of bad feeds for this frequency
            badlst = []
            #Loop over noise source epochs
            for nt in range(nsrc):
                #Loop over baselines 
                for bl in bl_list:
                    #nb is the index into the visibility array
                    nb = bl_list.index(bl)
                    #Integers for feeds in this baseline
                    ib = bl[0]
                    jb = bl[1]
                    #i,j are indices of fitted gain.  Fitted gains
                    #are numbered consecutively, baselines, in general,
                    #can have gaps
                    i = ind[ib]
                    if i < 0:
                        continue
                    j = ind[jb]
                    if j < 0:
                        continue
                    #Place measured visibilities in V matrix
                    Vmat[i,j] = vnsrc[nt,fi,nb]
                    Vmat[j,i] = Vmat[i,j].conj()
                    #Flag invalid elements with error<0
                    if vnsrc_err[nt,fi,nb] <= 0.0:
                        Verr[i,j] = -1.0
                    else:
                        sig = abs(Vmat[i,j])/vnsrc_err[nt,fi,nb]
                        if sig > min_signif:
                            Verr[i,j] = vnsrc_err[nt,fi,nb]
                        else:
                            Verr[i,j] = -1.0
                    Verr[j,i] = Verr[i,j]
#Check that V[i,0] values are all valid.  This is necessary for the initial 
#approximation as coded, but is more restrictive than necessary
                badt = False
                for i in range(nfeed):
                    if Verr[i][0] <= 0.0:
                        badt = True
                        for bad in range(maxfeed):
                            if i==ind[bad]:
                                if bad not in badlst:
                                    badlst = badlst + [bad]
                                break
                if badt:
                    #Count a bad epoch for this frequency
                    nbadnt = nbadnt + 1
#                    message = "Skip f=%i t=%i bad=%s" \
#                        % (fi,nt,badlst)
#                    print(message)
                    #Chisq < 0 marks invalid fit
                    chisq[nt,fi] = -1.0
                    continue

                norm = np.sqrt(abs(Vmat[0,0]))
                g = np.zeros(nfeed,dtype=np.complex128)
                g = Vmat[:,0]/norm


                x = np.zeros(2*nfeed-1,dtype=np.float64)
                x[0] = g[0].real
                for i in range(1,nfeed):
                    x[2*i-1] = g[i].real
                    x[2*i] = g[i].imag
#                print("Initial approximation x=",x)

                args=(Vmat,Verr)

                result = opt.minimize(nsCalFitFun,x,args=args, \
                                          jac=True,tol=fit_tol)
                dof = (nfeed-1)*(nfeed-1)
                chidof = result.fun/dof
                #Issue debug message to logger if fit status is not "success".
                #So far, the only observed failure mode is due to a fit
                #tolerance that is lower than can be achieved with 64 bit
                #precision.  However, in these cases, the fit is valid, so
                #for now, the fit is accepted even when success=False
                if not result.success:
                    message = "Fit failed f=%i t=%i chidof=%.1f (%i) nit=%i %s" \
                      % (fi, nt, chidof, dof, result.success, result.nit, \
                             result.message)
                    logger.debug(message)


                # Save fit results
                chisq[nt,fi] = chidof
                x = result.x
                gfit[nt,fi,0] = complex(x[0],0.0)
                for nf in range (1,nfeed):
                    gfit[nt,fi,nf] = complex(x[2*nf-1],x[2*nf])
            #issue debug message to logger if fit was not performed
            #because of bad data
            if nbadnt>0:
                message = "fbin=%i has %i/%i bad noise source epochs" \
                    % (fi,nbadnt,nsrc)
                logger.debug(message)
                nbadch = len(badlst)
                if nbadch<=10:
                    message = "Bad channels=%s" % (badlst)
                else:
                    message = "Bad channels=%s and %i others" \
                        % (badlst[0:20],nbadch-10)
                logger.debug(message)

        return

    def __nsCalFinish(self, tpts, vnsrc, vnsrc_err, rt,fitres):
        """Plot the fit results and other quantities related to the noise 
        source calibration.  The actual gain correction is made here so that 
        the b-spline interpolation (used also for plotting) is only performed 
        once
        """
        #Maximum chi-squared for a valid fit
        max_chisq = self.params['max_chisq']
        #Minimum S/N for a valid noise source measurement
        min_signif = self.params['min_signif']
        #Directory (relative to output directory) for plot files
        fig_dir = self.params['fig_dir']
        on_time = rt['ns_on'].attrs['on_time']
        period = rt['ns_on'].attrs['period']
        #Plot flags
        plot_corr = self.params['plot_corr']
        plot_fit = self.params['plot_fit']
        plot_resp = self.params['plot_resp']
        plot_adj = self.params['plot_adj']
        plot_chisq = self.params['plot_chisq']
        #Baselines to plot
        bl_incl = self.params['bl_incl']
        #Baselines to exclude from plotting when bl_incl='all'
        bl_excl = self.params['bl_excl']
        #Frequencies to plot
        freq_incl = self.params['freq_incl']
        #Frequencies to exclude from plotting when freq_incl='all'
        freq_excl = self.params['freq_excl']
        #Noise source epochs to plot (first one is 0)
        time_incl = self.params['time_incl']
        tag_output_iter = self.params['tag_output_iter']
        iteration = self.iteration
        #list of baselines
        bl_list = fitres['bl_list']

        #number of noise source epochs
        nsrc = vnsrc.shape[0]
        #number of frequency bins
        nfpt = vnsrc.shape[1]
        #number of baselines
        nbl = vnsrc.shape[2]
        #number of time bins
        ntpt = rt.vis.shape[0]

        #Make list of baselines to be plotted
        if bl_incl == 'all':
            bls_plt = [ list(bl) for bl in rt.bl ]
        else:
            bls_plt = [ list(bl) for bl in bl_incl if not bl in bl_excl ]
        #Make sure requested baselines are present
        bls_plt = [ bl for bl in bls_plt if bl in bl_list ]
        
        
        #Make list of frequency bins to be plotted
        if freq_incl == 'all':
            freq_plt = range(nfpt)
        else:
            if freq_incl == 'mid':
                freq_plt = [nfpt/2]
            else:
                freq_plt = [ fi for fi in freq_incl if not fi in freq_excl ]
        #Make sure requested baselines are present
        freq_plt = [ fi for fi in freq_plt if fi in range(nfpt) ]
 
        #Make list of noise source epochs to be plotted
        if time_incl == 'all':
            time_plt = range(nsrc)
        else:
            if time_incl == 'mid':
                time_plt = [nsrc/2]
            else:
                time_plt = time_incl
       #Make sure requested noise source epochs are present
        time_plt = [ nt for nt in time_plt if nt in range(nsrc) ]
 
                
        #Get frequency array
        freq = rt.freq[:]
        #And endpoints
        fstart = freq[0]
        fstop = freq[-1]
        #its is the integer version of tpts (used to index plot data)
        its = tpts.astype(np.int)

        # See if Time correlation plots are requested                    
        if plot_corr:  
#            print("Do correlation plots")
            #Just do frequency at midpoint.  Don't use freq_plt selection
            fi = nfpt/2
            fpt = rt.freq[fi]
            #Number of time bin correlations to calculate (max=100)
            ncorr = min(100,period-2*on_time)
            #Create correlation vector
            corr = np.zeros(ncorr,dtype=np.complex64)
            #Create time bin vector
            tbin = range(1,ncorr+1)

            #Loop over baselines being plotted      
            for bl in bls_plt:
                #get index of this baseline into visibility array
                inx = bl_list.index(bl)
                #variance of this baseline
                var = 0.0
                #zero correlation vector for summation
                corr[:] = np.complex(0.0,0.0)
                #Loop over noise source epochs skipping first and last
                for n in range(1,nsrc-1):
                    #Get time bin range for this noise souce epoch
                    ki = its[n] + on_time
                    kf = ki + ncorr
                    #Get variance and correlations for this noise source epoch
                    for k in range(ki,kf):
                        vabs = np.abs(rt.vis[k,fi,inx])
                        var = var + vabs*vabs
                        for i in range(ncorr):
                            corr[i] = corr[i]  \
                                + rt.vis[k,fi,inx]*rt.vis[k+i+1,fi,inx].conj()
                #Normalize correlation vector to 1=complete correlation
                corr = corr/var
                #Decompose into amplitude and phase
                amp = np.abs(corr)
                phase = np.angle(corr)
                #Make amplitude/phase plot
                plt.figure()
                fig, ax = plt.subplots(2,sharex=True)
                ax[0].plot(tbin,amp,'r')
                vislabel = 'V(%d,%d)' % (bl[0],bl[1])
                ax0 = ax[0]
                plt.text(0.8,0.85,vislabel,transform=ax0.transAxes)
                flabel = '%5.1f Mhz' % fpt
                plt.text(0.8,0.75,flabel,transform=ax0.transAxes)
                ax[1].plot(tbin,phase)
                ax[0].set_ylabel(r'Correlation amplitude')
                ax[1].set_ylabel(r'Correlation phase')
                ax[1].set_xlabel('Time bin')
                fig_name = '%s/correlation_%dX%d.png' % (fig_dir,bl[0],bl[1])
                fig_name = output_path(fig_name)
                plt.text(0.8,0.85,vislabel,transform=ax0.transAxes)
                flabel = '%5.1f Mhz' % fpt
                plt.text(0.8,0.75,flabel,transform=ax0.transAxes)
                ax[1].plot(tbin,phase)
                ax[0].set_ylabel(r'Correlation amplitude')
                ax[1].set_ylabel(r'Correlation phase')
                ax[1].set_xlabel('Time bin')
                fig_name = '%s/correlation_%dX%d.png' % (fig_dir,bl[0],bl[1])
                fig_name = output_path(fig_name)
                plt.savefig(fig_name)
                plt.close('all')

        #See if we are making noise source response versus frequency plots
        if plot_resp:
#            print("Starting frequency plots")
            #Make min_signif into an array to keep plotting routines happy
            level = np.asarray(min_signif,dtype=np.float32)
            #Create legend text
            subtext = "Overall phase and delay subtracted"
            #Create arrays needed for plots
            amp = np.empty(nfpt,dtype=np.float32)
            angle = np.empty(nfpt,dtype=np.float32)
            sig = np.empty(nfpt,dtype=np.float32)
            wgt = np.empty(nfpt,dtype=np.float32)
            cond = np.empty(nfpt,dtype=np.float32)
            #Loop over baselines to be plotted
            for bl in bls_plt:
                inx = bl_list.index(bl)
                #Loop over noise source epochs to be plotted
                for i in time_plt:
                    for fi in range(nfpt):
                        amp[fi] = np.abs(vnsrc[i,fi,inx])
                        sig[fi] = amp[fi]/vnsrc_err[i,fi,inx]
                        angle[fi] = np.angle(vnsrc[i,fi,inx])
                        wgt[fi] = 1.0/vnsrc_err[i,fi,inx]
                        cond[fi] = np.abs(vnsrc[i,fi,inx])/vnsrc_err[i,fi,inx] \
                                 < min_signif
                    angle = np.unwrap(angle)
                    mangle = ma.masked_where(cond,angle)
                    nmsk = ma.count_masked(mangle)
                    nok = ma.count(mangle)
                    tsub = False
                    if nok>10:
                        fpoly = ma.polyfit(freq,mangle,1, \
                               w=wgt,full=False,cov=True)
                        angle = angle - fpoly[0][1] - fpoly[0][0]*freq
                        tsub = True
                    plt.figure()
                    fig, ax = plt.subplots(2,sharex=True)
                    ax0 = ax[0]
                    ax[0].plot(freq,sig,'r')
                    ax0.hlines(level,fstart,fstop)
                    vislabel = 'V(%d,%d)' % (bl[0],bl[1])
                    plt.text(0.8,0.85,vislabel,transform=ax0.transAxes)
                    ax[1].plot(freq,amp)
                    ax[0].set_ylabel(r'Noise level (sigma)')
                    ax[1].set_ylabel(r'Amplitude (uncalibrated)')
                    ax[1].set_xlabel('Frequency (MHz)')
                    fig_name = '%s/amplitude_%dX%d-%d.png' \
                        % (fig_dir,bl[0],bl[1],i)
                    fig_name = output_path(fig_name)
                    plt.savefig(fig_name)
                    plt.close('all')
                    #Skip phase plot for autocorrelation
                    if bl[0]==bl[1]:
                        continue
                    plt.figure()
                    fig, ax = plt.subplots(2,sharex=True)
                    ax[0].plot(freq,amp,'r')
                    ax0 = ax[0]
                    ax1 = ax[1]
                    vislabel = 'V(%d,%d)' % (bl[0],bl[1])
                    plt.text(0.8,0.85,vislabel,transform=ax0.transAxes)
                    ax0.set_ylabel(r'Amplitude (uncalibrated')
                    ax1.plot(freq,angle)
                    ax1.set_ylabel(r'Phase (radians)')
                    ax1.set_ylim([-5.0,5.0])
                    ax1.set_xlabel('Frequency (MHz)')
                    if tsub:
                        plt.text(0.75,0.90,"Subtracted",transform=ax1.transAxes)
                        sublabel = "phase=%.1f" % (fpoly[0][1])
                        plt.text(0.75,0.80,sublabel,transform=ax1.transAxes)
                        delay = 500.0*fpoly[0][0]/np.pi
                        sublabel = "delay=%.1f ns" % (delay)
                        plt.text(0.75,0.70,sublabel,transform=ax1.transAxes)
                    fig_name = '%s/phase_%dX%d-%d.png' % (fig_dir,bl[0],bl[1],i)
                    fig_name = output_path(fig_name)
                    plt.savefig(fig_name)
                    plt.close('all')
                
#Make gain corrections based on bspline interpolation of the fitted gains
#at the noise source epochs
        ind = fitres['ind']
        maxfeed = len(ind)
        chisq = fitres['chisq']
        gfit = fitres['gfit']

        nfeed = gfit.shape[2]
        gcorr = np.zeros([ntpt,nfeed],dtype=np.complex64)
        nord = 2
        #Loop over frequencies
        for fi in range(nfpt):
            
#Require good fit at all time points
#This could be relaxed, but in general, if there is a problem
#at one epoch it usually means that the noise source strength is
#an issue for calibration at the frequency bin=fi
            
            #bad counts the number of bad epochs
            bad = 0
            #Loop over noise source epochs
            for nt in range(nsrc):
                if chisq[nt,fi] < 0.0:
                    bad = bad + 1
                if chisq[nt,fi] > max_chisq:
                    bad = bad + 1
            #See if any bad epoch
            if bad > 0:
                    # Mask all times and baselines for this frequency
                    rt.vis_mask[:,fi,:] = rt.vis_mask[:,fi,:] | 2
                    message = "Noise source calibration failed for fbin=%i" \
                      % (fi)
                    logger.debug(message)
                    continue

            #Good fits for all epochs.  Loop over feeds 
            for n in range(maxfeed):
                #Fit result index for this feed
                nf = ind[n]
                #B-spline interpolation of real part
                bsplre = interp.make_interp_spline(tpts,gfit[:,fi,nf].real,nord)
                #interpolated curve for the real part
                greal= interp.BSpline(bsplre.t,bsplre.c,nord)
                #Same for imaginary part
                bsplim = interp.make_interp_spline(tpts,gfit[:,fi,nf].imag,nord)
                gimag= interp.BSpline(bsplim.t,bsplim.c,nord)
#                print("t=",bsplre.t)
#                print("c=",bsplre.c)
#                print("k=",bsplre.k)
#                print("xtrap=",bsplre.extrapolate)
#                print("axis=",bsplre.axis)
                tbin = np.arange(ntpt)
                gr = greal(tbin)
                gi = gimag(tbin)
                #Loop over time bins.  complex method only handles scalars
                for nt in range(ntpt):
                    gcorr[nt,nf]=np.complex(gr[nt],gi[nt])

#Correct each baseline

            for nb in range(nbl):
                bl = bl_list[nb]
                bl0 = bl[0]
                bl1 = bl[1]
                nf1 = ind[bl0]
                nf2 = ind[bl1]
                blcorr = gcorr[:,nf1]*gcorr[:,nf2].conj()
                #Divide by zero should never happen, but maybe a check should
                #be added just in case
                rt.vis[:,fi,nb]=rt.vis[:,fi,nb]/blcorr
            
                #Done if not plotting fit results
                if not plot_fit:
                    continue
                #Done if this frequency is not on the list
                if fi not in freq_plt:
                    continue
                #Done if this baseline is not on the list
                if bl not in bls_plt:
                    continue

                amp=np.abs(vnsrc[:,fi,nb])
                err=vnsrc_err[:,fi,nb]
                #Need a diagnostic here rather than just skipping plot
                if amp.argmin() < 1.e-6:
                    continue
                #Relative error in absolute gain.
                #This is also used as the phase error, an approximation that
                #is correct if the gain error is a circle in the complex plane
                perr = err/amp
                #Plot difference in amplitude relative to the mean
                aveamp = np.mean(amp)
                if aveamp>0.0:
                    amp = (amp-aveamp)/aveamp
                ampcurve = np.abs(blcorr)
                aveamp = np.mean(ampcurve)
                if aveamp>0.0:
                    ampcurve = (ampcurve-aveamp)/aveamp
                #Plot phase data and curve
                phase=np.angle(vnsrc[:,fi,nb])
                phscurve = np.angle(blcorr)

                bltup = tuple(bl)
                plt.figure()
                fig, ax = plt.subplots(2, sharex=True)
                ax_val = np.array([ datetime.fromtimestamp(sec) for sec in rt['sec1970'][:] ])
                xlabel = '%s' % ax_val[0].date()
                ax_val = mdates.date2num(ax_val)
                ax[0].plot(ax_val, ampcurve)
                ax[0].errorbar(ax_val[its], amp,fmt='ro',yerr=perr,ms=3)
                ax[0].set_ylabel(r'$\Delta |g|$')
                ax0 = ax[0]
                vislabel = 'V(%d,%d)' % (bl[0],bl[1])
                plt.text(0.8,0.85,vislabel,transform=ax0.transAxes)
                fpt = freq[fi]
                flabel = '%5.1f Mhz' % fpt
                plt.text(0.8,0.75,flabel,transform=ax0.transAxes)
                ax1 = ax[1]
                ax1.plot(ax_val, phscurve)
                ax1.errorbar(ax_val[its], phase, fmt='ro',yerr=perr,ms=3)
                # Optionally plot adjacent frequency bins
                if plot_adj:
                    if fi<nfpt-1:
                        phase=np.angle(vnsrc[:,fi+1,nb])
                        ax1.plot(ax_val[its], phase,'k^')
                    if fi>0:
                        phase=np.angle(vnsrc[:,fi-1,nb])
                        ax1.plot(ax_val[its], phase,'kv')
                #Label x axis with real time
                duration = (ax_val[-1] - ax_val[0])
                dt = duration / ntpt
                ext = max(0.05*duration, 5*dt)
                ax1.set_xlim([ax_val[0]-ext, ax_val[-1]+ext])
                ax1.xaxis_date()
                date_format = mdates.DateFormatter('%H:%M')
                ax1.xaxis.set_major_formatter(date_format)
                locator = MaxNLocator(nbins=6)
                ax1.xaxis.set_major_locator(locator)
                ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
                ax1.set_xlabel(xlabel)
                ax1.set_ylabel(r'$\Delta \phi$ / radian')
                fig_name = '%s/fit_%f_%d_%d.png' % (fig_dir, fpt, bl[0], bl[1])
                if tag_output_iter:
                    fig_name = output_path(fig_name, iteration=iteration)
                else:
                    fig_name = output_path(fig_name)

                plt.savefig(fig_name)
                plt.close('all')

        #See if chi-squared summary plot is requested
        if plot_chisq:
            for nt in time_plt:
                plt.figure()
                plt.plot(freq,chisq[nt,:])
                plt.xlabel(r'$\nu$ / MHz')
                plt.ylabel(r'$\chi^2$ / dof')
#                plt.legend(loc='best')
                fig_name = '%s/chisq_%d.png' % (fig_dir,nt)
                fig_name = output_path(fig_name)
                plt.savefig(fig_name)
                plt.close('all')

        return


    def cal(self, vis, vis_mask, li, gi, fbl, rt, **kwargs):
        """No longer used."""
        num_mean = self.params['num_mean']
        phs_only = self.params['phs_only']
        bls_plt = kwargs['bls_plt']
        freq_plt = kwargs['freq_plt']
        bl_list = kwargs['bl_list']
        vnsrc = kwargs['vnsrc']
        vnsrc_err = kwargs['vnsrc_err']
        ledge = kwargs['ledge']
        tedge = kwargs['tedge']
        
        return




 




