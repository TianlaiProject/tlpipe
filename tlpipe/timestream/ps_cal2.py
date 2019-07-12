"""Calibration using a strong point source.

Inheritance diagram
-------------------

.. inheritance-diagram:: PsCal
   :parts: 2

"""

import re
import itertools
import time
from datetime import date
import numpy as np
import numpy.ma as ma
from scipy import linalg as la
import ephem
import h5py
import aipy as a
import timestream_task
from tlpipe.container.timestream import Timestream
from tlpipe.core import constants as const

from caput import mpiutil
from caput import mpiarray
from tlpipe.utils.path_util import output_path
from tlpipe.utils import progress
from tlpipe.utils import rpca_decomp
import tlpipe.plot
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate
#
from datetime import datetime
from tlpipe.utils import date_util
import logging

logger = logging.getLogger(__name__)

class PsCal2(timestream_task.TimestreamTask):
    """Calibration using a strong point source.

    The observed visibility of a strong point source with flux :math:`S_c` is

    .. math:: V_{ij} &= g_i g_j^* A_i(\\hat{\\boldsymbol{n}}_0) A_j^*(\\hat{\\boldsymbol{n}}_0) S_c e^{2 \\pi i \\hat{\\boldsymbol{n}}_0 \\cdot (\\vec{\\boldsymbol{u}}_i - \\vec{\\boldsymbol{u}}_j)} \\\\
                     &= S_c \\cdot g_i A_i(\\hat{\\boldsymbol{n}}_0) e^{2 \\pi i \\hat{\\boldsymbol{n}}_0 \\cdot \\vec{\\boldsymbol{u}}_i} \\cdot (g_j A_j(\\hat{\\boldsymbol{n}}_0) e^{2 \\pi i \\hat{\\boldsymbol{n}}_0 \\cdot \\vec{\\boldsymbol{u}}_j})^* \\\\
                     &= S_c \\cdot G_i G_j^*,

    where :math:`G_i = g_i A_i(\\hat{\\boldsymbol{n}}_0) e^{2 \\pi i \\hat{\\boldsymbol{n}}_0 \\cdot \\vec{\\boldsymbol{u}}_i}`.

    In the presence of outliers and noise, we have

    .. math:: \\frac{V_{ij}}{S_c} = G_i G_j^* + S_{ij} + n_{ij}.

    Written in matrix form, it is

    .. math:: \\frac{\\boldsymbol{\\mathbf{V}}}{S_c} = \\boldsymbol{\\mathbf{V}}_0 + \\boldsymbol{\\mathbf{S}} + \\boldsymbol{\\mathbf{N}},

    where :math:`\\boldsymbol{\\mathbf{V}}_0 = \\boldsymbol{\\mathbf{G}} \\boldsymbol{\\mathbf{G}}^H`
    is a rank 1 matrix represents the visibilities of the strong point source,
    :math:`\\boldsymbol{\\mathbf{S}}` is a sparse matrix whose elements are outliers
    or misssing values, and :math:`\\boldsymbol{\\mathbf{N}}` is a matrix with dense
    small elements represents the noise.

    By solve the optimization problem

    .. math:: \\min_{V_0, S} \\frac{1}{2} \| V_0 + S - V \|_F^2 + \\lambda \| S \|_0

    we can get :math:`V_0`, :math:`S` and :math:`N` and solve the gain.

    """


    params_init = {
                    'calibrator': 'cyg',
                    'catalog': 'misc', # or helm,nvss
#                    'vis_conj': False, # if True, conjugate the vis first
                    'span': 60, # second
                    'plot_Vmat': False,
                    'plot_gain_vs_time': False,
                    'plot_gain_vs_freq': False,
                    'fig_name': 'gain/gain',
                    'save_src_vis': False, # save the extracted calibrator visibility
                    'src_vis_file': 'src_vis/src_vis.hdf5',
                    'subtract_src': False, # subtract vis of the calibrator from data
                    'apply_gain': True,
                    'save_gain': False,
                    'gain_file': 'gain/gain.hdf5',
                    'temperature_convert': False,
                    'ntmin': 1,
                    'rmsfact': 2.0,
                  }

    prefix = 'pc_'

    def process(self, ts):

        assert isinstance(ts, Timestream), '%s only works for Timestream object' % self.__class__.__name__

        calibrator = self.params['calibrator']
        catalog = self.params['catalog']
        span = self.params['span']
        plot_Vmat = self.params['plot_Vmat']
        plot_gain_vs_time = self.params['plot_gain_vs_time']
        plot_gain_vs_freq = self.params['plot_gain_vs_freq']
        fig_prefix = self.params['fig_name']
        tag_output_iter = self.params['tag_output_iter']
        save_src_vis = self.params['save_src_vis']
        src_vis_file = self.params['src_vis_file']
        subtract_src = self.params['subtract_src']
        apply_gain = self.params['apply_gain']
        save_gain = self.params['save_gain']
        gain_file = self.params['gain_file']
        temperature_convert = self.params['temperature_convert']
        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']
        ntmin = self.params['ntmin']
        rmsfact = self.params['rmsfact']
        MASKNOCAL = 4


#       Abort if polarization type is not linear
        pol_type = ts['pol'].attrs['pol_type']
        if pol_type != 'linear':
            raise RuntimeError('Can not do ps_cal for pol_type: %s' % pol_type)

        ts.redistribute('baseline')

                

        feedno = ts['feedno'][:].tolist()
        pol = [ ts.pol_dict[p] for p in ts['pol'][:] ] # as string
#        print("index=",pol.index('xx'),pol.index('yy'),pol.index('xy'),pol.index('yx'))

        gain_pd = {'xx': 0, 'yy': 1,    0: 'xx', 1: 'yy'} # for gain related op
        bl = mpiutil.gather_array(ts.local_bl[:], root=None, comm=ts.comm)
        bls = [ tuple(b) for b in bl ]
        # # antpointing = np.radians(ts['antpointing'][-1, :, :]) # radians
        # transitsource = ts['transitsource'][:]
        # transit_time = transitsource[-1, 0] # second, sec1970
        # int_time = ts.attrs['inttime'] # second

        #Get catalog info for calibrator
        srclist, cutoff, catalogs = a.scripting.parse_srcs(calibrator, catalog)
        cat = a.src.get_catalog(srclist, cutoff, catalogs)
        assert(len(cat) == 1), 'Allow only one calibrator'
        s = cat.values()[0]
        if mpiutil.rank0:
            print 'Calibrating for source %s with' % calibrator,
            print 'strength', s._jys, 'Jy',
            print 'measured at', s.mfreq, 'GHz',
            print 'with index', s.index

            # get transit time of calibrator
            # array
        aa = ts.array
        aa.set_jultime(ts['jul_date'][0]) # the first obs time point

        tsrc = ts['transitsource'][:]
        next_transit = aa.next_transit(s)
        print("next_transit=",next_transit)
            #conversion factor radians->degrees
        pir = 180.0/np.pi
            #get ra,dec of source in degrees.
        ra = pir*np.float64(repr(s.ra))
        dec = pir*np.float64(repr(s.dec))
            #number of sources in transitsource table
        nsource = tsrc.shape[0]
            #initialize transit time = not found in table
        transit_time = -1.0
        for ns in range(nsource):
            print(tsrc[ns,0],tsrc[ns,1],tsrc[ns,2],tsrc[ns,3],tsrc[ns,4])
            if abs(tsrc[ns,1]-ra)>1.0:
                continue
            if abs(tsrc[ns,2]-dec)>1.0:
                continue
           # print("i=",i)
           # print("ts['transitsource'][1,1]=",tsrc[i,1],"  180*s.ra/pi=",ra)
           # print("ts['transitsource'][1,2]=",tsrc[i,2],"  180*s.dec/pi=",dec)
            transit = datetime.utcfromtimestamp(tsrc[ns,0])
            transit_time = date_util.get_juldate(transit,tzone='UTC+00h')
            message = "Found calibrator in at ra=%.1f dec=%.1f transit source table line=%i" \
                % (ra,dec,ns)
            logger.debug(message)
            message = "Transit time=%.6f (Julian date)" % transit_time
            logger.debug(message)
            next_transit = a.phs.juldate2ephem(transit_time)
            break
        
        #If transit_time is not found in transitsource array, use ephem meridian
        #transit time
        if transit_time<0.0:
            transit_time = a.phs.ephem2juldate(next_transit) # Julian date
        # get time zone
        pattern = '[-+]?\d+'
        tz = re.search(pattern, ts.attrs['timezone']).group()
        tz = int(tz)
        # ...plus 8h to get Beijing time
        local_next_transit = ephem.Date(next_transit + tz * ephem.hour) 
        if transit_time > max(ts['jul_date'][-1], ts['jul_date'][:].max()):
            raise RuntimeError('Data does not contain local transit time %s of source %s' \
                                   % (local_next_transit, calibrator))

        # the first transit index
        transit_inds = [ np.searchsorted(ts['jul_date'][:], transit_time) ]
        # find all other transit indices
        aa.set_jultime(ts['jul_date'][0] + 1.0)
        transit_time = a.phs.ephem2juldate(aa.next_transit(s)) # Julian date
        cnt = 2
        while(transit_time <= ts['jul_date'][-1]):
            transit_inds.append(np.searchsorted(ts['jul_date'][:], transit_time))
            aa.set_jultime(ts['jul_date'][0] + 1.0*cnt)
            transit_time = a.phs.ephem2juldate(aa.next_transit(s)) # Julian date
            cnt += 1

#  CHANGE TO DEBUG MESSAGE
#        if mpiutil.rank0:
#            print 'transit ind of %s: %s, time: %s' % (calibrator, transit_inds, local_next_transit)

        # For now, only use the first transit point to do the cal
        transit_ind = transit_inds[0]
        int_time = ts.attrs['inttime'] # second
        start_ind = transit_ind - np.int(span / int_time)
        # plus 1 to make transit_ind at the center
        end_ind = transit_ind + np.int(span / int_time) + 1 

        # check if data contain this range
        if start_ind < 0:
            raise RuntimeError('start_ind: %d < 0' % start_ind)
        if end_ind > ts.vis.shape[0]:
            raise RuntimeError('end_ind: %d > %d' % (end_ind, ts.vis.shape[0]))

        t_inds = range(start_ind, end_ind)
        ntbin = end_ind - start_ind
        freq = ts.freq[:]
        nfreq = len(freq)
        nbl = len(ts.local_bl[:])
        nlb =  nbl
        nfeed = len(feedno)
        #List of time bins x frequency bins x 2 like (xx,yy) polarizations
        tfp_inds = list(itertools.product(t_inds, range(nfreq), [pol.index('xx'), pol.index('yy')]))
        ns, ss, es = mpiutil.split_all(len(tfp_inds), comm=ts.comm)
        # gather data to make each process to have its own data which has all bls
        for ri, (ni, si, ei) in enumerate(zip(ns, ss, es)):
            lvis = np.zeros((ni, nbl), dtype=ts.vis.dtype)
            lvis_mask = np.zeros((ni, nbl), dtype=ts.vis_mask.dtype)
            for ii, (ti, fi, pi) in enumerate(tfp_inds[si:ei]):
                lvis[ii] = ts.local_vis[ti, fi, pi]
                lvis_mask[ii] = ts.local_vis_mask[ti, fi, pi]
            # gather vis from all process for separate bls
            gvis = mpiutil.gather_array(lvis, axis=1, root=ri, comm=ts.comm)
            gvis_mask = mpiutil.gather_array(lvis_mask, axis=1, root=ri, comm=ts.comm)
            if ri == mpiutil.rank:
                tfp_linds = tfp_inds[si:ei] # inds for this process
                this_vis = gvis
                this_vis_mask = gvis_mask
        del tfp_inds
        del lvis
        del lvis_mask
        tfp_len = len(tfp_linds)

        #Create additional visibility arrays, if needed
        if save_src_vis or subtract_src:
            # save calibrator src vis
            lsrc_vis = np.full((tfp_len, nfeed, nfeed), complex(np.nan,np.nan), dtype=ts.vis.dtype)
            if save_src_vis:
                # save sky vis
             #   print("****")
                lsky_vis = np.full((tfp_len, nfeed, nfeed), \
                           complex(np.nan,np.nan), dtype=ts.vis.dtype)
             #   print("lsky_vis=",lsky_vis.dtype,lsky_vis.shape)
                # save outlier vis
             #   print("ts.vis.dtype=",ts.vis.dtype)
                lotl_vis = np.full((tfp_len, nfeed, nfeed), \
                           complex(np.nan,np.nan), dtype=ts.vis.dtype)
             #   print("lotl_vis=",lotl_vis)
        lGain = np.full((tfp_len, nfeed), complex(np.nan,np.nan), \
                            dtype=ts.vis.dtype)
        lGerr = np.full(tfp_len,True,dtype=np.bool)        
        # construct visibility matrix for a single time, freq, pol
        Vmat = np.zeros((nfeed, nfeed), dtype=ts.vis.dtype)
        # Flattened array for absolute values of off-diagonal elements of Vmat
        ntri = nfeed*(nfeed-1)/2
        tri = np.zeros(ntri,dtype=np.float64)

        #Get source intensity
        Sc = s.get_jys()

        #Default units are Jy
        calunit = 'J'            
        #See if conversion to temperature is desired
        if temperature_convert:
            if ts.is_dish:
                d = ts.attrs['dishdiam']
                    #Assume effective dish area = 50% of aperture
                effArea = 0.5*(np.pi/4.0)*d*d
            else:
                effArea = 0.5*ts.attrs['cylen']*  \
                    ts.attrs['cywid']/ts.attrs['nfeeds']
            factor = 1.0e-26*effArea / (2 * const.k_B)
            Sc *= factor
            calunit = 'K'

        print("Sc=",Sc)
        if show_progress and mpiutil.rank0:
            pg = progress.Progress(tfp_len, step=progress_step)
        for ii, (ti, fi, pi) in enumerate(tfp_linds):
           # print("ti,fi,pi",ti,fi,pi)
            if show_progress and mpiutil.rank0:
                pg.show(ii)
            # when noise on, just pass
            if 'ns_on' in ts.iterkeys() and ts['ns_on'][ti]:
                continue
                # aa.set_jultime(ts['jul_date'][ti])
                # s.compute(aa)
                # get fluxes vs. freq of the calibrator
                # Sc = s.get_jys()
                # get the topocentric coordinate of the calibrator at the current time
                # s_top = s.get_crds('top', ncrd=3)
                # aa.sim_cache(cat.get_crds('eq', ncrd=3)) # for compute bm_response and sim
            mask_cnt = 0
            for i, ai in enumerate(feedno):
                for j, aj in enumerate(feedno):
                    try:
                        bi = bls.index((ai, aj))
#and->or
                        if this_vis_mask[ii, bi]!=0 or not np.isfinite(this_vis[ii, bi]):
                            mask_cnt += 1
                            Vmat[i, j] = 0
                        else:
                            Vmat[i, j] = this_vis[ii, bi] # xx, yy
                    except ValueError:
                        bi = bls.index((aj, ai))
                        if this_vis_mask[ii, bi]!=0 or not np.isfinite(this_vis[ii, bi]):
                            mask_cnt += 1
                            Vmat[i, j] = 0
                        else:
                            Vmat[i, j] = np.conj(this_vis[ii, bi]) # xx, yy
                        
            if save_src_vis:
                lsky_vis[ii] = Vmat

            # if too many masks
#            if mask_cnt > 0.3 * nfeed**2:
#                continue
            #For now, we don't allow any masked data.  
            #It is possible to relax this requirement
            if mask_cnt > 0:
                print("Skipping t,f,p",ti,fi,pi," Masked=",mask_cnt)
                continue

           #
           # get triangular matrix absolute values
           #
            ns = 0
            for n in range(nfeed-1):
                ne = ns + nfeed - n - 1
                tri[ns:ne] = np.abs(Vmat[n+1:nfeed,n])
                ns = ne
            #Statistics on off-diagonal elements
            Vmean = np.mean(tri)
            Vmin = np.min(tri)
            Vmax = np.max(tri)
            #All visibilities should be non-zero
            if Vmin==0.0:
                print("Skipping t,f,p",ti,fi,pi," Vmin=0")
                continue
            #All off-diagonal visibilities should have approximately the 
            #same magnitude assuming:
            #1. A single point source is dominant
            #2  The uncalibrated gains are approximately equal
            #We expect deviations from assumption 2 at the band edges,
            #where the filter edges lead to large differences in gain
            fpos = np.float64(fi)/np.float64(nfreq) - 0.5
            Vratio_max = 25.0 + 100.0*fpos**2
            if Vmax/Vmin>Vratio_max:
                print("Skipping t,f,p",ti,fi,pi,"Vratio=",Vmax/Vmin, \
                          ">",Vratio_max)
                continue
            if Vmean>10.0:
                print("Vmean=",ti,fi,pi,Vmean," max=", \
                          np.max(tri)," min=",np.min(tri))

            # set invalid val to 0
            #Redundant?
            #Vmat = np.where(np.isfinite(Vmat), Vmat, 0)

            S0 = np.zeros((nfeed,nfeed),dtype=ts.vis.dtype)
            for n in range(nfeed):
                S0[n,n] = Vmat[n,n] - Vmean
        

            # initialize the outliers
            #med = np.median(Vmat.real) + 1.0J * np.median(Vmat.imag)
            #diff = Vmat - med
            #S0 = np.where(np.abs(diff)>3.0*rpca_decomp.MAD(Vmat), diff, 0)
            # stable PCA decomposition
            V0, S, error = rpca_decomp.decompose(Vmat, rank=1, S=S0, max_iter=100, \
                           threshold='hard', tol=1.0e-6, debug=False)
            if error==0:
                lGerr[ii] = False
            else:
                print("rpca error index=",ii," t,f,p=",ti,fi,pi,lGerr[ii])
 #               print("tri=",tri)
                # V0, S = rpca_decomp.decompose(Vmat, rank=1, S=S0, max_iter=100, threshold='soft', tol=1.0e-6, debug=False)
                continue

            if save_src_vis or subtract_src:
                lsrc_vis[ii] = V0
                if save_src_vis:
                    lotl_vis[ii] = S

            # plot
            if plot_Vmat:
                ind = ti - start_ind
                # plot Vmat
                plt.figure(figsize=(13, 5))
                plt.subplot(121)
                plt.imshow(Vmat.real, aspect='equal', origin='lower', interpolation='nearest')
                plt.colorbar(shrink=1.0)
                plt.subplot(122)
                plt.imshow(Vmat.imag, aspect='equal', origin='lower', interpolation='nearest')
                plt.colorbar(shrink=1.0)
                fig_name = '%s_V_%d_%d_%s.png' % (fig_prefix, ind, fi, pol[pi])
                if tag_output_iter:
                    fig_name = output_path(fig_name, iteration=self.iteration)
                else:
                    fig_name = output_path(fig_name)
                plt.savefig(fig_name)
                plt.close()
                # plot V0
                plt.figure(figsize=(13, 5))
                plt.subplot(121)
                plt.imshow(V0.real, aspect='equal', origin='lower', interpolation='nearest')
                plt.colorbar(shrink=1.0)
                plt.subplot(122)
                plt.imshow(V0.imag, aspect='equal', origin='lower', interpolation='nearest')
                plt.colorbar(shrink=1.0)
                fig_name = '%s_V0_%d_%d_%s.png' % (fig_prefix, ind, fi, pol[pi])
                if tag_output_iter:
                    fig_name = output_path(fig_name, iteration=self.iteration)
                else:
                    fig_name = output_path(fig_name)
                plt.savefig(fig_name)
                plt.close()
                # plot S
                plt.figure(figsize=(13, 5))
                plt.subplot(121)
                plt.imshow(S.real, aspect='equal', origin='lower', interpolation='nearest')
                plt.colorbar(shrink=1.0)
                plt.subplot(122)
                plt.imshow(S.imag, aspect='equal', origin='lower', interpolation='nearest')
                plt.colorbar(shrink=1.0)
                fig_name = '%s_S_%d_%d_%s.png' % (fig_prefix, ind, fi, pol[pi])
                if tag_output_iter:
                    fig_name = output_path(fig_name, iteration=self.iteration)
                else:
                    fig_name = output_path(fig_name)
                plt.savefig(fig_name)
                plt.close()
                # plot N
                N = Vmat - V0 - S
                plt.figure(figsize=(13, 5))
                plt.subplot(121)
                plt.imshow(N.real, aspect='equal', origin='lower', interpolation='nearest')
                plt.colorbar(shrink=1.0)
                plt.subplot(122)
                plt.imshow(N.imag, aspect='equal', origin='lower', interpolation='nearest')
                plt.colorbar(shrink=1.0)
                fig_name = '%s_N_%d_%d_%s.png' % (fig_prefix, ind, fi, pol[pi])
                if tag_output_iter:
                    fig_name = output_path(fig_name, iteration=self.iteration)
                else:
                    fig_name = output_path(fig_name)
                plt.savefig(fig_name)
                plt.close()

            e, U = la.eigh(V0 / Sc[fi], eigvals=(nfeed-1, nfeed-1))
            g = U[:, -1] * e[-1]**0.5
            #Choose g[0] sign to be positive
            if g.real[0]<0.0:
                g[:] = -g[:]
            lGain[ii] = g

            # plot Gain versus feed
            if plot_Vmat:
                plt.figure()
                plt.plot(feedno, g.real, 'b-', label='real')
                plt.plot(feedno, g.real, 'bo')
                plt.plot(feedno, g.imag, 'g-', label='imag')
                plt.plot(feedno, g.imag, 'go')
                plt.plot(feedno, np.abs(g), 'r-', label='abs')
                plt.plot(feedno, np.abs(g), 'ro')
                plt.xlim(feedno[0]-1, feedno[-1]+1)
                yl, yh = plt.ylim()
                plt.ylim(yl, yh+(yh-yl)/5)
                plt.xlabel('Feed number')
                plt.legend()
                fig_name = '%s_ants_%d_%d_%s.png' % (fig_prefix, ind, fi, pol[pi])
                if tag_output_iter:
                    fig_name = output_path(fig_name, iteration=self.iteration)
                else:
                    fig_name = output_path(fig_name)
                plt.savefig(fig_name)
                plt.close()


        # subtract the vis of calibrator from self.vis
        if subtract_src:
            nbl = len(bls)
            lv = np.zeros((lsrc_vis.shape[0], nbl), dtype=lsrc_vis.dtype)
            for bi, (fd1, fd2) in enumerate(bls):
                b1, b2 = feedno.index(fd1), feedno.index(fd2)
                lv[:, bi] = lsrc_vis[:, b1, b2]
            lv = mpiarray.MPIArray.wrap(lv, axis=0, comm=ts.comm)
            lv = lv.redistribute(axis=1).local_array.reshape(ntbin, nfreq, 2, -1)
            ts.local_vis[start_ind:end_ind, :, pol.index('xx')] -= lv[:, :, 0]
            ts.local_vis[start_ind:end_ind, :, pol.index('yy')] -= lv[:, :, 1]

            del lv


        if not save_src_vis:
            if subtract_src:
                del lsrc_vis
        else:
            if tag_output_iter:
                src_vis_file = output_path(src_vis_file, iteration=self.iteration)
            else:
                src_vis_file = output_path(src_vis_file)
            # create file and allocate space first by rank0
            if mpiutil.rank0:
                with h5py.File(src_vis_file, 'w') as f:
                        # allocate space
                    shp = (ntbin, nfreq, 2, nfeed, nfeed)
                    f.create_dataset('sky_vis', shp, dtype=lsky_vis.dtype)
                    f.create_dataset('src_vis', shp, dtype=lsrc_vis.dtype)
                    f.create_dataset('outlier_vis', shp, dtype=lotl_vis.dtype)
                    f.attrs['calibrator'] = calibrator
                    f.attrs['dim'] = 'time, freq, pol, feed, feed'
                    f.attrs['time'] = ts.time[start_ind:end_ind]
                    f.attrs['freq'] = freq
                    f.attrs['pol'] = np.array(['xx', 'yy'])
                    f.attrs['feed'] = np.array(feedno)

            mpiutil.barrier()

            # write data to file
            for i in range(10):
                try:
                    # NOTE: if write simultaneously, will lose data with processes 
                    #distributed in several nodes
                    for ri in xrange(mpiutil.size):
                        if ri == mpiutil.rank:
                            with h5py.File(src_vis_file, 'r+') as f:
                                for ii, (ti, fi, pi) in enumerate(tfp_linds):
                                    ti_ = ti-start_ind
                                    pi_ = gain_pd[pol[pi]]
                                    f['sky_vis'][ti_, fi, pi_] = lsky_vis[ii]
                                    f['src_vis'][ti_, fi, pi_] = lsrc_vis[ii]
                                    f['outlier_vis'][ti_, fi, pi_] = lotl_vis[ii]
                        mpiutil.barrier()
                    break
                except IOError:
                    time.sleep(0.5)
                    continue
            else:
                raise RuntimeError('Could not open file: %s...' % src_vis_file)

            del lsrc_vis
            del lsky_vis
            del lotl_vis

            mpiutil.barrier()

        if apply_gain or save_gain:
            #c = nt/2 # center ind
            #li = max(0, c - 10)
            #hi = min(nt, c + 10 + 1)
            #print("c,li,hi=",c,li,hi)
            # compute s_top for this time range
            n0 = np.zeros((ntbin,3),dtype=np.float64)
 #           print("n0",n0.dtype)
 #           print("start,end",start_ind,end_ind)
 #           print("ts.time",ts.time[start_ind:end_ind])
            for ti, jt in enumerate(ts.time[start_ind:end_ind]):
                aa.set_jultime(jt)
                s.compute(aa)
                n0[ti] = s.get_crds('top', ncrd=3)
 #               print("ti,jt",ti,jt)
 #               print("XX",ts.time)
 #               print("n0",n0[ti])
 #           print ("n0",n0.dtype)
            
            # get the positions of feeds
            feedpos = ts['feedpos'][:]

            # wrap and redistribute Gain
#            print("lGain=",lGain)
#            print("lGain.shape=",lGain.shape)
#            Gain = mpiarray.MPIArray.wrap(lGain, axis=0, comm=ts.comm)
#            print("Gain.shape=",Gain.shape)
#            Gain = Gain.redistribute(axis=1).reshape(ntbin, nfreq, 2, None).redistribute(axis=0).reshape(None, nfreq*2*nfeed).redistribute(axis=1)
#            Gerr = mpiarray.MPIArray.wrap(lGerr, axis=0, comm=ts.comm)
#            print("Gerr.shape=",Gerr.shape)
#            Gerr = Gerr.reshape(None, nfreq, 2)
#            print("lGerr.shape=",lGerr.shape)
            rGain = lGain.reshape((ntbin,nfreq*2,nfeed))
#            print("lGain.shape=",lGain.shape)
            rGerr = lGerr.reshape((ntbin,nfreq*2))
#            print("lGerr.shape=",lGerr.shape)
            Gain = mpiarray.MPIArray.wrap(rGain, axis=1, comm=ts.comm)
            Gerr = mpiarray.MPIArray.wrap(rGerr, axis=1, comm=ts.comm)
            print("Gain.shape=",Gain.shape)
            print("Gain.global_shape=",Gain.global_shape)
            print("Gain.axis=",Gain.axis)
#            print("Gain=",Gain)
            

            fp_inds = list(itertools.product(range(nfreq), range(2))) 
            fp_linds = mpiutil.mpilist(fp_inds, method='con', comm=ts.comm)
            del fp_inds
            # create data to save the solved gain for each feed
            lgain = np.full((len(fp_linds),nfeed), complex(np.nan,np.nan),dtype=Gain.dtype) 
            
            # solve for gain
            for ii, (fi, pi) in enumerate(fp_linds):
                # position of this feed (relative to the first feed) in unit of wavelength
                for di in range(nfeed):
                    ui = (feedpos[di] - feedpos[0]) * (1.0e6*freq[fi]) / const.c
                    exp_factor = np.exp(2.0J * np.pi * np.dot(n0, ui))
#                print("Gi=",Gain.local_array[li:hi,ii])
#                print("Gerr=",Gerr.local_array[li:hi,ii])
                  #  print("exp_factor=",exp_factor)
                    Gi = ma.array(Gain.local_array[:, ii,di]/exp_factor,\
                                  mask=Gerr.local_array[:,ii])
#                    print("Gi.shape=",Gi.shape)
#                print("Gerr=",fi,pi,Gerr[:,fi,pi])
                    nval = ma.count(Gi)
#                print("nval",ii,nval)
                    if nval<ntmin:
                        continue
#                print("Gi.shape",Gi.shape)
#                print("Gi=",Gi)
#Check at least n positions
#                ntpt = len(Gi)
                    ave = ma.mean(Gi)
                    rms = ma.std(Gi,ddof=1)
                    tol = rmsfact*rms
                    dG = ma.abs(Gi-ave)
#                    print("dG=",dG)
#                    print("dG.data=",dG.data)
#                    print("tol=",tol,rmsfact,rms)
                    dG = ma.array(np.where(np.isfinite(dG.data),dG.data,2*tol),\
                                      mask=dG.mask)
#                    print("xxx=",xxx)
#                    dG.data = xxx
# IF dG is not finite and mask=False
                    nG = len(Gi)
                    gBad = False
                    for nn in range(nG):
                        if not np.isfinite(dG[nn]) and not dG[nn].mask:
                            gBad = True
                    if gBad:
                        print("Gi",Gi)
                        print("Gi.mask",Gi.mask)
                        print("dG",dG)
                        print("dG.mask",dG.mask)
                        print("ave,rms",ave,rms)
#
#                print("dG=",dG)
#Want factor>1
#Gi.count() will give number of valid entries
                    print("fi=",fi," pi=",pi,"di=",di)
                    print("ave=",ave,"rms=",rms,"tol=",tol)
                    print("dG=",dG)
#                Gi = ma.masked_where(dG.data>tol,Gi)
                    nmask1 = ma.count_masked(Gi)
#                if nmask1>0:
#                    print("****",ii,nmask1)
#
                    print("Gi1=",Gi)
                    Gi = ma.masked_where(dG.data>tol,Gi)
                    print("Gi2=",Gi)
                    nmask2 = ma.count_masked(Gi)
                    if nmask2>nmask1:
#                    print("nmask=",nmask1,nmask2)
                        ave = ma.mean(Gi)
                        rms = ma.std(Gi,ddof=1)
                            #print("ave2=",ave," rms=",rms)
                    nval = ma.count(Gi)
                    
#Gain should always be non-zero
                    if abs(ave)==0.0:
                        print("Gain=0 f,p,d=",fi,pi,di)
                        continue
                    if nval>0:
                        lgain[ii] = ave
                        sigma = rms/np.sqrt(nval)
#
#        Make plot gain magnitude & phase
#
                    if plot_gain_vs_time:
                            #print("Gi.mask=",Gi.mask)
                            #Gi.mask[3] = True
                        amp = ma.masked_invalid(np.abs(Gi.data))
                        nval = ma.count(amp)
                        if nval==0:
                            continue
                        hiamp = np.max(amp)
                        loamp = np.min(amp)
                        spamp = hiamp-loamp
                        if spamp<0.1*hiamp:
                            spamp = 0.1*hiamp
                        hiamp = hiamp + spamp/2.0
                        loamp = max(loamp/2.0,loamp-spamp/2.0)
                        amperr = np.zeros_like(amp)
                        amperr[:] = sigma
                        phase = (180.0/np.pi)*np.angle(Gi.data)
                        hiphs = np.max(phase)
                        lophs = np.min(phase)
                        spphs = hiphs-lophs
                        if spphs<2.0:
                            spphs=2.0
                        hiphs = min(200.0,hiphs+spphs/2.0)
                        lophs = max(-200.0,lophs-spphs/2.0)
                        phserr = np.zeros_like(phase)
                        phserr[:] = (180.0/np.pi)*amperr[:]/amp[:]
                        tbin = np.arange(ntbin)
                        plt.figure()
                        ax1 = plt.axes()
                        ax1.set_xlabel("Time Bin")
                        ax1.set_ylabel("Amplitude")
                        ax1.set_ylim(loamp,hiamp)
                        ax1.errorbar(tbin,amp,yerr=amperr,fmt='rs', \
                                         label='Amplitude',capsize=2.0)
                        ax1.axhline(y=np.abs(ave),color='red')
                        ax1.legend(loc=2)
                        ax2 = ax1.twinx()
                        ax2.set_ylabel("Phase")
                        ax2.set_ylim(lophs,hiphs)
                        ix = np.nonzero(~Gi.mask)
                        ax2.errorbar(tbin[ix],phase[ix],yerr=phserr[ix], \
                                         fmt='bo',label='Phase',capsize=2.0)
                        ax2.legend(loc=1)
                        ix = np.nonzero(Gi.mask)
                        ax2.errorbar(tbin[ix],phase[ix],yerr=phserr[ix], \
                                         fmt='ko',label='Phase',capsize=2.0)


                            #plt.plot(tbin,phase)
                        fig_name = '%s_gt_%d_%d_%s.png' % \
                            (fig_prefix, fi, di, pol[pi])
                        if tag_output_iter:
                            fig_name = output_path(fig_name, iteration=self.iteration)
                        else:
                            fig_name = output_path(fig_name)
                        plt.savefig(fig_name)
                        plt.close()



            # gather local gain
            gain = mpiutil.gather_array(lgain, axis=0, root=None, comm=ts.comm)
            del lgain
            gain = gain.reshape(nfreq, 2, nfeed)
#            print("gain.shape",gain.shape)

            gcurve = np.zeros(nfreq,dtype=np.complex64)
            fimin = 0
            fimax = nfreq

            for di in range(nfeed):
                for pi in range(2):
#                    print("dish=",di,"pol=",pi)
#                    print(nfreq)
                    fc = np.arange(nfreq)
                    nfs = (nfreq % 32)/2
                    #print("nfs=",nfs)
                    #print("t=",t)
                    gcurve[:] = gain[:,pi,di]
                    amp = np.abs(gcurve,dtype=np.float64)
                    phase = np.array(np.angle(gcurve),dtype=np.float64)
#xx
                    igd = np.isfinite(amp)
                    #print("igd=",igd)
                    fc = fc[igd]
                    if fc[0]>fimin:
                        fimin = fc[0]
                    if fc[-1]<fimax:
                        fimax = fc[-1]
                    amp = amp[igd]
                    phase = phase[igd]
                    #print("fc=",fc)
                    nspt = len(fc)/32
                    #print("nspt=",nspt)
                    spt = np.floor_divide(len(fc),nspt)
                    #print("spt=",spt)
                    t = np.zeros(nspt-1,dtype=np.int)
                    print("t1=",len(t),t.dtype,t)
                    for n in range(nspt-1):
                        #print("n=",n)
                        index = int((n+1)*spt)
                        #print("index=",index)
                        t[n] = fc[index]
                        #print("t[n]=",t[n])
                    k = 3
                    t = np.r_[(fc[0],)*(k+1),t,(fc[-1],)*(k+1)]
                    print("t2=",len(t),t.dtype,t)
                
#Need to reorganize below
                    for ntry in range(3):
                        igd = np.isfinite(amp)
                        fc=fc[igd]
                        amp=amp[igd]
                        phase=phase[igd]
                        npt = len(amp)
                        #print("npt=",npt)
                        #print("len(amp)",len(amp))
                        #print("len(igd)",len(igd))
                        #print("len(fc]",len(fc))
                        #print("igd=",igd)
                        #print("amp=",amp)
        
                        asp = interpolate.make_lsq_spline(fc,amp,t,k)
                        afit = interpolate.BSpline(asp.t,asp.c,k)
                        if ntry>=2:
                            break
                        diff = amp - afit(fc)
#    print("diff=",diff)
#    ind = np.zeros(npt,np.int)
                        ind = np.argsort(np.abs(diff),kind='qucksort')
                    #    for n in range(npt-10,npt):
                    #        i = ind[n]
                    #        print("sorted",n,i,diff[i])
                        nlo = np.int(np.floor(0.9*npt))
                     #   print(nlo)
#
                        maxdev = 10.0*np.abs(diff[nlo])
                        if maxdev<np.abs(diff[-1])/10.0:
                            maxdev = np.abs(diff[-1]/10.0)
                            #maxdev = 1000000.0

                     #   print("ntry,maxdev=",ntry,maxdev)
                        for n in range(npt-1,nlo,-1):
                            ix = ind[n]
                            if np.abs(diff[ix])>maxdev:
                                amp[ix] = np.nan
                                print("iter=",ntry,"Drop point=",ix)
                            else:
                                break

                    fbin = np.arange(nfreq)
                    ampfit = afit(fbin)
                    phase = np.unwrap(phase)
                    psp = interpolate.make_lsq_spline(fc,phase,t,k)
                    pfit = interpolate.BSpline(psp.t,psp.c,k)
                    phase = pfit(fbin) % (2.0*np.pi)
                    phsfit = (180./np.pi) * \
                        np.where(phase>np.pi,phase-2.0*np.pi,phase)
#
                    if plot_gain_vs_freq:
                        gfplot = np.zeros(nfreq,dtype=np.complex64)
                        gfplot[:] = gain[:,pi,di]
                        plt.figure()
                        ax1 = plt.axes()
                        ax1.set_xlabel("Frequency Bin")
                        ax1.set_ylabel("Amplitude")
                        amp = ma.masked_invalid(np.abs(gfplot))
                        #print("\nGain dish=",di," pol=",pi)
                            #for ii in range(nf):
                            #    print("ii=",ii,"amp=",amp[ii])
                        hiamp = 1.5*ma.max(amp)
                #        print("amp,hiamp=",amp,hiamp)
                        ax1.set_ylim(0.0,hiamp)
                        ax1.plot(fbin,amp,'rs',ms=5.0,label='Amplitude')
                        ax1.plot(fbin,ampfit,'k-',lw=3.0,label='Amp Fit')
                        ax1.legend(loc=2)
                        ax2 = ax1.twinx()
                        ax2.set_ylabel("Phase")
                        ax2.set_ylim(-200.0,250.0)
                        phase = ma.masked_invalid((180.0/np.pi)*np.angle(gfplot))
                        ax2.plot(fbin,phase,'bo',ms=5.0,label='Phase')
                        ax2.plot(fbin,phsfit,'k-',lw=3.0,label='Phase Fit')
                        ax2.legend(loc=1)
                        fig_name = '%s_gf_%d_%s.png' % (fig_prefix, di, pol[pi])
                        if tag_output_iter:
                            fig_name = output_path(fig_name, iteration=self.iteration)
                        else:
                            fig_name = output_path(fig_name)
                            #print("fig_name",fig_name)
                        plt.savefig(fig_name)
                        plt.close()
                    #Store complex gain as results of fit
                    phsfit = (np.pi/180.0)*phsfit
                    for nx in range(nfreq):
#                        print("n,phase",nx,phsfit[nx])
                        gain[:,pi,di] = ampfit*(np.cos(phsfit)+1.0j*np.sin(phsfit))
#                    for nx in range(nfreq):
#                        print("n,gain",nx,gain[nx,pi,di])

            # apply gain to vis
            if apply_gain:
                if temperature_convert:
                    ts.vis.attrs['unit'] = 'K'

                for fi in range(nfreq):
                    for pi in [pol.index('xx'), pol.index('yy')]:
                        pi_ = gain_pd[pol[pi]]
                        for bi, (fd1, fd2) in enumerate(ts['blorder'].local_data):
                            g1 = gain[fi, pi_, feedno.index(fd1)]
                            g2 = gain[fi, pi_, feedno.index(fd2)]
                            if np.isfinite(g1) and np.isfinite(g2):
                                ts.local_vis[:, fi, pi, bi] /= (g1 * np.conj(g2))
                            else:
                                    # mask the un-calibrated vis
                                ts.local_vis_mask[:, fi, pi, bi] |= MASKNOCAL

            # save gain to file
 #           print("save_gain=",save_gain)
            if save_gain:
                if tag_output_iter:
                    gain_file = output_path(gain_file, iteration=self.iteration)
                else:
                    gain_file = output_path(gain_file)
  #              print("gain_file=",gain_file)
  #              print("mpiutil.rank0=",mpiutil.rank0)
                if mpiutil.rank0:
   #                 print("Before h5py file")
                    with h5py.File(gain_file, 'w') as f:
                    #    print("After h5py file")
                        f.attrs['cal_algorith'] = 'ps_cal2'
                        f.attrs['history'] = ts.attrs['history']
                        f.attrs['calibrator'] = calibrator
                        f.attrs['cal_intens'] = s._jys
                        f.attrs['cal_freq'] = s.mfreq
                        f.attrs['cal_index'] = s.index
                        f.attrs['cal_time'] = next_transit
                        f.attrs['cal_unit'] = calunit
                        caldate = date.today()
                        f.attrs['cal_date'] = caldate.isoformat()

#Data used to make calibration?

                        # allocate space for Gain
                        dset = f.create_dataset('Gain', \
                               (ntbin, nfreq, 2, nfeed), dtype=Gain.dtype)
                        dset.attrs['calibrator'] = calibrator
                        dset.attrs['dim'] = 'time, freq, pol, feed'
                        dset.attrs['time'] = ts.time[start_ind:end_ind]
                        dset.attrs['freq'] = freq
                        dset.attrs['fimin'] = fimin
                        dset.attrs['fimax'] = fimax
                        dset.attrs['pol'] = np.array(['xx', 'yy'])
                        dset.attrs['feed'] = np.array(feedno)
                            # save gain
                        dset = f.create_dataset('gain', data=gain)
                        dset.attrs['calibrator'] = calibrator
                        dset.attrs['dim'] = 'freq, pol, feed'
                        dset.attrs['freq'] = freq
                        dset.attrs['pol'] = np.array(['xx', 'yy'])
                        dset.attrs['feed'] = np.array(feedno)
                mpiutil.barrier()
                # save Gain
                for i in range(10):
                    try:
                            # NOTE: if write simultaneously, will loss data with processes distributed in several nodes
                        for ri in xrange(mpiutil.size):
                            if ri == mpiutil.rank:
                                with h5py.File(gain_file, 'r+') as f:
                                    for ii, (ti, fi, pi) in enumerate(tfp_linds):
                                        ti_ = ti-start_ind
                                        pi_ = gain_pd[pol[pi]]
                                        f['Gain'][ti_, fi, pi_] = lGain[ii]
                            mpiutil.barrier()
                        break
                    except IOError:
                        time.sleep(0.5)
                        continue
                else:
                    raise RuntimeError('Could not open file: %s...' % gain_file)

                mpiutil.barrier()


        # convert vis from intensity unit to temperature unit in K
#        if temperature_convert:
#            if 'unit' in ts.vis.attrs.keys() and ts.vis.attrs['unit'] == 'K':
#                if mpiutil.rank0:
#                    print 'vis is already in unit K, do nothing...'
#            else:
#                factor = 1.0e-26 * (const.c**2 / (2 * const.k_B * (1.0e6*freq)**2)) # NOTE: 1Jy = 1.0e-26 W m^-2 Hz^-1
#                ts.local_vis[:] *= factor[np.newaxis, :, np.newaxis, np.newaxis]
#                ts.vis.attrs['unit'] = 'K'


        return super(PsCal2, self).process(ts)
