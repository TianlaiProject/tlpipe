"""Calibration using a strong point source.

Inheritance diagram
-------------------

.. inheritance-diagram:: PsCal
   :parts: 2

"""

import re
import itertools
import time
import numpy as np
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


class PsCal(timestream_task.TimestreamTask):
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
                    'vis_conj': False, # if True, conjugate the vis first
                    'zero_diag': False, # if True, fill 0 to the diagonal of vis matrix before SPCA
                    'span': 60, # second
                    'plot_figs': False,
                    'fig_name': 'gain/gain',
                    'save_src_vis': False, # save the extracted calibrator visibility
                    'src_vis_file': 'src_vis/src_vis.hdf5',
                    'subtract_src': False, # subtract vis of the calibrator from data
                    'apply_gain': True,
                    'save_gain': False,
                    'save_phs_change': False,
                    'gain_file': 'gain/gain.hdf5',
                    'temperature_convert': False,
                  }

    prefix = 'pc_'

    def process(self, ts):

        assert isinstance(ts, Timestream), '%s only works for Timestream object' % self.__class__.__name__

        calibrator = self.params['calibrator']
        catalog = self.params['catalog']
        vis_conj = self.params['vis_conj']
        zero_diag = self.params['zero_diag']
        span = self.params['span']
        plot_figs = self.params['plot_figs']
        fig_prefix = self.params['fig_name']
        tag_output_iter = self.params['tag_output_iter']
        save_src_vis = self.params['save_src_vis']
        src_vis_file = self.params['src_vis_file']
        subtract_src = self.params['subtract_src']
        apply_gain = self.params['apply_gain']
        save_gain = self.params['save_gain']
        save_phs_change = self.params['save_phs_change']
        gain_file = self.params['gain_file']
        temperature_convert = self.params['temperature_convert']
        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']

        if save_src_vis or subtract_src or apply_gain or save_gain:
            pol_type = ts['pol'].attrs['pol_type']
            if pol_type != 'linear':
                raise RuntimeError('Can not do ps_cal for pol_type: %s' % pol_type)

            ts.redistribute('baseline')

            feedno = ts['feedno'][:].tolist()
            pol = [ ts.pol_dict[p] for p in ts['pol'][:] ] # as string
            gain_pd = {'xx': 0, 'yy': 1,    0: 'xx', 1: 'yy'} # for gain related op
            bls = mpiutil.gather_array(ts.local_bl[:], root=None, comm=ts.comm)
            # # antpointing = np.radians(ts['antpointing'][-1, :, :]) # radians
            # transitsource = ts['transitsource'][:]
            # transit_time = transitsource[-1, 0] # second, sec1970
            # int_time = ts.attrs['inttime'] # second

            # calibrator
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
            next_transit = aa.next_transit(s)
            transit_time = a.phs.ephem2juldate(next_transit) # Julian date
            # get time zone
            pattern = '[-+]?\d+'
            tz = re.search(pattern, ts.attrs['timezone']).group()
            tz = int(tz)
            local_next_transit = ephem.Date(next_transit + tz * ephem.hour) # plus 8h to get Beijing time
            # if transit_time > ts['jul_date'][-1]:
            if transit_time > max(ts['jul_date'][-1], ts['jul_date'][:].max()):
                raise RuntimeError('Data does not contain local transit time %s of source %s' % (local_next_transit, calibrator))

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

            if mpiutil.rank0:
                print 'transit ind of %s: %s, time: %s' % (calibrator, transit_inds, local_next_transit)

            ### now only use the first transit point to do the cal
            ### may need to improve in the future
            transit_ind = transit_inds[0]
            int_time = ts.attrs['inttime'] # second
            start_ind = transit_ind - np.int(span / int_time)
            end_ind = transit_ind + np.int(span / int_time) + 1 # plus 1 to make transit_ind is at the center

            start_ind = max(0, start_ind)
            end_ind = min(end_ind, ts.vis.shape[0])

            # # check if data contain this range
            # if start_ind < 0:
            #     raise RuntimeError('start_ind: %d < 0' % start_ind)
            # if end_ind > ts.vis.shape[0]:
            #     raise RuntimeError('end_ind: %d > %d' % (end_ind, ts.vis.shape[0]))

            if vis_conj:
                ts.local_vis[:] = ts.local_vis.conj()

            nt = end_ind - start_ind
            t_inds = range(start_ind, end_ind)
            freq = ts.freq[:]
            nf = len(freq)
            nlb = len(ts.local_bl[:])
            nfeed = len(feedno)
            tfp_inds = list(itertools.product(t_inds, range(nf), [pol.index('xx'), pol.index('yy')])) # only for xx and yy
            ns, ss, es = mpiutil.split_all(len(tfp_inds), comm=ts.comm)
            # gather data to make each process to have its own data which has all bls
            for ri, (ni, si, ei) in enumerate(zip(ns, ss, es)):
                lvis = np.zeros((ni, nlb), dtype=ts.vis.dtype)
                lvis_mask = np.zeros((ni, nlb), dtype=ts.vis_mask.dtype)
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

            cnan = complex(np.nan, np.nan) # complex nan
            if save_src_vis or subtract_src:
                # save calibrator src vis
                lsrc_vis = np.full((tfp_len, nfeed, nfeed), cnan, dtype=ts.vis.dtype)
                if save_src_vis:
                    # save sky vis
                    lsky_vis = np.full((tfp_len, nfeed, nfeed), cnan, dtype=ts.vis.dtype)
                    # save outlier vis
                    lotl_vis = np.full((tfp_len, nfeed, nfeed), cnan, dtype=ts.vis.dtype)

            if apply_gain or save_gain:
                lGain = np.full((tfp_len, nfeed), cnan, dtype=ts.vis.dtype)

            # find indices mapping between Vmat and vis
            # bis = range(nbl)
            bis_conj = [] # indices that shold be conj
            mis = [] # indices in the nfeed x nfeed matrix by flatten it to a vector
            mis_conj = [] # indices (of conj vis) in the nfeed x nfeed matrix by flatten it to a vector
            for bi, (fdi, fdj) in enumerate(bls):
                ai, aj = feedno.index(fdi), feedno.index(fdj)
                mis.append(ai * nfeed + aj)
                if ai != aj:
                    bis_conj.append(bi)
                    mis_conj.append(aj * nfeed + ai)

            # construct visibility matrix for a single time, freq, pol
            Vmat = np.full((nfeed, nfeed), cnan, dtype=ts.vis.dtype)
            Sc = s.get_jys()
            if show_progress and mpiutil.rank0:
                pg = progress.Progress(tfp_len, step=progress_step)
            for ii, (ti, fi, pi) in enumerate(tfp_linds):
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
                Vmat.flat[mis] = np.ma.array(this_vis[ii], mask=this_vis_mask[ii]).filled(cnan)
                Vmat.flat[mis_conj] = np.ma.array(this_vis[ii, bis_conj], mask=this_vis_mask[ii, bis_conj]).conj().filled(cnan)

                if save_src_vis:
                    lsky_vis[ii] = Vmat

                # set invalid val to 0
                invalid = ~np.isfinite(Vmat) # a bool array
                # if too many masks
                if np.where(invalid)[0].shape[0] > 0.3 * nfeed**2:
                    continue
                Vmat[invalid] = 0

                # fill diagonal of Vmat to 0
                if zero_diag:
                    np.fill_diagonal(Vmat, 0)

                # initialize the outliers
                med = np.median(Vmat.real) + 1.0J * np.median(Vmat.imag)
                diff = Vmat - med
                S0 = np.where(np.abs(diff)>3.0*rpca_decomp.MAD(Vmat), diff, 0)
                # stable PCA decomposition
                V0, S = rpca_decomp.decompose(Vmat, rank=1, S=S0, max_iter=100, threshold='hard', tol=1.0e-6, debug=False)
                # V0, S = rpca_decomp.decompose(Vmat, rank=1, S=S0, max_iter=100, threshold='soft', tol=1.0e-6, debug=False)
                if save_src_vis or subtract_src:
                    lsrc_vis[ii] = V0
                    if save_src_vis:
                        lotl_vis[ii] = S

                # plot
                if plot_figs:
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

                if apply_gain or save_gain:
                    e, U = la.eigh(V0 / Sc[fi], eigvals=(nfeed-1, nfeed-1))
                    g = U[:, -1] * e[-1]**0.5
                    if g[0].real < 0:
                        g *= -1.0 # make all g[0] phase 0, instead of pi
                    lGain[ii] = g

                    # plot Gain
                    if plot_figs:
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
                lv = lv.redistribute(axis=1).local_array.reshape(nt, nf, 2, -1)
                if 'ns_on' in ts.iterkeys():
                    lv[ts['ns_on'][:]] = 0 # avoid ns_on signal to become nan
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
                        shp = (nt, nf, 2, nfeed, nfeed)
                        f.create_dataset('sky_vis', shp, dtype=lsky_vis.dtype)
                        f.create_dataset('src_vis', shp, dtype=lsrc_vis.dtype)
                        f.create_dataset('outlier_vis', shp, dtype=lotl_vis.dtype)
                        f.attrs['calibrator'] = calibrator
                        f.attrs['dim'] = 'time, freq, pol, feed, feed'
                        try:
                            f.attrs['time'] = ts.time[start_ind:end_ind]
                        except RuntimeError:
                            f.create_dataset('time', data=ts.time[start_ind:end_ind])
                            f.attrs['time'] = '/time'
                        f.attrs['freq'] = freq
                        f.attrs['pol'] = np.array(['xx', 'yy'])
                        f.attrs['feed'] = np.array(feedno)

                mpiutil.barrier()

                # write data to file
                for i in range(10):
                    try:
                        # NOTE: if write simultaneously, will loss data with processes distributed in several nodes
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
                # flag outliers in lGain along each feed
                lG_abs = np.full_like(lGain, np.nan, dtype=lGain.real.dtype)
                for i in range(lGain.shape[0]):
                    valid_inds = np.where(np.isfinite(lGain[i]))[0]
                    if len(valid_inds) > 3:
                        vabs = np.abs(lGain[i, valid_inds])
                        vmed = np.median(vabs)
                        vabs_diff = np.abs(vabs - vmed)
                        vmad = np.median(vabs_diff) / 0.6745
                        lG_abs[i, valid_inds] = np.where(vabs_diff>3.0*vmad, np.nan, vabs)

                # choose data slice near the transit time
                li = max(start_ind, transit_ind - 10) - start_ind
                hi = min(end_ind, transit_ind + 10 + 1) - start_ind
                # compute s_top for this time range
                n0 = np.zeros(((hi-li), 3))
                for ti, jt in enumerate(ts.time[start_ind:end_ind][li:hi]):
                    aa.set_jultime(jt)
                    s.compute(aa)
                    n0[ti] = s.get_crds('top', ncrd=3)
                if save_phs_change:
                    n0t = np.zeros((nt, 3))
                    for ti, jt in enumerate(ts.time[start_ind:end_ind]):
                        aa.set_jultime(jt)
                        s.compute(aa)
                        n0t[ti] = s.get_crds('top', ncrd=3)

                # get the positions of feeds
                feedpos = ts['feedpos'][:]

                # wrap and redistribute Gain and flagged G_abs
                Gain = mpiarray.MPIArray.wrap(lGain, axis=0, comm=ts.comm)
                Gain = Gain.redistribute(axis=1).reshape(nt, nf, 2, None).redistribute(axis=0).reshape(None, nf*2*nfeed).redistribute(axis=1)
                G_abs = mpiarray.MPIArray.wrap(lG_abs, axis=0, comm=ts.comm)
                G_abs = G_abs.redistribute(axis=1).reshape(nt, nf, 2, None).redistribute(axis=0).reshape(None, nf*2*nfeed).redistribute(axis=1)

                fpd_inds = list(itertools.product(range(nf), range(2), range(nfeed))) # only for xx and yy
                fpd_linds = mpiutil.mpilist(fpd_inds, method='con', comm=ts.comm)
                del fpd_inds
                # create data to save the solved gain for each feed
                lgain = np.full((len(fpd_linds),), cnan, dtype=Gain.dtype) # gain for each feed
                if save_phs_change:
                    lphs = np.full((nt, len(fpd_linds)), np.nan, dtype=Gain.real.dtype) # phase change with time for each feed

                # check for conj
                num_conj = 0
                for ii, (fi, pi, di) in enumerate(fpd_linds):
                    y = G_abs.local_array[li:hi, ii]
                    inds = np.where(np.isfinite(y))[0]
                    if len(inds) >= max(4, 0.5 * len(y)):
                        # get the approximate magnitude by averaging the central G_abs
                        # solve phase by least square fit
                        ui = (feedpos[di] - feedpos[0]) * (1.0e6*freq[fi]) / const.c # position of this feed (relative to the first feed) in unit of wavelength
                        exp_factor = np.exp(2.0J * np.pi * np.dot(n0, ui))
                        ef = exp_factor
                        Gi = Gain.local_array[li:hi, ii]
                        e_phs = np.dot(ef[inds].conj(), Gi[inds]/y[inds]) / len(inds)
                        ea = np.abs(e_phs)
                        e_phs_conj = np.dot(ef[inds], Gi[inds]/y[inds]) / len(inds)
                        eac = np.abs(e_phs_conj)
                        if eac > ea:
                            num_conj += 1
                # reduce num_conj from all processes
                num_conj = mpiutil.allreduce(num_conj, comm=ts.comm)
                if num_conj > 0.5 * (nf * 2 * nfeed): # 2 for 2 pols
                    if mpiutil.rank0:
                        print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
                        print '!!!   Detect data should be their conjugate...   !!!'
                        print '!!!   Correct it automatically...                !!!'
                        print '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
                    mpiutil.barrier()
                    # correct vis
                    ts.local_vis[:] = ts.local_vis.conj()
                    # correct G
                    Gain.local_array[:] = Gain.local_array.conj()

                # solve for gain
                for ii, (fi, pi, di) in enumerate(fpd_linds):
                    y = G_abs.local_array[li:hi, ii]
                    inds = np.where(np.isfinite(y))[0]
                    if len(inds) >= max(4, 0.5 * len(y)):
                        # get the approximate magnitude by averaging the central G_abs
                        mag = np.mean(y[inds])
                        # solve phase by least square fit
                        ui = (feedpos[di] - feedpos[0]) * (1.0e6*freq[fi]) / const.c # position of this feed (relative to the first feed) in unit of wavelength
                        exp_factor = np.exp(2.0J * np.pi * np.dot(n0, ui))
                        ef = exp_factor
                        Gi = Gain.local_array[li:hi, ii]
                        e_phs = np.dot(ef[inds].conj(), Gi[inds]/y[inds]) / len(inds)
                        ea = np.abs(e_phs)
                        if np.abs(ea - 1.0) < 0.1:
                            # compute gain for this feed
                            lgain[ii] = mag * e_phs
                            if save_phs_change:
                                lphs[:, ii] = np.angle(np.exp(-2.0J * np.pi * np.dot(n0t, ui)) * Gain.local_array[:, ii])
                        else:
                            e_phs_conj = np.dot(ef[inds], Gi[inds]/y[inds]) / len(inds)
                            eac = np.abs(e_phs_conj)
                            if eac > ea:
                                if np.abs(eac - 1.0) < 0.01:
                                    print 'feedno = %d, fi = %d, pol = %s: may need to be conjugated' % (feedno[di], fi, gain_pd[pi])
                            else:
                                print 'feedno = %d, fi = %d, pol = %s: maybe wrong abs(e_phs): %s' % (feedno[di], fi, gain_pd[pi], ea)


                # gather local gain
                gain = mpiutil.gather_array(lgain, axis=0, root=None, comm=ts.comm)
                del lgain
                gain = gain.reshape(nf, 2, nfeed)
                if save_phs_change:
                    phs = mpiutil.gather_array(lphs, axis=1, root=0, comm=ts.comm)
                    del lphs
                    if mpiutil.rank0:
                        phs = phs.reshape(nt, nf, 2, nfeed)

                # apply gain to vis
                if apply_gain:
                    for fi in range(nf):
                        for pi in [pol.index('xx'), pol.index('yy')]:
                            pi_ = gain_pd[pol[pi]]
                            for bi, (fd1, fd2) in enumerate(ts['blorder'].local_data):
                                g1 = gain[fi, pi_, feedno.index(fd1)]
                                g2 = gain[fi, pi_, feedno.index(fd2)]
                                if np.isfinite(g1) and np.isfinite(g2):
                                    ts.local_vis[:, fi, pi, bi] /= (g1 * np.conj(g2))
                                else:
                                    # mask the un-calibrated vis
                                    ts.local_vis_mask[:, fi, pi, bi] = True

                # save gain to file
                if save_gain:
                    if tag_output_iter:
                        gain_file = output_path(gain_file, iteration=self.iteration)
                    else:
                        gain_file = output_path(gain_file)
                    if mpiutil.rank0:
                        with h5py.File(gain_file, 'w') as f:
                            # allocate space for Gain
                            dset = f.create_dataset('Gain', (nt, nf, 2, nfeed), dtype=Gain.dtype)
                            dset.attrs['calibrator'] = calibrator
                            dset.attrs['dim'] = 'time, freq, pol, feed'
                            try:
                                dset.attrs['time'] = ts.time[start_ind:end_ind]
                            except RuntimeError:
                                f.create_dataset('time', data=ts.time[start_ind:end_ind])
                                dset.attrs['time'] = '/time'
                            dset.attrs['freq'] = freq
                            dset.attrs['pol'] = np.array(['xx', 'yy'])
                            dset.attrs['feed'] = np.array(feedno)
                            # save gain
                            dset = f.create_dataset('gain', data=gain)
                            dset.attrs['calibrator'] = calibrator
                            dset.attrs['dim'] = 'freq, pol, feed'
                            dset.attrs['freq'] = freq
                            dset.attrs['pol'] = np.array(['xx', 'yy'])
                            dset.attrs['feed'] = np.array(feedno)
                            # save phs
                            if save_phs_change:
                                f.create_dataset('phs', data=phs)

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
                        raise RuntimeError('Could not open file: %s...' % src_vis_file)

                    mpiutil.barrier()


        # convert vis from intensity unit to temperature unit in K
        if temperature_convert:
            if 'unit' in ts.vis.attrs.keys() and ts.vis.attrs['unit'] == 'K':
                if mpiutil.rank0:
                    print 'vis is already in unit K, do nothing...'
            else:
                factor = 1.0e-26 * (const.c**2 / (2 * const.k_B * (1.0e6*freq)**2)) # NOTE: 1Jy = 1.0e-26 W m^-2 Hz^-1
                ts.local_vis[:] *= factor[np.newaxis, :, np.newaxis, np.newaxis]
                ts.vis.attrs['unit'] = 'K'


        return super(PsCal, self).process(ts)
