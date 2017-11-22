"""Calibration using a strong point source.

Inheritance diagram
-------------------

.. inheritance-diagram:: PsCal
   :parts: 2

"""

import re
import itertools
import warnings
import time
import numpy as np
from scipy import linalg as la
from scipy import optimize
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


# Equation for Gaussian
def fg(x, a, b, c, d):
    return a * np.exp(-(x - b)**2.0 / (2 * c**2)) + d

def fc(x, a, b, c, d):
    return a * np.sinc(c * (x - b)) + d


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
                    'catalog': 'misc,helm,nvss',
                    'span': 60, # second
                    'plot_figs': False,
                    'fig_name': 'gain/gain',
                    'save_src_vis': False, # save the extracted calibrator visibility
                    'src_vis_file': 'src_vis/src_vis.hdf5',
                    'subtract_src': False, # subtract vis of the calibrator from data
                    'apply_gain': True,
                    'save_gain': False,
                    'gain_file': 'gain/gain.hdf5',
                    'temperature_convert': False,
                  }

    prefix = 'pc_'

    def process(self, ts):

        assert isinstance(ts, Timestream), '%s only works for Timestream object' % self.__class__.__name__

        calibrator = self.params['calibrator']
        catalog = self.params['catalog']
        span = self.params['span']
        plot_figs = self.params['plot_figs']
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

        pol_type = ts['pol'].attrs['pol_type']
        if pol_type != 'linear':
            raise RuntimeError('Can not do ps_cal for pol_type: %s' % pol_type)

        ts.redistribute('baseline')

        feedno = ts['feedno'][:].tolist()
        pol = [ ts.pol_dict[p] for p in ts['pol'][:] ] # as string
        gain_pd = {'xx': 0, 'yy': 1,    0: 'xx', 1: 'yy'} # for gain related op
        bl = mpiutil.gather_array(ts.local_bl[:], root=None, comm=ts.comm)
        bls = [ tuple(b) for b in bl ]
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
        if transit_time > ts['jul_date'][-1]:
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

        # check if data contain this range
        if start_ind < 0:
            raise RuntimeError('start_ind: %d < 0' % start_ind)
        if end_ind > ts.vis.shape[0]:
            raise RuntimeError('end_ind: %d > %d' % (end_ind, ts.vis.shape[0]))

        ############################################
        # if ts.is_cylinder:
        #     ts.local_vis[:] = ts.local_vis.conj() # now for cylinder array
        ############################################

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
        lGain = np.empty((tfp_len, nfeed), dtype=ts.vis.dtype)
        lGain[:] = complex(np.nan, np.nan)
        if save_src_vis or subtract_src:
            # save calibrator src vis
            lsrc_vis = np.empty((tfp_len, nfeed, nfeed), dtype=ts.vis.dtype)
            lsrc_vis[:] = complex(np.nan, np.nan)
            if save_src_vis:
                # save sky vis
                lsky_vis = np.empty((tfp_len, nfeed, nfeed), dtype=ts.vis.dtype)
                lsky_vis[:] = complex(np.nan, np.nan)
                # save outlier vis
                lotl_vis = np.empty((tfp_len, nfeed, nfeed), dtype=ts.vis.dtype)
                lotl_vis[:] = complex(np.nan, np.nan)

        # construct visibility matrix for a single time, freq, pol
        Vmat = np.zeros((nfeed, nfeed), dtype=ts.vis.dtype)
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
            mask_cnt = 0
            for i, ai in enumerate(feedno):
                for j, aj in enumerate(feedno):
                    try:
                        bi = bls.index((ai, aj))
                        if this_vis_mask[ii, bi] and not np.isfinite(this_vis[ii, bi]):
                            mask_cnt += 1
                            Vmat[i, j] = 0
                        else:
                            Vmat[i, j] = this_vis[ii, bi] / Sc[fi] # xx, yy
                    except ValueError:
                        bi = bls.index((aj, ai))
                        if this_vis_mask[ii, bi] and not np.isfinite(this_vis[ii, bi]):
                            mask_cnt += 1
                            Vmat[i, j] = 0
                        else:
                            Vmat[i, j] = np.conj(this_vis[ii, bi] / Sc[fi]) # xx, yy

            if save_src_vis:
                lsky_vis[ii] = Vmat

            # if too many masks
            if mask_cnt > 0.3 * nfeed**2:
                continue

            # set invalid val to 0
            # Vmat = np.where(np.isfinite(Vmat), Vmat, 0)

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

            e, U = la.eigh(V0, eigvals=(nfeed-1, nfeed-1))
            g = U[:, -1] * e[-1]**0.5
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
            for ii, (ti, fi, pi) in enumerate(tfp_linds):
                for bi, (fd1, fd2) in enumerate(ts.local_bl):
                    b1, b2 = feedno.index(fd1), feedno.index(fd2)
                    ts.local_vis[ti, fi, pi, bi] -= lsrc_vis[ii, b1, b2]

        if not save_src_vis:
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
                    f.attrs['time'] = ts.time[start_ind:end_ind]
                    f.attrs['freq'] = freq
                    f.attrs['pol'] = np.array(['xx', 'yy'])
                    f.attrs['feed'] = np.array(feedno)

            mpiutil.barrier()

            # write data to file
            for i in range(10):
                try:
                    with h5py.File(src_vis_file, 'r+') as f:
                        for ii, (ti, fi, pi) in enumerate(tfp_linds):
                            ti_ = ti-start_ind
                            pi_ = gain_pd[pol[pi]]
                            f['sky_vis'][ti_, fi, pi_] = lsky_vis[ii]
                            f['src_vis'][ti_, fi, pi_] = lsrc_vis[ii]
                            f['outlier_vis'][ti_, fi, pi_] = lotl_vis[ii]
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

        del tfp_linds

        # flag outliers in lGain along each feed
        lG_abs = np.abs(lGain)
        median = np.ma.median(lG_abs, axis=1)
        abs_diff = np.ma.abs(lG_abs - median[:, np.newaxis])
        mad = np.ma.median(abs_diff, axis=1) / 0.6745
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'invalid value encountered in greater')
            warnings.filterwarnings('ignore', 'invalid value encountered in greater_equal')
            warnings.filterwarnings('ignore', 'invalid value encountered in absolute')
            lG_abs = np.where(abs_diff>3.0*mad[:, np.newaxis], np.nan, lG_abs)

        # choose data slice near the transit time
        c = nt/2 # center ind
        li = max(0, c - 100)
        hi = min(nt, c + 100 + 1)
        x = np.arange(li, hi)
        # compute s_top for this time range
        n0 = np.zeros(((hi-li), 3))
        for ti, jt in enumerate(ts.time[start_ind:end_ind][li:hi]):
            aa.set_jultime(jt)
            s.compute(aa)
            n0[ti] = s.get_crds('top', ncrd=3)

        # get the positions of feeds
        feedpos = ts['feedpos'][:]

        # wrap and redistribute Gain and flagged G_abs
        Gain = mpiarray.MPIArray.wrap(lGain, axis=0, comm=ts.comm)
        Gain = Gain.redistribute(axis=1).reshape(nt, nf, 2, None).redistribute(axis=0).reshape(None, nf*2*nfeed).redistribute(axis=1)
        G_abs = mpiarray.MPIArray.wrap(lG_abs, axis=0, comm=ts.comm)
        G_abs = G_abs.redistribute(axis=1).reshape(nt, nf, 2, None).redistribute(axis=0).reshape(None, nf*2*nfeed).redistribute(axis=1)

        fpd_inds = list(itertools.product(range(nf), range(2), range(nfeed))) # only for xx and yy
        fpd_linds = mpiutil.mpilist(fpd_inds)
        del fpd_inds
        # create data to save the solved gain for each feed
        lgain = np.zeros((len(fpd_linds),), dtype=Gain.dtype) # gain for each feed
        lgain[:] = complex(np.nan, np.nan)
        for ii, (fi, pi, di) in enumerate(fpd_linds):
            # gaussian/sinc fit
            y = G_abs.local_array[li:hi, ii]
            inds = np.where(np.isfinite(y))[0]
            if len(inds) >= max(4, 0.75 * len(y)): # min 4 for curve_fit
                # get the best estimate of the central val
                cval = y[inds[np.argmin(np.abs(inds-c))]]
                try:
                    # gaussian fit
                    # popt, pcov = optimize.curve_fit(fg, x[inds], y[inds], p0=(cval, c, 90, 0))
                    # sinc function seems fit better
                    popt, pcov = optimize.curve_fit(fc, x[inds], y[inds], p0=(cval, c, 1.0e-2, 0))
                    # print 'popt:', popt
                except Exception:
                    print 'curve_fit failed for fi = %d, pol = %s, feed = %d' % (fi, gain_pd[pi], feedno[fi])
                    continue

                An = y / fc(popt[1], *popt) # the beam profile
                ui = (feedpos[di] - feedpos[0]) * (1.0e6*freq[fi]) / const.c # position of this feed (relative to the first feed) in unit of wavelength
                exp_factor = np.exp(2.0J * np.pi * np.dot(n0, ui))
                Ae = An * exp_factor
                Gi = Gain.local_array[li:hi, ii]
                # compute gain for this feed
                lgain[ii] = np.dot(Ae[inds].conj(), Gi[inds]) / np.dot(Ae[inds].conj(), Ae[inds])


        # gather local gain
        gain = mpiutil.gather_array(lgain, axis=0, root=None, comm=ts.comm)
        del lgain
        gain = gain.reshape(nf, 2, nfeed)

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
                    dset.attrs['time'] = ts.time[start_ind:end_ind]
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

            mpiutil.barrier()
            del gain

            # save Gain
            for i in range(10):
                try:
                    with h5py.File(gain_file, 'r+') as f:
                        for ii, (fi, pi, di) in enumerate(fpd_linds):
                            f['Gain'][:, fi, pi, di] = Gain.local_array[:, ii]
                    break
                except IOError:
                    time.sleep(0.5)
                    continue
            else:
                raise RuntimeError('Could not open file: %s...' % src_vis_file)

            del Gain

            mpiutil.barrier()


        # convert vis from intensity unit to temperature unit in K
        if temperature_convert:
            factor = 1.0e-26 * (const.c**2 / (2 * const.k_B * (1.0e6*freq)**2)) # NOTE: 1Jy = 1.0e-26 W m^-2 Hz^-1
            ts.local_vis[:] *= factor[np.newaxis, :, np.newaxis, np.newaxis]
            ts.vis.attrs['unit'] = 'K'


        return super(PsCal, self).process(ts)
