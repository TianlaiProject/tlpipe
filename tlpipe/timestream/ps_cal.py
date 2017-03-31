"""Calibration using a strong point source.

Inheritance diagram
-------------------

.. inheritance-diagram:: PsCal
   :parts: 2

"""

import re
import itertools
import numpy as np
from scipy import linalg as la
from scipy import optimize
import ephem
import h5py
import aipy as a
import tod_task
from timestream import Timestream
from tlpipe.core import constants as const

from caput import mpiutil
from tlpipe.utils.path_util import output_path
from tlpipe.utils import rpca_decomp
import tlpipe.plot
import matplotlib.pyplot as plt


# Equation for Gaussian
def fg(x, a, b, c, d):
    return a * np.exp(-(x - b)**2.0 / (2 * c**2)) + d

def fc(x, a, b, c, d):
    return a * np.sinc(c * (x - b)) + d


class PsCal(tod_task.TaskTimestream):
    """Calibration using a strong point source.

    The calibration is done by using the Eigen-decomposition method.

    May be more explain to this...

    """

    params_init = {
                    'calibrator': 'cyg',
                    'catalog': 'misc,helm,nvss',
                    'span': 60, # second
                    'plot_figs': False,
                    'fig_name': 'gain/gain',
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
        save_gain = self.params['save_gain']
        gain_file = self.params['gain_file']
        temperature_convert = self.params['temperature_convert']

        ts.redistribute('baseline')

        feedno = ts['feedno'][:].tolist()
        pol = [ ts.pol_dict[p] for p in ts['pol'][:] ]
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
        lGain = np.empty((len(tfp_linds), nfeed), dtype=np.complex128)
        lGain[:] = complex(np.nan, np.nan)

        # construct visibility matrix for a single time, freq, pol
        Vmat = np.zeros((nfeed, nfeed), dtype=ts.vis.dtype)
        Sc = s.get_jys()
        for ii, (ti, fi, pi) in enumerate(tfp_linds):
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

            # if too many masks
            if mask_cnt > 0.3 * nfeed**2:
                continue

            # Eigen decomposition
            # Vmat = np.where(np.isfinite(Vmat), Vmat, 0)
            V0, S = rpca_decomp.decompose(Vmat, max_iter=100, threshold='hard', tol=1.0e-6, debug=False)
            # V0, S = rpca_decomp.decompose(Vmat, max_iter=100, threshold='soft', tol=1.0e-6, debug=False)

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
                fig_name = '%s_V_%d_%d_%s.png' % (fig_prefix, ind, fi, ts.pol_dict[pi])
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
                fig_name = '%s_V0_%d_%d_%s.png' % (fig_prefix, ind, fi, ts.pol_dict[pi])
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
                fig_name = '%s_S_%d_%d_%s.png' % (fig_prefix, ind, fi, ts.pol_dict[pi])
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
                fig_name = '%s_N_%d_%d_%s.png' % (fig_prefix, ind, fi, ts.pol_dict[pi])
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
                plt.xlabel('Feed number')
                plt.legend()
                fig_name = '%s_ants_%d_%d_%s.png' % (fig_prefix, ind, fi, ts.pol_dict[pi])
                if tag_output_iter:
                    fig_name = output_path(fig_name, iteration=self.iteration)
                else:
                    fig_name = output_path(fig_name)
                plt.savefig(fig_name)
                plt.close()

        # gather Gain from each processes
        Gain = mpiutil.gather_array(lGain, axis=0, root=None, comm=ts.comm)
        Gain = Gain.reshape(nt, nf, 2, nfeed)

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

        # create data to save the solved gain for each feed
        local_fp_inds = mpiutil.mpilist(list(itertools.product(range(nf), range(2))))
        lgain = np.zeros((len(local_fp_inds), nfeed), dtype=Gain.dtype) # gain for each feed
        lgain[:] = complex(np.nan, np.nan)

        for ii, (fi, pi) in enumerate(local_fp_inds):
            data = np.abs(Gain[li:hi, fi, pi, :]).T
            # flag outliers
            median = np.ma.median(data, axis=0)
            abs_diff = np.ma.abs(data - median[np.newaxis, :])
            mad = np.ma.median(abs_diff, axis=0) / 0.6745
            data = np.where(abs_diff>3.0*mad[np.newaxis, :], np.nan, data)
            # gaussian/sinc fit
            for idx in range(nfeed):
                y = data[idx]
                inds = np.where(np.isfinite(y))[0]
                if len(inds) > 0.75 * len(y):
                    # get the best estimate of the central val
                    cval = y[inds[np.argmin(np.abs(inds-c))]]
                    try:
                        # gaussian fit
                        # popt, pcov = optimize.curve_fit(fg, x[inds], y[inds], p0=(cval, c, 90, 0))
                        # sinc function seems fit better
                        popt, pcov = optimize.curve_fit(fc, x[inds], y[inds], p0=(cval, c, 1.0e-2, 0))
                        # print 'popt:', popt
                    except RuntimeError:
                        print 'curve_fit failed for fi = %d, pol = %s, feed = %d' % (fi, ['xx', 'yy'][pi], feedno[idx])
                        continue

                    An = y / fc(popt[1], *popt) # the beam profile
                    ui = (feedpos[idx] - feedpos[0]) * (1.0e6*freq[fi]) / const.c # position of this feed (relative to the first feed) in unit of wavelength
                    exp_factor = np.exp(2.0J * np.pi * np.dot(n0, ui))
                    Ae = An * exp_factor
                    Gi = Gain[li:hi, fi, pi, idx]
                    # compute gain for this feed
                    lgain[ii, idx] = np.dot(Ae[inds].conj(), Gi[inds]) / np.dot(Ae[inds].conj(), Ae[inds])

        # gather local gain
        gain = mpiutil.gather_array(lgain, axis=0, root=None, comm=ts.comm)
        gain = gain.reshape(nf, 2, nfeed)

        # apply gain to vis
        for fi in range(nf):
            for pi in [pol.index('xx'), pol.index('yy')]:
                for bi, (fd1, fd2) in enumerate(ts['blorder'].local_data):
                    g1 = gain[fi, pi, feedno.index(fd1)]
                    g2 = gain[fi, pi, feedno.index(fd2)]
                    if np.isfinite(g1) and np.isfinite(g2):
                        ts.local_vis[:, fi, pi, bi] /= (g1 * np.conj(g2))
                    else:
                        # mask the un-calibrated vis
                        ts.local_vis_mask[:, fi, pi, bi] = True

        # convert vis from intensity unit to temperature unit in K
        if temperature_convert:
            factor = 1.0e-26 * (const.c**2 / (2 * const.k_B * (1.0e6*freq)**2)) # NOTE: 1Jy = 1.0e-26 W m^-2 Hz^-1
            ts.local_vis[:] *= factor[np.newaxis, :, np.newaxis, np.newaxis]
            ts.vis.attrs['unit'] = 'K'

        # save gain to file
        if mpiutil.rank0 and save_gain:
            if tag_output_iter:
                gain_file = output_path(gain_file, iteration=self.iteration)
            else:
                gain_file = output_path(gain_file)
            with h5py.File(gain_file, 'w') as f:
                # save Gain
                Gain = f.create_dataset('Gain', data=Gain)
                Gain.attrs['dim'] = 'time, freq, pol, feed'
                Gain.attrs['time'] = ts.time[start_ind:end_ind]
                Gain.attrs['freq'] = freq
                Gain.attrs['pol'] = np.array(['xx', 'yy'])
                Gain.attrs['feed'] = np.array(feedno)
                # save gain
                gain = f.create_dataset('gain', data=gain)
                gain.attrs['dim'] = 'freq, pol, feed'
                gain.attrs['freq'] = freq
                gain.attrs['pol'] = np.array(['xx', 'yy'])
                gain.attrs['feed'] = np.array(feedno)


        return super(PsCal, self).process(ts)
