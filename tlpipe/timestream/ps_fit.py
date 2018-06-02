"""Calibration by strong point source fitting.

Inheritance diagram
-------------------

.. inheritance-diagram:: PsFit
   :parts: 2

"""

import os
import numpy as np
import ephem
import aipy as a
import timestream_task
from tlpipe.container.timestream import Timestream
from caput import mpiutil
from tlpipe.utils.path_util import output_path
from tlpipe.core import constants as const
import tlpipe.plot
import matplotlib.pyplot as plt


def fit(vis_obs, vis_mask, vis_sim, start_ind, end_ind, num_shift, idx, plot_fit, fig_prefix, iteration, tag_output_iter, bls_plt, freq_plt):
    vis_obs = np.ma.array(vis_obs, mask=vis_mask)
    num_nomask = vis_obs.count()
    if num_nomask == 0: # no valid vis data
        return 1.0, 0

    fi, pi, (i, j) = idx

    gains = []
    chi2s = []
    shifts = xrange(-num_shift/2, num_shift/2+1)
    for si in shifts:
        # vis = vis_obs[start_ind+si:end_ind+si].astype(np.complex128) # improve precision
        vis = vis_obs[start_ind+si:end_ind+si]
        num_nomask = vis.count()
        if num_nomask == 0: # no valid vis data
            continue
        else:
            xx = np.ma.dot(vis_sim.conj(), vis_sim)
            xy = np.ma.dot(vis_sim.conj(), vis)
            gain = xy / xx
            vis_cal = gain * vis_sim
            err = vis - vis_cal
            chi2 = np.ma.dot(err.conj(), err).real
            gains.append(gain)
            chi2s.append(chi2/num_nomask)

    if len(gains) == 0: # no valid vis data
        return 1.0, 0

    chi2s = np.array(chi2s)
    if np.allclose(chi2s, np.sort(chi2s)):
        if mpiutil.rank0:
            print 'Warn: chi2 increasing for %s...' % (idx,)
    if np.allclose(chi2s, np.sort(chi2s)[::-1]):
        if mpiutil.rank0:
            print 'Warn: chi2 decreasing for %s...' % (idx,)

    ind = np.argmin(chi2s)
    gain = gains[ind]
    # chi2 = chi2s[ind]
    si = shifts[ind]
    obs_data = np.ma.array(vis_obs[start_ind:end_ind], mask=vis_mask[start_ind:end_ind])
    factor = np.ma.max(np.ma.abs(obs_data)) / np.max(np.abs(vis_sim))
    obs_data = obs_data / factor # make amp close to each other
    vis_cal = np.ma.array(vis_obs[start_ind+si:end_ind+si], mask=vis_mask[start_ind+si:end_ind+si]) / gain
    if si != 0 and mpiutil.rank0:
        print 'shift %d for %s...' % (si, idx)

    if plot_fit and (fi in freq_plt and (i, j) in bls_plt):
        # plot the fit
        plt.figure()
        plt.subplot(311)
        plt.plot(obs_data.real, label='obs, real')
        if not vis_cal is np.ma.masked: # in case gain is --
            plt.plot(vis_cal.real, label='cal, real')
        plt.plot(vis_sim.real, label='sim, real')
        plt.legend(loc='best')
        plt.subplot(312)
        plt.plot(obs_data.imag, label='obs, imag')
        if not vis_cal is np.ma.masked: # in case gain is --
            plt.plot(vis_cal.imag, label='cal, imag')
        plt.plot(vis_sim.imag, label='sim, imag')
        plt.legend(loc='best')
        plt.subplot(313)
        plt.plot(np.abs(obs_data), label='obs, abs')
        if not vis_cal is np.ma.masked: # in case gain is --
            plt.plot(np.abs(vis_cal), label='cal, abs')
        plt.plot(np.abs(vis_sim), label='sim, abs')
        plt.legend(loc='best')
        fig_name = '%s_%d_%d_%d_%d.png' % (fig_prefix, fi, pi, i, j)
        if tag_output_iter:
            fig_name = output_path(fig_name, iteration=iteration)
        else:
            fig_name = output_path(fig_name)
        plt.savefig(fig_name)
        plt.close()

    return gain, si


class PsFit(timestream_task.TimestreamTask):
    """Calibration by strong point source fitting.

    This works by minimize

    .. math:: \\chi^2 = \| V_{ij}^{\\text{obs}}(t + \\Delta t) - G_{ij} V_{ij}^{\\text{sim}}(t) \|^2

    Its solution is

    .. math:: G_{ij} = \\frac{V_{ij}^{\\text{sim} \\dagger} V_{ij}^{\\text{obs}}}{V_{ij}^{\\text{sim} \\dagger} V_{ij}^{\\text{sim}}}

    """

    params_init = {
                    'calibrator': 'cas',
                    'catalog': 'misc', # or helm,nvss
                    'span': 1200.0, # second
                    'shift': 600.0, # second
                    'plot_fit': False, # plot the smoothing fit
                    'fig_name': 'fit/fit',
                    'bl_incl': 'all', # or a list of include (bl1, bl2)
                    'bl_excl': [],
                    'freq_incl': 'all', # or a list of include freq idx
                    'freq_excl': [],
                  }

    prefix = 'pf_'

    def process(self, ts):

        assert isinstance(ts, Timestream), '%s only works for Timestream object' % self.__class__.__name__

        calibrator = self.params['calibrator']
        catalog = self.params['catalog']
        span = self.params['span']
        shift = self.params['shift']
        plot_fit = self.params['plot_fit']
        fig_prefix = self.params['fig_name']
        tag_output_iter = self.params['tag_output_iter']
        bl_incl = self.params['bl_incl']
        bl_excl = self.params['bl_excl']
        freq_incl = self.params['freq_incl']
        freq_excl = self.params['freq_excl']

        ts.redistribute('baseline')

        if bl_incl == 'all':
            bls_plt = [ tuple(bl) for bl in ts.local_bl ]
        else:
            bls_plt = [ bl for bl in bl_incl if not bl in bl_excl ]

        if freq_incl == 'all':
            freq_plt = range(ts.freq.shape[0])
        else:
            freq_plt = [ fi for fi in freq_incl if not fi in freq_excl ]

        feedno = ts['feedno'][:].tolist()
        freq = ts['freq'][:]
        nfreq = len(freq)
        pol = ts['pol'][:].tolist()
        bl = ts.local_bl[:] # local bls
        bls = [ tuple(b) for b in bl ]

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
        if transit_time > ts['jul_date'][-1]:
            local_next_transit = ephem.Date(next_transit + 8.0 * ephem.hour)
            raise RuntimeError('Data does not contain local transit time %s of source %s' % (local_next_transit, calibrator))

        # the first transit index
        transit_inds = [ np.searchsorted(ts['jul_date'][:], transit_time) ]
        # find all other transit indices
        aa.set_jultime(ts['jul_date'][0] + 1.0) # maybe should use a sidereal day which is a litter shorter than 1.0 ???
        transit_time = a.phs.ephem2juldate(aa.next_transit(s)) # Julian date
        cnt = 2
        while(transit_time <= ts['jul_date'][-1]):
            transit_inds.append(np.searchsorted(ts['jul_date'][:], transit_time))
            aa.set_jultime(ts['jul_date'][0] + 1.0*cnt)
            transit_time = a.phs.ephem2juldate(aa.next_transit(s)) # Julian date
            cnt += 1

        if mpiutil.rank0:
            print 'transit inds: ', transit_inds

        ### now only use the first transit point to do the cal
        ### may need to improve in the future
        transit_ind = transit_inds[0]
        int_time = ts.attrs['inttime'] # second
        start_ind = max(0, transit_ind - np.int(span / int_time))
        end_ind = min(len(ts.local_time), transit_ind + np.int(span / int_time))
        num_shift = np.int(shift / int_time)
        num_shift = min(num_shift, end_ind - start_ind)

        ############################################
        # if ts.is_cylinder:
        #     ts.local_vis[:] = ts.local_vis.conj() # now for cylinder array
        ############################################

        vis = ts.local_vis
        vis_mask = ts.local_vis_mask
        # vis[ts.local_vis_mask] = complex(np.nan, np.nan) # set masked vis to nan
        nt = end_ind - start_ind
        # vis_sim = np.zeros((nt,)+vis.shape[1:], dtype=np.complex128) # to hold the simulated vis, use float64 to have better precision
        vis_sim = np.zeros((nt,)+vis.shape[1:], dtype=vis.dtype)

        # get beam solid angle (suppose it is the same for all feeds)
        Omega_ij = aa[0].beam.Omega
        pre_factor = 1.0e-26 * (const.c**2 / (2 * const.k_B * (1.0e6*freq)**2) / Omega_ij) # NOTE: 1Jy = 1.0e-26 W m^-2 Hz^-1

        for ind, ti in enumerate(xrange(start_ind, end_ind)):
            aa.set_jultime(ts['jul_date'][ti])
            s.compute(aa)
            # get fluxes vs. freq of the calibrator
            Sc = s.get_jys()
            # get the topocentric coordinate of the calibrator at the current time
            s_top = s.get_crds('top', ncrd=3)
            aa.sim_cache(cat.get_crds('eq', ncrd=3)) # for compute bm_response and sim

            # for pi in range(len(pol)):
            for pi in xrange(2): # only cal for xx, yy
                aa.set_active_pol(pol[pi])
                # assume all have the same beam responce, speed the calculation
                # resp1 = aa[0].bm_response(s_top, pol=pol[pi][0]).transpose()
                # resp2 = aa[0].bm_response(s_top, pol=pol[pi][1]).transpose()
                # bmij = resp1 * np.conjugate(resp2)
                bmij = aa.bm_response(0, 0).reshape(-1)
                factor = pre_factor * Sc * bmij
                for bi, (i, j) in enumerate(bls):
                    ai = feedno.index(i)
                    aj = feedno.index(j)
                    uij = aa.gen_uvw(ai, aj, src='z')[:, 0, :] # (rj - ri)/lambda
                    # bmij = aa.bm_response(ai, aj).reshape(-1)
                    vis_sim[ind, :, pi, bi] = factor * np.exp(-2.0J * np.pi * np.dot(s_top, uij)) # Unit: K
                    # vis_sim[ind, :, pi, bi] = Sc * bmij * np.exp(-2.0J * np.pi * np.dot(s_top, uij))

        mpiutil.barrier()

        # iterate over freq
        for fi in xrange(nfreq):
            # for pi in xrange(len(pol)):
            for pi in xrange(2): # only cal for xx, yy
                for bi, (i, j) in enumerate(bls):
                    gain, si = fit(vis[:, fi, pi, bi], vis_mask[:, fi, pi, bi], vis_sim[:, fi, pi, bi], start_ind, end_ind, num_shift, (fi, pi, (i, j)), plot_fit, fig_prefix, self.iteration, tag_output_iter, bls_plt, freq_plt)
                    # cal for vis
                    ts.local_vis[:, fi, pi, bi] = np.roll(vis[:, fi, pi, bi], -si) / gain # NOTE the use of -si
                    ts.local_vis_mask[:, fi, pi, bi] = np.roll(vis_mask[:, fi, pi, bi], -si) # NOTE the use of -si

        mpiutil.barrier()

        return super(PsFit, self).process(ts)
