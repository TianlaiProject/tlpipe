"""Calibration by strong point source fitting."""

import os
import numpy as np
import ephem
import aipy as a
import tod_task
from caput import mpiutil
from tlpipe.utils.path_util import output_path
import tlpipe.plot
import matplotlib.pyplot as plt


def fit(vis_obs, vis_mask, vis_sim, start_ind, end_ind, num_shift, idx, plot_fit, fig_prefix, iteration):
    vis_obs = np.ma.array(vis_obs, mask=vis_mask)
    num_nomask = vis_obs.count()
    if num_nomask == 0: # no valid vis data
        return 1.0, 0

    fi, pi, (i, j) = idx

    gains = []
    chi2s = []
    shifts = xrange(-num_shift/2, num_shift/2+1)
    for si in shifts:
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
    factor = np.max(np.ma.abs(obs_data)) / np.max(np.abs(vis_sim))
    obs_data = obs_data / factor # make amp close to each other
    vis_cal = np.ma.array(vis_obs[start_ind+si:end_ind+si], mask=vis_mask[start_ind+si:end_ind+si]) / gain
    if si != 0 and mpiutil.rank0:
        print 'shift %d for %s...' % (si, idx)

    if plot_fit:
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
        fig_name = output_path(fig_name, iteration=iteration)
        plt.savefig(fig_name)
        plt.close()

    return gain, si


class PsFit(tod_task.IterTimestream):
    """Calibration by strong point source fitting."""

    params_init = {
                    'calibrator': 'cas',
                    'catalog': 'misc,helm',
                    'span': 1200.0, # second
                    'shift': 600.0, # second
                    'plot_fit': False, # plot the smoothing fit
                    'fig_name': 'fit',
                  }

    prefix = 'pf_'

    def process(self, ts):

        calibrator = self.params['calibrator']
        catalog = self.params['catalog']
        span = self.params['span']
        shift = self.params['shift']
        plot_fit = self.params['plot_fit']
        fig_prefix = self.params['fig_name']

        ts.redistribute('baseline')

        feedno = ts['feedno'][:].tolist()
        nfreq = len(ts['freq'][:])
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

        print transit_inds

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
        vis_sim = np.zeros((nt,)+vis.shape[1:], dtype=vis.dtype) # to hold the simulated vis

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
                for bi, (i, j) in enumerate(bls):
                    ai = feedno.index(i)
                    aj = feedno.index(j)
                    uij = aa.gen_uvw(ai, aj, src='z')[:, 0, :] # (rj - ri)/lambda
                    bmij = aa.bm_response(ai, aj).reshape(-1)
                    # vis_sim[ind, :, pi, bi] = (const.c**2 / (2 * const.k_B * freq**2) / Omega_ij) * Sc * bmij * np.exp(-2.0J * np.pi * np.dot(s_top, uij)) # Unit: K
                    vis_sim[ind, :, pi, bi] = Sc * bmij * np.exp(-2.0J * np.pi * np.dot(s_top, uij))

        # iterate over freq
        for fi in xrange(nfreq):
            # for pi in xrange(len(pol)):
            for pi in xrange(2): # only cal for xx, yy
                for bi, (i, j) in enumerate(bls):
                    gain, si = fit(vis[:, fi, pi, bi], vis_mask[:, fi, pi, bi], vis_sim[:, fi, pi, bi], start_ind, end_ind, num_shift, (fi, pi, (i, j)), plot_fit, fig_prefix, self.iteration)
                    # cal for vis
                    ts.local_vis[:, fi, pi, bi] = np.roll(vis[:, fi, pi, bi], -si) / gain # NOTE the use of -si
                    ts.local_vis_mask[:, fi, pi, bi] = np.roll(vis_mask[:, fi, pi, bi], -si) # NOTE the use of -si

        ts.add_history(self.history)

        return ts
