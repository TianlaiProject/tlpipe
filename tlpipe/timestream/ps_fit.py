"""Calibration by strong point source fitting."""

import os
import numpy as np
import ephem
import aipy as a
import tod_task
from caput import mpiutil


def fit(vis_obs, vis_sim, start_ind, end_ind, num_shift, idx, plot_fit, fig_prefix):
    vis_obs = np.ma.masked_invalid(vis_obs)
    num_nomask = vis_obs.count()
    # vis_sim = np.ma.masked_invalid(vis_sim)
    fi, pi, (i, j) = idx
    gains = []
    chi2s = []
    shifts = range(-num_shift/2, num_shift/2+1)
    for si in shifts:
        vis = vis_obs[start_ind+si:end_ind+si]
        xx = np.ma.dot(vis_sim.conj(), vis_sim)
        xy = np.ma.dot(vis_sim.conj(), vis)
        gain = xy / xx
        vis_cal = gain * vis_sim
        err = vis - vis_cal
        chi2 = np.ma.dot(err.conj(), err)
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
    chi2 = chi2s[ind]
    si = shifts[ind]
    obs_data = vis_obs[start_ind:end_ind].copy()
    factor = np.max(np.ma.abs(obs_data)) / np.max(np.abs(vis_sim))
    obs_data /= factor # make amp close to each other
    vis_cal = vis_obs[start_ind+si:end_ind+si].copy() / gain
    if si != 0 and mpiutil.rank0:
        print 'shift %d for %s...' % (si, idx)

    if plot_fit:
        # plot the fit
        import tlpipe.plot
        import matplotlib.pyplot as plt
        from tlpipe.utils.path_util import output_path

        plt.figure()
        plt.subplot(311)
        plt.plot(obs_data.real, label='obs, real')
        plt.plot(vis_cal.real, label='cal, real')
        plt.plot(vis_sim.real, label='sim, real')
        plt.legend(loc='best')
        plt.subplot(312)
        plt.plot(obs_data.imag, label='obs, imag')
        plt.plot(vis_cal.imag, label='cal, imag')
        plt.plot(vis_sim.imag, label='sim, imag')
        plt.legend(loc='best')
        plt.subplot(313)
        plt.plot(np.abs(obs_data), label='obs, abs')
        plt.plot(np.abs(vis_cal), label='cal, abs')
        plt.plot(np.abs(vis_sim), label='sim, abs')
        plt.legend(loc='best')
        fig_name = '%s_%d_%d_%d_%d.png' % (fig_prefix, fi, pi, i, j)
        fig_name = output_path(fig_name)
        plt.savefig(fig_name)
        plt.clf()

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
        bl = ts.bl.local_data[:] # local bls
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
        aa.set_jultime(ts['jul_date'][0] + 1.0)
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
        start_ind = transit_ind - np.int(span / int_time)
        end_ind = transit_ind + np.int(span / int_time)
        num_shift = np.int(shift / int_time)

        vis = ts['vis'].local_data.copy()
        vis[ts['ns_on'][:]] = complex(np.nan, np.nan) # mask noise on
        nt = end_ind - start_ind
        vis_sim = np.zeros((nt,)+vis.shape[1:], dtype=vis.dtype) # to hold the simulated vis

        for ind, ti in enumerate(range(start_ind, end_ind)):
            aa.set_jultime(ts['jul_date'][ti])
            s.compute(aa)
            # get fluxes vs. freq of the calibrator
            Sc = s.get_jys()
            # get the topocentric coordinate of the calibrator at the current time
            s_top = s.get_crds('top', ncrd=3)
            aa.sim_cache(cat.get_crds('eq', ncrd=3)) # for compute bm_response and sim
            # for pi in range(len(pol)):
            for pi in range(2): # only cal for xx, yy
                aa.set_active_pol(pol[pi])
                for bi, (i, j) in enumerate(bls):
                    ai = feedno.index(i)
                    aj = feedno.index(j)
                    uij = aa.gen_uvw(ai, aj, src='z')[:, 0, :] # (rj - ri)/lambda
                    bmij = aa.bm_response(ai, aj).reshape(-1)
                    # print uij.shape, bmij.shape
                    vis_sim[ind, :, pi, bi] = Sc * bmij * np.exp(-2.0J * np.pi * np.dot(s_top, uij))

        # iterate over freq
        for fi in range(nfreq):
            # for pi in range(len(pol)):
            for pi in range(2): # only cal for xx, yy
                for bi, (i, j) in enumerate(bls):
                    gain, si = fit(vis[:, fi, pi, bi].copy(), vis_sim[:, fi, pi, bi], start_ind, end_ind, num_shift, (fi, pi, (i, j)), plot_fit, fig_prefix)
                    # cal for vis
                    ts['vis'].local_data[:, fi, pi, bi] = np.roll(vis[:, fi, pi, bi], -si) / gain # NOTE the use of -si

        # set mask status of 'vis'
        ts['vis'].attrs['masked'] = True

        ts.add_history(self.history)

        return ts
