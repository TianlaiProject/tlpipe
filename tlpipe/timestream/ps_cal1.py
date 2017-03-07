"""Calibration using a strong point source.

Inheritance diagram
-------------------

.. inheritance-diagram:: PsCal
   :parts: 2

"""

import itertools
import numpy as np
from scipy import linalg as la
import ephem
import aipy as a
import tod_task

from caput import mpiutil
from tlpipe.utils.path_util import output_path
from tlpipe.utils import rpca_decomp
import tlpipe.plot
import matplotlib.pyplot as plt


class PsCal(tod_task.TaskTimestream):
    """Calibration using a strong point source.

    The calibration is done by using the Eigen-decomposition method.

    May be more explain to this...

    """

    params_init = {
                    'calibrator': 'cyg',
                    'catalog': 'misc,helm,nvss',
                    'span': 60, # second
                    'fig_name': 'gain/gain',
                    'save_gain': False,
                    'gain_file': 'gain.hdf5',
                  }

    prefix = 'pc_'

    def process(self, ts):

        calibrator = self.params['calibrator']
        catalog = self.params['catalog']
        span = self.params['span']
        fig_prefix = self.params['fig_name']
        tag_output_iter = self.params['tag_output_iter']
        save_gain = self.params['save_gain']
        gain_file = self.params['gain_file']

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

        if mpiutil.rank0:
            print 'transit ind of %s: %s' % (calibrator, transit_inds)

        ### now only use the first transit point to do the cal
        ### may need to improve in the future
        transit_ind = transit_inds[0]
        int_time = ts.attrs['inttime'] # second
        start_ind = transit_ind - np.int(span / int_time)
        end_ind = transit_ind + np.int(span / int_time)

        nt = end_ind - start_ind
        t_inds = range(start_ind, end_ind)
        nf = len(ts.freq[:])
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
        lgain = np.empty((len(tfp_linds), nfeed), dtype=np.complex128)
        lgain[:] = complex(np.nan, np.nan)

        # construct visibility matrix for a single time, freq, pol
        Vmat = np.zeros((nfeed, nfeed), dtype=ts.vis.dtype)
        for ii, (ti, fi, pi) in enumerate(tfp_linds):
            # when noise on, just pass
            if 'ns_on' in ts.iterkeys() and ts['ns_on'][ti]:
                continue
            aa.set_jultime(ts['jul_date'][ti])
            s.compute(aa)
            # get fluxes vs. freq of the calibrator
            Sc = s.get_jys()
            # get the topocentric coordinate of the calibrator at the current time
            # s_top = s.get_crds('top', ncrd=3)
            # aa.sim_cache(cat.get_crds('eq', ncrd=3)) # for compute bm_response and sim
            for i, ai in enumerate(feedno):
                for j, aj in enumerate(feedno):
                    try:
                        bi = bls.index((ai, aj))
                        if this_vis_mask[ii, bi]:
                            Vmat[i, j] = 0
                        else:
                            Vmat[i, j] = this_vis[ii, bi] / Sc[fi] # xx, yy
                    except ValueError:
                        bi = bls.index((aj, ai))
                        if this_vis_mask[ii, bi]:
                            Vmat[i, j] = 0
                        else:
                            Vmat[i, j] = np.conj(this_vis[ii, bi] / Sc[fi]) # xx, yy

            # Eigen decomposition
            Vmat = np.where(np.isfinite(Vmat), Vmat, 0)
            V0, S = rpca_decomp.decompose(Vmat, debug=True)

            # plot
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
            lgain[ii] = g

            # plot gain
            plt.figure()
            plt.plot(range(1, nfeed+1), g.real, 'b-', label='real')
            plt.plot(range(1, nfeed+1), g.real, 'bo')
            plt.plot(range(1, nfeed+1), g.imag, 'g-', label='imag')
            plt.plot(range(1, nfeed+1), g.imag, 'go')
            plt.plot(range(1, nfeed+1), np.abs(g), 'r-', label='abs')
            plt.plot(range(1, nfeed+1), np.abs(g), 'ro')
            plt.xlim(0, nfeed+2)
            plt.xlabel('Feed number')
            plt.legend()
            fig_name = '%s_ants_%d_%d_%s.png' % (fig_prefix, ind, fi, ts.pol_dict[pi])
            if tag_output_iter:
                fig_name = output_path(fig_name, iteration=self.iteration)
            else:
                fig_name = output_path(fig_name)
            plt.savefig(fig_name)
            plt.close()

        # gather gain from each processes
        gain = mpiutil.gather_array(lgain, axis=0, root=0, comm=ts.comm)
        if mpiutil.rank0:
            gain = gain.reshape(nt, nf, 2, nfeed)
            for idx, fd in enumerate(feedno):
                plt.figure()
                plt.plot(np.abs(gain[:, 0, 0, idx]), label='xx') # only plot fi == 0
                plt.plot(np.abs(gain[:, 0, 1, idx]), label='yy') # only plot fi == 0
                plt.legend()
                fig_name = '%s_%d.png' % (fig_prefix, fd)
                if tag_output_iter:
                    fig_name = output_path(fig_name, iteration=self.iteration)
                else:
                    fig_name = output_path(fig_name)
                plt.savefig(fig_name)
                plt.close()


        return super(PsCal, self).process(ts)
