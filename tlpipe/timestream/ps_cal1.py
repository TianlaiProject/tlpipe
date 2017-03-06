"""Calibration using a strong point source.

Inheritance diagram
-------------------

.. inheritance-diagram:: PsCal
   :parts: 2

"""

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

        ts.redistribute('frequency')

        lfreq = ts.local_freq[:] # local freq

        feedno = ts['feedno'][:].tolist()
        pol = [ ts.pol_dict[p] for p in ts['pol'][:] ]
        bl = ts.bl[:]
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

        print transit_inds

        ### now only use the first transit point to do the cal
        ### may need to improve in the future
        transit_ind = transit_inds[0]
        int_time = ts.attrs['inttime'] # second
        start_ind = transit_ind - np.int(span / int_time)
        end_ind = transit_ind + np.int(span / int_time)

        nt = end_ind - start_ind
        nfeed = len(feedno)
        gain = np.empty((nt, nfeed, 2, len(lfreq)), dtype=np.complex128)
        gain[:] = complex(np.nan, np.nan)

        # construct visibility matrix for a single time, pol, freq
        Vmat = np.zeros((nfeed, nfeed), dtype=ts.vis.dtype)
        for ind, ti in enumerate(range(start_ind, end_ind)):
            # when noise on, just pass
            if 'ns_on' in ts.iterkeys() and ts['ns_on'][ti]:
                continue
            aa.set_jultime(ts['jul_date'][ti])
            s.compute(aa)
            # get fluxes vs. freq of the calibrator
            Sc = s.get_jys()
            # get the topocentric coordinate of the calibrator at the current time
            # s_top = s.get_crds('top', ncrd=3)
            aa.sim_cache(cat.get_crds('eq', ncrd=3)) # for compute bm_response and sim
            for pi in [pol.index('xx'), pol.index('yy')]: # xx, yy
                # aa.set_active_pol(pol[pi])
                for fi, freq in enumerate(lfreq): # mpi among freq
                    for i, ai in enumerate(feedno):
                        for j, aj in enumerate(feedno):
                            try:
                                bi = bls.index((ai, aj))
                                if ts.local_vis_mask[ti, fi, pi, bi]:
                                    Vmat[i, j] = 0
                                else:
                                    Vmat[i, j] = ts.local_vis[ti, fi, pi, bi] / Sc[fi] # xx, yy
                            except ValueError:
                                bi = bls.index((aj, ai))
                                if ts.local_vis_mask[ti, fi, pi, bi]:
                                    Vmat[i, j] = 0
                                else:
                                    Vmat[i, j] = np.conj(ts.local_vis[ti, fi, pi, bi] / Sc[fi]) # xx, yy

                    # Eigen decomposition
                    Vmat = np.where(np.isfinite(Vmat), Vmat, 0)
                    V0, S = rpca_decomp.decompose(Vmat, debug=True)

                    # plot
                    # plot Vmat
                    plt.figure(figsize=(13, 5))
                    plt.subplot(121)
                    plt.imshow(Vmat.real, aspect='equal', origin='lower')
                    plt.colorbar(shrink=1.0)
                    plt.subplot(122)
                    plt.imshow(Vmat.imag, aspect='equal', origin='lower')
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
                    plt.imshow(V0.real, aspect='equal', origin='lower')
                    plt.colorbar(shrink=1.0)
                    plt.subplot(122)
                    plt.imshow(V0.imag, aspect='equal', origin='lower')
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
                    plt.imshow(S.real, aspect='equal', origin='lower')
                    plt.colorbar(shrink=1.0)
                    plt.subplot(122)
                    plt.imshow(S.imag, aspect='equal', origin='lower')
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
                    plt.imshow(N.real, aspect='equal', origin='lower')
                    plt.colorbar(shrink=1.0)
                    plt.subplot(122)
                    plt.imshow(N.imag, aspect='equal', origin='lower')
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
                    gain[ind, :, pi, fi] = g

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

        if mpiutil.rank0:
            for idx, fd in enumerate(feedno):
                plt.figure()
                plt.plot(np.abs(gain[:, idx, 0, 0]), label='xx') # only plot fi == 0
                plt.plot(np.abs(gain[:, idx, 1, 0]), label='yy') # only plot fi == 0
                plt.legend()
                fig_name = '%s_%d.png' % (fig_prefix, fd)
                if tag_output_iter:
                    fig_name = output_path(fig_name, iteration=self.iteration)
                else:
                    fig_name = output_path(fig_name)
                plt.savefig(fig_name)
                plt.close()


        return super(PsCal, self).process(ts)
