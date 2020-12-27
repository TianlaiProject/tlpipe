"""Noise source calibration by using eigen-decomposition method.

Inheritance diagram
-------------------

.. inheritance-diagram:: NsCal
   :parts: 2

"""

import itertools
import time
import numpy as np
from scipy import linalg as la
from scipy.interpolate import InterpolatedUnivariateSpline
import h5py
from . import timestream_task
from tlpipe.container.timestream import Timestream

from caput import mpiutil
from tlpipe.utils.path_util import output_path
from tlpipe.utils import progress
from tlpipe.utils import rpca_decomp
# import tlpipe.plot
# import matplotlib.pyplot as plt


class NsCal(timestream_task.TimestreamTask):
    """Noise source calibration by using eigen-decomposition method.

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
                    'num_mean': 3, # use the mean of num_mean signals
                    'save_ns_vis': False, # save the extracted calibrator visibility
                    'ns_vis_file': 'ns_vis/ns_vis.hdf5',
                    'apply_gain': True,
                    'save_gain': False,
                    'gain_file': 'ns_gain/ns_gain.hdf5',
                  }

    prefix = 'nc_'

    def process(self, ts):

        assert isinstance(ts, Timestream), '%s only works for Timestream object' % self.__class__.__name__

        num_mean = self.params['num_mean']
        save_ns_vis = self.params['save_ns_vis']
        ns_vis_file = self.params['ns_vis_file']
        apply_gain = self.params['apply_gain']
        save_gain = self.params['save_gain']
        gain_file = self.params['gain_file']
        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']
        tag_output_iter = self.params['tag_output_iter']

        if save_ns_vis or apply_gain or save_gain:
            pol_type = ts['pol'].attrs['pol_type']
            if pol_type != 'linear':
                raise RuntimeError('Can not do ns_eigcal for pol_type: %s' % pol_type)

            ts.redistribute('baseline')

            nt = ts.local_vis.shape[0]
            freq = ts.freq[:]
            pol = [ ts.pol_dict[p] for p in ts['pol'][:] ] # as string
            bls = mpiutil.gather_array(ts.local_bl[:], root=None, comm=ts.comm)
            feedno = ts['feedno'][:].tolist()

            nf = ts.local_freq.shape[0]
            npol = ts.local_pol.shape[0]
            nlb = ts.local_bl.shape[0]
            nfeed = len(feedno)

            if num_mean <= 0:
                raise RuntimeError('Invalid num_mean = %s' % num_mean)
            ns_on = ts['ns_on'][:]
            ns_on = np.where(ns_on, 1, 0)
            diff_ns = np.diff(ns_on)
            on_si = np.where(diff_ns==1)[0] + 1 # start inds of ON
            on_ei = np.where(diff_ns==-1)[0] + 1 # (end inds + 1) of ON
            if on_ei[0] < on_si[0]:
                on_ei = on_ei[1:]
            if on_si[-1] > on_ei[-1]:
                on_si = on_si[:-1]

            if on_si[0] < num_mean+1: # not enough off data in the beginning to use
                on_si = on_si[1:]
                on_ei = on_ei[1:]

            if len(on_si) != len(on_ei):
                raise RuntimeError('len(on_si) != len(on_ei)')
            num_on = len(on_si)
            cal_inds = (on_si + on_ei) / 2 # cal inds are the center inds on ON


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


            tfp_inds = list(itertools.product(list(range(num_on)), list(range(nf)), list(range(npol))))
            ns, ss, es = mpiutil.split_all(len(tfp_inds), comm=ts.comm)
            # gather data to make each process to have its own data which has all bls
            for ri, (ni, si, ei) in enumerate(zip(ns, ss, es)):
                lon_off = np.zeros((ni, nlb), dtype=ts.vis.dtype)
                for ii, (ti, fi, pi) in enumerate(tfp_inds[si:ei]):
                    si_on, ei_on = on_si[ti], on_ei[ti]
                    # mean of ON - mean of OFF
                    if ei_on - si_on > 3:
                        # does not use the two ends if there are more than three ONs
                        lon_off[ii] = np.mean(ts.local_vis[si_on+1:ei_on-1, fi, pi], axis=0) - np.ma.mean(np.ma.array(ts.local_vis[si_on-num_mean-1:si_on-1, fi, pi], mask=ts.local_vis_mask[si_on-num_mean-1:si_on-1, fi, pi]), axis=0)
                    else:
                        lon_off[ii] = np.mean(ts.local_vis[si_on:ei_on, fi, pi], axis=0) - np.ma.mean(np.ma.array(ts.local_vis[si_on-num_mean-1:si_on-1, fi, pi], mask=ts.local_vis_mask[si_on-num_mean-1:si_on-1, fi, pi]), axis=0)

                # gather on_off from all process for separate bls
                on_off = mpiutil.gather_array(lon_off, axis=1, root=ri, comm=ts.comm)
                if ri == mpiutil.rank:
                    tfp_linds = tfp_inds[si:ei] # inds for this process
                    this_on_off = on_off
            del tfp_inds
            del lon_off
            tfp_len = len(tfp_linds)


            cnan = complex(np.nan, np.nan) # complex nan
            if save_ns_vis:
                # save the extracted noise source vis
                lsrc_vis = np.full((tfp_len, nfeed, nfeed), cnan, dtype=ts.vis.dtype)
                # save sky vis
                lsky_vis = np.full((tfp_len, nfeed, nfeed), cnan, dtype=ts.vis.dtype)
                # save outlier vis
                lotl_vis = np.full((tfp_len, nfeed, nfeed), cnan, dtype=ts.vis.dtype)

            if apply_gain or save_gain:
                lgain = np.zeros((tfp_len, nfeed), dtype=ts.vis.dtype)
                lgain_mask = np.zeros((tfp_len, nfeed), dtype=bool)

            # construct visibility matrix for a single time, freq, pol
            Vmat = np.full((nfeed, nfeed), cnan, dtype=ts.vis.dtype)

            if show_progress and mpiutil.rank0:
                pg = progress.Progress(len(tfp_linds), step=progress_step)

            for ii, (ti, fi, pi) in enumerate(tfp_linds):
                if show_progress and mpiutil.rank0:
                    pg.show(ii)

                Vmat.flat[mis] = this_on_off[ii]
                Vmat.flat[mis_conj] = this_on_off[ii, bis_conj].conj()

                if save_ns_vis:
                    lsky_vis[ii] = Vmat

                # initialize the outliers
                med = np.median(Vmat.real) + 1.0J * np.median(Vmat.imag)
                diff = Vmat - med
                S0 = np.where(np.abs(diff)>3.0*rpca_decomp.MAD(Vmat), diff, 0)
                # stable PCA decomposition
                V0, S = rpca_decomp.decompose(Vmat, rank=1, S=S0, max_iter=100, threshold='hard', tol=1.0e-6, debug=False)
                if save_ns_vis:
                    lsrc_vis[ii] = V0
                    lotl_vis[ii] = S

                if apply_gain or save_gain:
                    e, U = la.eigh(V0, eigvals=(nfeed-1, nfeed-1))
                    g = U[:, -1] * e[-1]**0.5
                    # g = U[:, -1] * nfeed**0.5 # to make g_i g_j^* ~ 1
                    if g[0].real < 0:
                        g *= -1.0 # make all g[0] phase 0, instead of pi
                    lgain[ii] = g
                    ### maybe does not flag abnormal values here to simplify the programming, the flag can be down in ps_cal
                    gabs = np.abs(g)
                    gmed = np.median(gabs)
                    gabs_diff = np.abs(gabs - gmed)
                    gmad = np.median(gabs_diff) / 0.6745
                    lgain_mask[ii, np.where(gabs_diff>3.0*gmad)[0]] = True # mask invalid feeds

            if save_ns_vis:
                if tag_output_iter:
                    ns_vis_file = output_path(ns_vis_file, iteration=self.iteration)
                else:
                    ns_vis_file = output_path(ns_vis_file)
                # create file and allocate space first by rank0
                if mpiutil.rank0:
                    with h5py.File(ns_vis_file, 'w') as f:
                        # allocate space
                        shp = (num_on, nf, npol, nfeed, nfeed)
                        f.create_dataset('sky_vis', shp, dtype=lsky_vis.dtype)
                        f.create_dataset('src_vis', shp, dtype=lsrc_vis.dtype)
                        f.create_dataset('outlier_vis', shp, dtype=lotl_vis.dtype)
                        f.attrs['dim'] = 'time, freq, pol, feed, feed'
                        try:
                            f.attrs['time_inds'] = (on_si + on_ei) / 2
                        except RuntimeError:
                            f.create_dataset('time_inds', data=(on_si + on_ei)/2)
                            f.attrs['time_inds'] = '/time_inds'
                        f.attrs['freq'] = ts.freq
                        f.attrs['pol'] = ts.pol
                        f.attrs['feed'] = np.array(feedno)

                mpiutil.barrier()

                # write data to file
                for i in range(10):
                    try:
                        # NOTE: if write simultaneously, will loss data with processes distributed in several nodes
                        for ri in range(mpiutil.size):
                            if ri == mpiutil.rank:
                                with h5py.File(ns_vis_file, 'r+') as f:
                                    for ii, (ti, fi, pi) in enumerate(tfp_linds):
                                        f['sky_vis'][ti, fi, pi] = lsky_vis[ii]
                                        f['src_vis'][ti, fi, pi] = lsrc_vis[ii]
                                        f['outlier_vis'][ti, fi, pi] = lotl_vis[ii]
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
                gain = mpiutil.gather_array(lgain, axis=0, root=None, comm=ts.comm)
                gain_mask = mpiutil.gather_array(lgain_mask, axis=0, root=None, comm=ts.comm)
                del lgain
                del lgain_mask
                gain = gain.reshape(num_on, nf, npol, nfeed)
                gain_mask = gain_mask.reshape(num_on, nf, npol, nfeed)

                # normalize gain to make its amp ~ 1
                gain_med = np.ma.median(np.ma.array(np.abs(gain), mask=gain_mask))
                gain /= gain_med

                # phi = np.angle(gain)

                # delta_phi = np.zeros((num_on, nf, npol))

                # # get phase change
                # for ti in range(1, num_on):
                #     delta_phi[ti] = np.ma.mean(np.ma.array(phi[ti], mask=gain_mask[ti]) - np.ma.array(phi[ti-1], mask=gain_mask[ti-1]), axis=2)

                # # save original gain
                # gain_original = gain.copy()
                # # compensate phase changes
                # gain *= np.exp(1.0J * delta_phi[:, :, :, np.newaxis])

                gain_alltimes = np.full((nt, nf, npol, nfeed), cnan, dtype=gain.dtype)
                gain_alltimes_mask = np.zeros((nt, nf, npol, nfeed), dtype=bool)

                # interpolate to all time points
                for fi in range(nf):
                    for pi in range(npol):
                        for di in range(nfeed):
                            valid_inds = np.where(np.logical_not(gain_mask[:, fi, pi, di]))[0]
                            if len(valid_inds) < 0.75 * num_on:
                                # no enough points to do good interpolation
                                gain_alltimes_mask[:, fi, pi, di] = True
                            else:
                                # gain_alltimes[:, fi, pi, di] = InterpolatedUnivariateSpline(cal_inds[valid_inds], gain[valid_inds, fi, pi, di].real)(np.arange(nt)) + 1.0J * InterpolatedUnivariateSpline(cal_inds[valid_inds], gain[valid_inds, fi, pi, di].imag)(np.arange(nt))
                                # interpolate amp and phase to avoid abrupt changes
                                amp = InterpolatedUnivariateSpline(cal_inds[valid_inds], np.abs(gain[valid_inds, fi, pi, di]))(np.arange(nt))
                                phs = InterpolatedUnivariateSpline(cal_inds[valid_inds], np.unwrap(np.angle(gain[valid_inds, fi, pi, di])))(np.arange(nt))
                                gain_alltimes[:, fi, pi, di] = amp * np.exp(1.0J * phs)

                # apply gain to vis
                for fi in range(nf):
                    for pi in range(npol):
                        for bi, (fd1, fd2) in enumerate(ts['blorder'].local_data):
                            g1 = gain_alltimes[:, fi, pi, feedno.index(fd1)]
                            g1_mask = gain_alltimes_mask[:, fi, pi, feedno.index(fd1)]
                            g2 = gain_alltimes[:, fi, pi, feedno.index(fd2)]
                            g2_mask = gain_alltimes_mask[:, fi, pi, feedno.index(fd2)]
                            g12 = g1 * np.conj(g2)
                            g12_mask = np.logical_or(g1_mask, g2_mask)

                            if fd1 == fd2:
                                # auto-correlation should be real
                                ts.local_vis[:, fi, pi, bi] /= g12.real
                            else:
                                ts.local_vis[:, fi, pi, bi] /= g12
                            ts.local_vis_mask[:, fi, pi, bi] = np.logical_or(ts.local_vis_mask[:, fi, pi, bi], g12_mask)


                if save_gain:
                    if tag_output_iter:
                        gain_file = output_path(gain_file, iteration=self.iteration)
                    else:
                        gain_file = output_path(gain_file)
                    if mpiutil.rank0:
                        with h5py.File(gain_file, 'w') as f:
                            # allocate space for Gain
                            # dset = f.create_dataset('gain', data=gain_original) # gain without phase compensation
                            dset = f.create_dataset('gain', data=gain) # gain without phase compensation
                            # f.create_dataset('delta_phi', data=delta_phi)
                            f.create_dataset('gain_mask', data=gain_mask)
                            dset.attrs['dim'] = 'time, freq, pol, feed'
                            try:
                                dset.attrs['time_inds'] = cal_inds
                            except RuntimeError:
                                f.create_dataset('time_inds', data=cal_inds)
                                dset.attrs['time_inds'] = '/time_inds'
                            dset.attrs['freq'] = ts.freq
                            dset.attrs['pol'] = ts.pol
                            dset.attrs['feed'] = np.array(feedno)
                            dset.attrs['gain_med'] = gain_med # record the normalization factor
                            # save gain_alltimes
                            dset = f.create_dataset('gain_alltimes', data=gain_alltimes)
                            f.create_dataset('gain_alltimes_mask', data=gain_alltimes_mask)
                            dset.attrs['dim'] = 'time, freq, pol, feed'
                            try:
                                dset.attrs['time_inds'] = np.arange(nt)
                            except RuntimeError:
                                f.create_dataset('all_time_inds', data=np.arange(nt))
                                dset.attrs['time_inds'] = '/all_time_inds'
                            dset.attrs['freq'] = ts.freq
                            dset.attrs['pol'] = ts.pol
                            dset.attrs['feed'] = np.array(feedno)

                            f.create_dataset('time', data=ts.local_time)

                    mpiutil.barrier()


        return super(NsCal, self).process(ts)
