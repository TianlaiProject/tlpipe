"""Calibration using a strong point source.

Inheritance diagram
-------------------

.. inheritance-diagram:: PsCal
   :parts: 2

"""

import re
from datetime import datetime, timedelta, timezone
import numpy as np
from scipy import linalg as la
import h5py
from . import timestream_task
from tlpipe.container.timestream import Timestream
from tlpipe.core import constants as const

from caput import mpiutil
from tlpipe.utils.path_util import output_path
from tlpipe.utils import rpca_decomp
from tlpipe.cal import calibrators


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
                    'calibrator': 'cas',
                    'catalog': 'misc', # or helm,nvss
                    'calibrator_index': 0,
                    'vis_conj': False, # if True, conjugate the vis first
                    'zero_diag': False, # if True, fill 0 to the diagonal of vis matrix before SPCA
                    'span': 10, # time points
                    'reserve_high_gain': False, # if True, will not flag those gain significantly higher than mean value, only flag significantly lower ones
                    'rpca_max_iter': 200, # max iteration number for rpca decomposition
                    'subtract_src': False, # subtract vis of the calibrator from data
                    'replace_with_src': False, # replace vis with the subtracted src_vis, only work when subtract_src = True
                    'apply_gain': True,
                    'save_gain': False,
                    'check_gain': False,
                    'gain_file': 'gain/gain.hdf5',
                  }

    prefix = 'pc_'

    def process(self, ts):

        assert isinstance(ts, Timestream), '%s only works for Timestream object' % self.__class__.__name__

        calibrator = self.params['calibrator']
        catalog = self.params['catalog']
        calibrator_index = self.params['calibrator_index']
        vis_conj = self.params['vis_conj']
        zero_diag = self.params['zero_diag']
        span = self.params['span']
        rpca_max_iter = self.params['rpca_max_iter']
        reserve_high_gain = self.params['reserve_high_gain']
        tag_output_iter = self.params['tag_output_iter']
        subtract_src = self.params['subtract_src']
        replace_with_src = self.params['replace_with_src']
        apply_gain = self.params['apply_gain']
        save_gain = self.params['save_gain']
        check_gain = self.params['check_gain']
        gain_file = self.params['gain_file']
        via_memmap = self.params['via_memmap']

        pol_type = ts['pol'].attrs['pol_type']
        if pol_type != 'linear':
            raise RuntimeError('Can not do ps_cal for pol_type: %s' % pol_type)


        ts.redistribute('time', via_memmap=via_memmap)

        # gather time to rank0
        sec1970 = mpiutil.gather_array(ts['sec1970'].local_data, axis=0, root=None, comm=ts.comm)
        jul_date = mpiutil.gather_array(ts.local_time, axis=0, root=None, comm=ts.comm)
        time = jul_date
        nt = len(time)
        feedno = ts['feedno'][:].tolist()
        pol = [ ts.pol_dict[p] for p in ts['pol'][:] ] # as string
        gain_pd = {'xx': 0, 'yy': 1,    0: 'xx', 1: 'yy'} # for gain related op
        bls = ts.bl[:]

        # get the calibrator
        try:
            s = calibrators.get_src(calibrator)
        except KeyError:
            if mpiutil.rank0:
                print('Calibrator %s is unavailable, available calibrators are:')
                for key, d in calibrators.src_data.items():
                    print('%8s  ->  %12s' % (key, d[0]))
            raise RuntimeError('Calibrator %s is unavailable')
        if mpiutil.rank0:
            print('Try to calibrate with %s...' % s.src_name)

        # get transit time of calibrator
        aa = ts.array # array

        transit_sec1970 = ts['transitsource'][calibrator_index][0]
        # get time zone
        pattern = '[-+]?\d+'
        try:
            tz = re.search(pattern, ts.attrs['timezone'].decode('ascii')).group() # ts.attrs['timezone'] is bytes in python3
        except AttributeError:
            tz = re.search(pattern, ts.attrs['timezone']).group() # ts.attrs['timezone'] is str in python3.10
        tz = int(tz)
        transit_local_time = datetime.fromtimestamp(transit_sec1970, tz=timezone(timedelta(hours=tz))).isoformat()
        transit_ind = np.searchsorted(sec1970, transit_sec1970)
        if transit_ind <= 0 or transit_ind >= nt:
            raise RuntimeError('Data does not contain local transit time %s of source %s' % (transit_local_time, calibrator))

        transit_time = time[transit_ind] # jul date
        aa.set_jultime(transit_time)
        # make all antennas point to the pointing direction
        az = np.radians(ts['transitsource'][calibrator_index][3])
        alt = np.radians(ts['transitsource'][calibrator_index][4])
        for ai in aa:
            ai.set_pointing(az=az, alt=alt, twist=0)


        # int_time = ts.attrs['inttime'] # second
        start_ind = transit_ind - span
        end_ind = transit_ind + span + 1 # plus 1 to make transit_ind is at the center

        start_ind = max(0, start_ind)
        end_ind = min(end_ind, ts.vis.shape[0])

        local_time_offset = ts.vis.local_offset[0]
        local_time_shape = ts.vis.local_shape[0]
        local_start_ind = max(start_ind, local_time_offset)
        local_end_ind = min(end_ind, local_time_offset + local_time_shape)
        lnt = max(0, local_end_ind - local_start_ind)

        if vis_conj:
            ts.local_vis[:] = ts.local_vis.conj()

        nt = end_ind - start_ind
        freq = ts.freq[:] # MHz
        nf = len(freq)
        nbl = len(bls)
        nfeed = len(feedno)

        # # lotl_mask = np.zeros((tfp_len, nfeed, nfeed), dtype=bool)
        cnan = complex(np.nan, np.nan) # complex nan
        if apply_gain or save_gain:
            lGain = np.full((lnt, nf, 2, nfeed), cnan, dtype=ts.vis.dtype)

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
        for ti in range(local_time_shape):
            if not subtract_src and ti + local_time_offset < start_ind:
                continue
            if not subtract_src and ti + local_time_offset >= end_ind:
                continue

            # when noise on, just pass
            if 'ns_on' in ts.keys() and ts['ns_on'].local_data[ti]:
                continue

            for fi in range(nf):
                for pi, pol_str in enumerate(['xx', 'yy']):
                    Vmat.flat[mis] = np.ma.array(ts.local_vis[ti, fi, pol.index(pol_str)], mask=ts.local_vis_mask[ti, fi, pol.index(pol_str)]).filled(cnan)
                    Vmat.flat[mis_conj] = np.ma.array(ts.local_vis[ti, fi, pol.index(pol_str), bis_conj], mask=ts.local_vis_mask[ti, fi, pol.index(pol_str), bis_conj]).conj().filled(cnan)

                    # set invalid val to 0
                    invalid = ~np.isfinite(Vmat) # a bool array
                    # if too many masks
                    if np.where(invalid)[0].shape[0] > 0.5 * nfeed**2:
                        continue
                    Vmat[invalid] = 0
                    # if all are zeros
                    if np.allclose(Vmat, 0.0):
                        continue

                    # fill diagonal of Vmat to 0
                    if zero_diag:
                        np.fill_diagonal(Vmat, 0)

                    # initialize the outliers
                    med = np.median(Vmat.real) + 1.0J * np.median(Vmat.imag)
                    diff = Vmat - med
                    S0 = np.where(np.abs(diff)>3.0*rpca_decomp.MAD(Vmat), diff, 0)
                    # stable PCA decomposition
                    V0, S = rpca_decomp.decompose(Vmat, rank=1, S=S0, max_iter=rpca_max_iter, threshold='hard', tol=1.0e-6, debug=False)

                    if subtract_src:
                        V0_copy = V0.copy()
                        # make imag part of auto-correlation to be 0
                        V0_copy[np.diag_indices(V0.shape[0])] = V0_copy[np.diag_indices(V0.shape[0])].real + 0j
                        if replace_with_src:
                            ts.local_vis[ti, fi, pol.index(pol_str)] = V0_copy.flat[mis]
                        else:
                            ts.local_vis[ti, fi, pol.index(pol_str)] -= V0_copy.flat[mis]

                    if (start_ind <= ti + local_time_offset < end_ind) and (apply_gain or save_gain):
                        # use v_ij = gi gj^* \int Ai Aj^* e^(2\pi i n \cdot uij) T(x) d^2n
                        # precisely, we shold have
                        # V0 = (lambda^2 * Sc / (2 k_B)) * gi gj^* Ai Aj^* e^(2\pi i n0 \cdot uij)
                        e, U = la.eigh(V0, eigvals=(nfeed-1, nfeed-1))
                        g = U[:, -1] * e[-1]**0.5 # = \sqrt(lambda^2 * Sc / (2 k_B)) * gi Ai * e^(2\pi i n0 \cdot ui)
                        if g[0].real < 0:
                            g *= -1.0 # make all g[0] phase 0, instead of pi
                        ii = ti + local_time_offset - local_start_ind
                        lGain[ii, fi, pi] = g


        if apply_gain or save_gain:
            # gather lGain to rank0
            Gain = mpiutil.gather_array(lGain, axis=0, root=0, comm=ts.comm)
            gain = None
            is_conj = False
            if mpiutil.rank0:
                G_abs = np.full_like(Gain, np.nan, dtype=Gain.real.dtype)
                for ti in range(nt):
                    for fi in range(nf):
                        for pi in range(2):
                            this_Gain = Gain[ti, fi, pi]
                            valid_inds = np.where(np.isfinite(this_Gain))[0]
                            if len(valid_inds) > 3:
                                vabs = np.abs(this_Gain[valid_inds])
                                vmed = np.median(vabs)
                                vabs_diff = np.abs(vabs - vmed)
                                vmad = np.median(vabs_diff) / 0.6745
                                if reserve_high_gain:
                                    # reserve significantly higher ones, flag only significantly lower ones
                                    G_abs[ti, fi, pi, valid_inds] = np.where(vmed-vabs>3.0*vmad, np.nan, vabs)
                                else:
                                    # flag both significantly higher and lower ones
                                    G_abs[ti, fi, pi, valid_inds] = np.where(vabs_diff>3.0*vmad, np.nan, vabs)

                # compute s_top for this time range
                n0 = np.zeros((nt, 3))
                for ti, jt in enumerate(time[start_ind:end_ind]):
                    aa.set_jultime(jt)
                    s.compute(aa)
                    n0[ti] = s.get_crds('top', ncrd=3)

                # get the positions of feeds
                feedpos = ts['feedpos'][:]

                # create data to save the solved gain for each feed
                gain = np.full((nf, 2, nfeed), cnan, dtype=Gain.dtype) # gain for each feed

                # check for conj
                num_conj = 0
                for fi in range(nf):
                    for pi in range(2):
                        for di in range(nfeed):
                            y = G_abs[:, fi, pi, di]
                            inds = np.where(np.isfinite(y))[0]
                            if len(inds) >= max(4, 0.5 * len(y)):
                                # get the approximate magnitude by averaging the central G_abs
                                # solve phase by least square fit
                                ui = (feedpos[di] - feedpos[0]) * (1.0e6*freq[fi]) / const.c # position of this feed (relative to the first feed) in unit of wavelength
                                exp_factor = np.exp(2.0J * np.pi * np.dot(n0, ui))
                                ef = exp_factor
                                Gi = Gain[:, fi, pi, di]
                                e_phs = np.dot(ef[inds].conj(), Gi[inds]/y[inds]) / len(inds)
                                ea = np.abs(e_phs)
                                e_phs_conj = np.dot(ef[inds], Gi[inds]/y[inds]) / len(inds)
                                eac = np.abs(e_phs_conj)
                                if eac > ea:
                                    num_conj += 1

                if num_conj > 0.5 * (nf * 2 * nfeed): # 2 for 2 pols
                    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                    print('!!!   Detect data should be their conjugate...   !!!')
                    print('!!!   Correct it automatically...                !!!')
                    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

                    is_conj = True
                    # correct G
                    Gain = Gain.conj()

                Sc = s.get_jys(1.0e-3 * freq)
                lmd = const.c / (1.0e6*freq)
                # solve for gain
                for fi in range(nf):
                    for pi in range(2):
                        for di in range(nfeed):
                            ui = (feedpos[di] - feedpos[0]) * (1.0e6*freq[fi]) / const.c # position of this feed (relative to the first feed) in unit of wavelength
                            y = G_abs[:, fi, pi, di]
                            inds = np.where(np.isfinite(y))[0]
                            if len(inds) >= max(4, 0.5 * len(y)):
                                # get the approximate magnitude by averaging the central G_abs
                                mag = np.mean(y[inds]) # = \sqrt(lambda^2 * Sc / (2 k_B)) * |gi| Ai
                                # solve phase by least square fit
                                Gi = Gain[:, fi, pi, di]
                                exp_factor = np.exp(2.0J * np.pi * np.dot(n0, ui))
                                ef = exp_factor
                                e_phs = np.dot(ef[inds].conj(), Gi[inds]/y[inds]) / len(inds) # the phase of gi
                                ea = np.abs(e_phs)
                                if np.abs(ea - 1.0) < 0.1:
                                    # compute gain for this feed
                                    gain[fi, pi, di] = mag * e_phs # \sqrt(lambda^2 * Sc / (2 k_B)) * gi Ai
                                else:
                                    e_phs_conj = np.dot(ef[inds], Gi[inds]/y[inds]) / len(inds)
                                    eac = np.abs(e_phs_conj)
                                    if eac > ea:
                                        if np.abs(eac - 1.0) < 0.01:
                                            print('feedno = %d, fi = %d, pol = %s: may need to be conjugated' % (feedno[di], fi, gain_pd[pi]))
                                    else:
                                        print('feedno = %d, fi = %d, pol = %s: maybe wrong abs(e_phs): %s' % (feedno[di], fi, gain_pd[pi], ea))


                # normalize to get the exact gain
                # Omega = aa.ants[0].beam.Omega ### TODO: implement Omega for dish
                # Ai = aa.ants[0].beam.response(n0[transit_ind - start_ind]) # no pointing set, use the following instead
                Ai = aa[0].bm_response(n0[transit_ind - start_ind])[:, 0] # Ai at transit time
                factor = np.sqrt((lmd**2 * 1.0e-26 * Sc) / (2 * const.k_B)) * Ai # NOTE: 1Jy = 1.0e-26 W m^-2 Hz^-1
                gain /= factor[:, np.newaxis, np.newaxis]

                if check_gain:
                    if nf > 6:
                        for pi in range(2):
                            g = np.ma.masked_invalid(np.abs(gain[:, pi, :])).mean(axis=1).filled(np.nan)

                            diffs = []

                            for i in range(2, len(g)):
                                d = g[i] - g[i-1]
                                if i < 6:
                                    if np.isfinite(d):
                                        diffs.append(d)
                                else:
                                    m = np.mean(diffs)
                                    s = np.std(diffs)
                                    if np.isfinite(g[i]):
                                        for j in range(1, i):
                                            if not np.isfinite(g[i-j]):
                                                continue
                                            else:
                                                d = g[i] - g[i-j]

                                                # print(m, s, np.abs(d - m))
                                                if np.abs(d - m) > 7.0 * s:
                                                    g[i] = np.nan
                                                else:
                                                    diffs.append(d)
                                                break

                            oinds = np.where(~np.isfinite(g))[0]
                            gain[oinds, pi, :] = cnan


                # save gain to file
                if save_gain:
                    if tag_output_iter:
                        gain_file = output_path(gain_file, iteration=self.iteration)
                    else:
                        gain_file = output_path(gain_file)
                    with h5py.File(gain_file, 'w') as f:
                        # save Gain
                        dset = f.create_dataset('Gain', data=Gain)
                        dset.attrs['calibrator'] = calibrator
                        dset.attrs['dim'] = 'time, freq, pol, feed'
                        try:
                            dset.attrs['time'] = time[start_ind:end_ind]
                        except RuntimeError:
                            f.create_dataset('time', data=time[start_ind:end_ind])
                            dset.attrs['time'] = '/time'
                        dset.attrs['freq'] = freq
                        # dset.attrs['pol'] = np.array(['xx', 'yy'])
                        dset.attrs['pol'] = np.string_(['xx', 'yy']) # np.string_ for python 3
                        dset.attrs['feed'] = np.array(feedno)
                        dset.attrs['transit_ind'] = transit_ind
                        # save gain
                        dset = f.create_dataset('gain', data=gain)
                        dset.attrs['calibrator'] = calibrator
                        dset.attrs['dim'] = 'freq, pol, feed'
                        dset.attrs['freq'] = freq
                        # dset.attrs['pol'] = np.array(['xx', 'yy'])
                        dset.attrs['pol'] = np.string_(['xx', 'yy']) # np.string_ for python 3
                        dset.attrs['feed'] = np.array(feedno)

            if apply_gain:
                is_conj = mpiutil.bcast(is_conj, root=0, comm=ts.comm)
                if is_conj:
                    # correct vis
                    ts.local_vis[:] = ts.local_vis.conj()
                gain = mpiutil.bcast(gain, root=0, comm=ts.comm)
                for fi in range(nf):
                    for pi in [pol.index('xx'), pol.index('yy')]:
                        pi_ = gain_pd[pol[pi]]
                        for bi, (fd1, fd2) in enumerate(ts['blorder'].local_data):
                            g1 = gain[fi, pi_, feedno.index(fd1)]
                            g2 = gain[fi, pi_, feedno.index(fd2)]
                            if np.isfinite(g1) and np.isfinite(g2):
                                if fd1 == fd2:
                                    # auto-correlation should be real
                                    ts.local_vis[:, fi, pi, bi] /= (g1 * np.conj(g2)).real
                                else:
                                    ts.local_vis[:, fi, pi, bi] /= (g1 * np.conj(g2))
                            else:
                                # mask the un-calibrated vis
                                ts.local_vis_mask[:, fi, pi, bi] = True

                # in unit K after the calibration
                ts.vis.attrs['unit'] = 'K'

                # save src transit_ind
                ts.vis.attrs['transit_ind'] = transit_ind


        return super(PsCal, self).process(ts)
