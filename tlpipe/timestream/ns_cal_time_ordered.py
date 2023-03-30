"""Relative phase calibration using the noise source signal.

Inheritance diagram
-------------------

.. inheritance-diagram:: NsCal
   :parts: 2

"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from caput import mpiutil
from caput import mpiarray
from . import timestream_task
from tlpipe.container.raw_timestream import RawTimestream
from tlpipe.container.timestream import Timestream
from tlpipe.utils import progress


class NsCal(timestream_task.TimestreamTask):
    """Relative phase calibration using the noise source signal.

    The noise source can be viewed as a near-field source, its visibility
    can be expressed as

    .. math:: V_{ij}^{\\text{ns}} = C \\cdot e^{i k (r_{i} - r_{j})}

    where :math:`C` is a real constant.

    .. math::

        V_{ij}^{\\text{on}} &= G_{ij} (V_{ij}^{\\text{sky}} + V_{ij}^{\\text{ns}} + n_{ij}) \\\\
        V_{ij}^{\\text{off}} &= G_{ij} (V_{ij}^{\\text{sky}} + n_{ij})

    where :math:`G_{ij}` is the gain of baseline :math:`i,j`.

    .. math::

        V_{ij}^{\\text{on}} - V_{ij}^{\\text{off}} &= G_{ij} V_{ij}^{\\text{ns}} \\\\
                                       &=|G_{ij}| e^{i k \\Delta L} C \\cdot e^{i k (r_{i} - r_{j})} \\\\
                                       & = C |G_{ij}| e^{i k (\\Delta L + (r_{i} - r_{j}))}

    where :math:`\\Delta L` is the equivalent cable length.

    .. math:: \\text{Arg}(V_{ij}^{\\text{on}} - V_{ij}^{\\text{off}}) = k (\\Delta L + (r_{i} - r_{j})) = k \\Delta L + const.

    To compensate for the relative phase change (due to :math:`\\Delta L`) of the
    visibility, we can do

    .. math:: V_{ij}^{\\text{rel-cal}} = e^{-i \\; \\text{Arg}(V_{ij}^{\\text{on}} - V_{ij}^{\\text{off}})} \\, V_{ij}

    .. note::
        Note there is still an unknown (constant) phase factor to be determined in
        :math:`V_{ij}^{\\text{rel-cal}}`, which may be done by absolute calibration.

    """

    params_init = {
                    'num_mean': 5, # use the mean of num_mean signals
                    'unmasked_only': False, # cal for unmasked time points only
                    'phs_only': True, # phase cal only
                    'normalize_amp': True, # amplitue normalization
                  }

    prefix = 'nc_'

    def process(self, rt):

        # assert isinstance(rt, RawTimestream), '%s only works for RawTimestream object currently' % self.__class__.__name__

        if not 'ns_on' in rt.keys():
            raise RuntimeError('No noise source info, can not do noise source calibration')

        num_mean = self.params['num_mean']
        phs_only = self.params['phs_only']
        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']
        via_memmap = self.params['via_memmap']

        rt.redistribute('time', via_memmap=via_memmap)

        if isinstance(rt, RawTimestream):
            is_ts = False
            redistribute_axis = 2
        elif isinstance(rt, Timestream):
            is_ts = True
            redistribute_axis = 3

        nt = rt.vis.shape[0] # global number of time points
        if num_mean <= 0:
            raise RuntimeError('Invalid num_mean = %s' % num_mean)
        ns_on = mpiutil.gather_array(rt['ns_on'].local_data, root=None, comm=rt.comm)
        ns_on = np.where(ns_on, 1, 0)
        diff_ns = np.diff(ns_on)
        inds = np.where(diff_ns==1)[0] # NOTE: these are inds just 1 before the first ON
        if inds[0]-1 < 0: # no off data in the beginning to use
            inds = inds[1:]
        if inds[-1]+2 > nt-1: # no on data in the end to use
            inds = inds[:-1]

        nbl = len(rt.bl) # number of baselines
        n, r = nbl // mpiutil.size, nbl % mpiutil.size # number of iterations
        if r != 0:
            n = n + 1
        if show_progress and mpiutil.rank0:
            pg = progress.Progress(n, step=progress_step)
        for i in range(n):
            if show_progress and mpiutil.rank0:
                pg.show(i)

            this_vis = rt.local_vis[..., i*mpiutil.size:(i+1)*mpiutil.size].copy()
            this_vis = mpiarray.MPIArray.wrap(this_vis, axis=0, comm=rt.comm).redistribute(axis=redistribute_axis)
            this_vis_mask = rt.local_vis_mask[..., i*mpiutil.size:(i+1)*mpiutil.size].copy()
            this_vis_mask = mpiarray.MPIArray.wrap(this_vis_mask, axis=0, comm=rt.comm).redistribute(axis=redistribute_axis)
            if this_vis.local_array.shape[-1] != 0:
                if isinstance(rt, RawTimestream):
                    for fi in range(this_vis.local_array.shape[1]):
                        self.cal(this_vis.local_array[:, fi, 0], this_vis_mask.local_array[:, fi, 0], 0, 0, 0, rt, is_ts=is_ts, inds=inds, pol=rt['bl_pol'][i*mpiutil.size+mpiutil.rank], bl=rt['true_blorder'][i*mpiutil.size+mpiutil.rank])
                elif isinstance(rt, Timestream):
                    for fi in range(this_vis.local_array.shape[1]):
                        for pi in range(this_vis.local_array.shape[2]):
                            self.cal(this_vis.local_array[:, fi, pi, 0], this_vis_mask.local_array[:, fi, pi, 0], 0, 0, 0, rt, is_ts=is_ts, inds=inds, pol=pi, bl=rt['blorder'][i*mpiutil.size+mpiutil.rank])

            this_vis = this_vis.redistribute(axis=0)
            rt.local_vis[..., i*mpiutil.size:(i+1)*mpiutil.size] = this_vis.local_array
            this_vis_mask = this_vis_mask.redistribute(axis=0)
            rt.local_vis_mask[..., i*mpiutil.size:(i+1)*mpiutil.size] = this_vis_mask.local_array


        return super(NsCal, self).process(rt)

    def cal(self, vis, vis_mask, li, gi, fbl, rt, **kwargs):
        """Function that does the actual cal."""

        num_mean = self.params['num_mean']
        unmasked_only = self.params['unmasked_only']
        phs_only = self.params['phs_only']
        normalize_amp = self.params['normalize_amp']
        is_ts = kwargs['is_ts']
        inds = kwargs['inds']
        pol = kwargs['pol']
        bl = kwargs['bl']

        if np.prod(vis.shape) == 0 :
            return

        nt = vis.shape[0]
        on_time = rt['ns_on'].attrs['on_time']
        # off_time = rt['ns_on'].attrs['off_time']
        period = rt['ns_on'].attrs['period']

        # the calculated phase and amp will be at the ind just 1 before ns ON (i.e., at the ind of the last ns OFF)
        valid_inds = []
        phase = []
        if not phs_only:
            amp = []
        for ii, ind in enumerate(inds):
            # drop the first and the last ind, as it may lead to exceptional vals
            if ind == inds[0] or ind == inds[-1]:
                continue

            lower = ind - num_mean
            off_sec = np.ma.array(vis[lower:ind], mask=(~np.isfinite(vis[lower:ind]))|vis_mask[lower:ind])
            if off_sec.count() == 0: # all are invalid values
                continue
            if unmasked_only and off_sec.count() < max(2, num_mean/2): # more valid sample to make stable
                continue

            valid = True
            upper = ind + 1 + on_time
            off_mean = np.ma.mean(off_sec)
            this_on = np.ma.masked_invalid(vis[ind+1:upper]) # all on signal
            # just to avoid the case of all invalid on values
            if this_on.count() > 0:
                on_mean = np.ma.mean(this_on) # mean for all valid on signals
            else:
                continue
            diff = on_mean - off_mean
            phs = np.angle(diff) # in radians
            if not np.isfinite(phs):
                valid = False
            if not phs_only:
                amp_ = np.abs(diff)
                if not (np.isfinite(amp_) and amp_ > 1.0e-8): # amp_ should > 0
                    valid = False
            if not valid:
                continue
            valid_inds.append(ind)
            phase.append( phs ) # in radians
            if not phs_only:
                amp.append( amp_ )

        # not enough valid data to do the ns_cal
        num_valid = len(valid_inds)
        if num_valid <= 3:
            vis_mask[:] = True # mask the vis as no ns_cal has done
            return

        phase = np.unwrap(phase) # unwrap 2pi discontinuity
        if not phs_only:
            if normalize_amp:
                amp = np.array(amp) / np.median(amp) # normalize
        # split valid_inds into consecutive chunks
        intervals = [0] + (np.where(np.diff(valid_inds) > 5 * period)[0] + 1).tolist() + [num_valid]
        itp_inds = []
        itp_phase = []
        if not phs_only:
            itp_amp = []
        for i in range(len(intervals) -1):
            this_chunk = valid_inds[intervals[i]:intervals[i+1]]
            if len(this_chunk) > 3:
                itp_inds.append(this_chunk)
                itp_phase.append(phase[intervals[i]:intervals[i+1]])
                if not phs_only:
                    itp_amp.append(amp[intervals[i]:intervals[i+1]])

        # if no such chunk, mask all the data
        num_itp = len(itp_inds)
        if num_itp == 0:
            vis_mask[:] = True

        # get itp pairs
        itp_pairs = []
        for it in itp_inds:
            # itp_pairs.append((max(0, it[0]-off_time), min(nt, it[-1]+period)))
            itp_pairs.append((max(0, it[0]-5), min(nt, it[-1]+5))) # not to out interpolate two much, which may lead to very inaccurate values

        # get mask pairs
        mask_pairs = []
        for i in range(num_itp):
            if i == 0:
                mask_pairs.append((0, itp_pairs[i][0]))
            if i == num_itp - 1:
                mask_pairs.append((itp_pairs[i][-1], nt))
            else:
                mask_pairs.append((itp_pairs[i][-1], itp_pairs[i+1][0]))

        # set mask for inds in mask_pairs
        for mp1, mp2 in mask_pairs:
            vis_mask[mp1:mp2] = True

        # interpolate for inds in itp_inds
        all_phase = np.array([np.nan]*nt)
        for this_inds, this_phase, (i1, i2) in zip(itp_inds, itp_phase, itp_pairs):
            # no need to interpolate for auto-correlation
            if bl[0] == bl[1] and rt.pol_dict[pol] in ['xx', 'yy']:
                all_phase[i1:i2] = 0
            else:
                f = InterpolatedUnivariateSpline(this_inds, this_phase)
                this_itp_phs = f(np.arange(i1, i2))
                # # make the interpolated values in the appropriate range
                # this_itp_phs = np.where(this_itp_phs>np.pi, np.pi, this_itp_phs)
                # this_itp_phs = np.where(this_itp_phs<-np.pi, np.pi, this_itp_phs)
                all_phase[i1:i2] = this_itp_phs
                # do phase cal for this range of inds
                vis[i1:i2] = vis[i1:i2] * np.exp(-1.0J * this_itp_phs)

        if is_ts and 'transit_ind' in rt.vis.attrs:
            transit_ind = rt.vis.attrs['transit_ind']
            if np.isfinite(all_phase[transit_ind]):
                vis[:] = vis[:] * np.exp(1.0J * all_phase[transit_ind])
            else:
                vis_mask[:] = True

        if not phs_only:
            all_amp = np.array([np.nan]*nt)
            for this_inds, this_amp, (i1, i2) in zip(itp_inds, itp_amp, itp_pairs):
                f = InterpolatedUnivariateSpline(this_inds, this_amp)
                this_itp_amp = f(np.arange(i1, i2))
                all_amp[i1:i2] = this_itp_amp
                # do amp cal for this range of inds
                vis[i1:i2] = vis[i1:i2] / this_itp_amp

            if is_ts and 'transit_ind' in rt.vis.attrs:
                transit_ind = rt.vis.attrs['transit_ind']
                if np.isfinite(all_amp[transit_ind]):
                    vis[:] = vis[:] / all_amp[transit_ind]
                else:
                    vis_mask[:] = True
