"""Bad/Exceptional visibility values detect.

Inheritance diagram
-------------------

.. inheritance-diagram:: Detect
   :parts: 2

"""

import numpy as np
from . import timestream_task
from tlpipe.container.raw_timestream import RawTimestream
from tlpipe.container.timestream import Timestream
from tlpipe.utils import progress
from caput import mpiutil
from caput import mpiarray


class Detect(timestream_task.TimestreamTask):
    """Bad/Exceptional visibility values detect.

    This task does a simple bad/exceptional values detection by mask those
    values that are not finite, and those have non-zero imaginary part of an
    auto-correlation.

    """

    params_init = {}

    prefix = 'bd_'

    def process(self, ts):
        via_memmap = self.params['via_memmap']
        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']

        ts.redistribute('time', via_memmap=via_memmap)

        vis = ts.local_vis
        vis_mask = ts.local_vis_mask

        # mask non-finite vis values
        vis_mask[:] = np.where(np.isfinite(vis), vis_mask, True)

        # mask where the imaginary part of an auto-correlation is non-zero
        if isinstance(ts, RawTimestream):
            is_ts = False
            redistribute_axis = 2
            for bi, (fi, fj) in enumerate(ts.local_bl):
                if fi == fj:
                    vis_mask[..., bi] = np.where(vis[..., bi].imag == 0.0, vis_mask[..., bi], True)
        elif isinstance(ts, Timestream):
            is_ts = True
            redistribute_axis = 3
            for bi, (fi, fj) in enumerate(ts.local_bl):
                for pi, pol in enumerate(ts.local_pol):
                    pol = ts.pol_dict[pol]
                    if fi == fj and pol in ['xx', 'yy']:
                        vis_mask[..., pi, bi] = np.where(vis[..., pi, bi].imag == 0.0, vis_mask[..., pi, bi], True)

        # mask bl that have no signal
        problematic_bls = []
        bad_bls = []

        nbl = len(ts.bl) # number of baselines
        n, r = nbl // mpiutil.size, nbl % mpiutil.size # number of iterations
        if r != 0:
            n = n + 1
        if show_progress and mpiutil.rank0:
            pg = progress.Progress(n, step=progress_step)
        for i in range(n):
            if show_progress and mpiutil.rank0:
                pg.show(i)

            this_vis = ts.local_vis[..., i*mpiutil.size:(i+1)*mpiutil.size].copy()
            this_vis = mpiarray.MPIArray.wrap(this_vis, axis=0, comm=ts.comm).redistribute(axis=redistribute_axis)
            this_vis_mask = ts.local_vis_mask[..., i*mpiutil.size:(i+1)*mpiutil.size].copy()
            this_vis_mask = mpiarray.MPIArray.wrap(this_vis_mask, axis=0, comm=ts.comm).redistribute(axis=redistribute_axis)

            if this_vis.local_array.shape[-1] != 0:
                bi = i * mpiutil.size + mpiutil.rank
                bl = tuple(ts.bl[bi])

                if isinstance(ts, RawTimestream):
                    # create a copy of vis for this bi, and fill 0 in masked positions
                    vis1 = np.where(this_vis_mask.local_array[:, :, 0], 0, this_vis.local_array[:, :, 0])
                    if np.allclose(vis1, 0): # all zeros
                        this_vis_mask.local_array[:, :, 0] = True # mask all
                        bad_bls.append(bl)
                        problematic_bls.append(bl)
                    elif np.sum(this_vis_mask.local_array[:, :, 0]) >= 0.5 * np.prod(this_vis.local_array[:, :, 0].shape):
                        problematic_bls.append(bl)
                elif isinstance(ts, Timestream):
                    for pi, pol in enumerate(ts.local_pol):
                        pol = ts.pol_dict[pol]
                        # create a copy of vis for this bi, and fill 0 in masked positions
                        vis1 = np.where(this_vis_mask.local_array[:, :, pi, 0], 0, this_vis.local_array[:, :, pi, 0])
                        if np.allclose(vis1, 0): # all zeros
                            this_vis_mask.local_array[:, :, pi, 0] = True # mask all
                            bad_bls.append((bl, pol))
                            problematic_bls.append((bl, pol))
                        elif np.sum(this_vis_mask.local_array[:, :, pi, 0]) >= 0.5 * np.prod(this_vis.local_array[:, :, pi, 0].shape):
                            problematic_bls.append((bl, pol))

            this_vis_mask = this_vis_mask.redistribute(axis=0)
            ts.local_vis_mask[..., i*mpiutil.size:(i+1)*mpiutil.size] = this_vis_mask.local_array

        # gather list
        problematic_bls = mpiutil.gather_list(problematic_bls, comm=ts.comm)
        bad_bls = mpiutil.gather_list(bad_bls, comm=ts.comm)

        # try to find bad channels or feeds
        if isinstance(ts, RawTimestream):
            problematic_chs = {}
            bad_chs = {}
            for ch1, ch2 in problematic_bls:
                problematic_chs.setdefault(ch1, 0)
                problematic_chs.setdefault(ch2, 0)
                problematic_chs[ch1] += 1
                problematic_chs[ch2] += 1
            for ch1, ch2 in bad_bls:
                bad_chs.setdefault(ch1, 0)
                bad_chs.setdefault(ch2, 0)
                bad_chs[ch1] += 1
                bad_chs[ch2] += 1

            num_chs = len(ts['channo'][:].flatten())
            pchs = []
            bchs = []
            for ch, val in bad_chs.items():
                if val > 0.5 * num_chs:
                    bchs.append(ch)
            # exclude those already in bchs
            for ch, val in problematic_chs.items():
                if val > 0.5 * num_chs and not ch in bchs:
                    pchs.append(ch)

            # set mask for bad channels
            for bi, (ch1, ch2) in enumerate(ts.local_bl):
                if ch1 in bchs or ch2 in bchs:
                    ts.local_vis_mask[..., bi] = True

        elif isinstance(ts, Timestream):
            problematic_feeds = {}
            bad_feeds = {}
            for (fd1, fd2), pol in problematic_bls:
                problematic_feeds.setdefault(pol, {})
                problematic_feeds[pol].setdefault(fd1, 0)
                problematic_feeds[pol].setdefault(fd2, 0)
                problematic_feeds[pol][fd1] += 1
                problematic_feeds[pol][fd2] += 1
            for (fd1, fd2), pol in bad_bls:
                bad_feeds.setdefault(pol, {})
                bad_feeds[pol].setdefault(fd1, 0)
                bad_feeds[pol].setdefault(fd2, 0)
                bad_feeds[pol][fd1] += 1
                bad_feeds[pol][fd2] += 1

            num_feeds = len(ts['feedno'][:])
            pfeeds = []
            bfeeds = []
            for pol, d in bad_feeds.items():
                for fd, val in d.items():
                    if val > 0.5 * num_feeds:
                        bfeeds.append((fd, pol))
            # exclude those already in bfeeds
            for pol, d in problematic_feeds.items():
                for fd, val in d.items():
                    if val > 0.5 * num_feeds and not (fd, pol) in bfeeds:
                        pfeeds.append((fd, pol))

            # set mask for bad feeds and pol
            for bi, (fd1, fd2) in enumerate(ts.local_bl):
                for pi, pol in enumerate(ts.local_pol):
                    pol = ts.pol_dict[pol]
                    if (fd1, pol) in bfeeds or (fd2, pol) in bfeeds:
                        ts.local_vis_mask[..., pi, bi] = True


        if mpiutil.rank0:
            print('Bad baseline: ', bad_bls)
            print('Problematic baseline: ', [ bl for bl in problematic_bls if not bl in bad_bls ])
            if isinstance(ts, RawTimestream):
                print('Bad channels: ', bchs)
                print('Problematic channels: ', pchs)
            elif isinstance(ts, Timestream):
                print('Bad feeds: ', bfeeds)
                print('Problematic feeds: ', pfeeds)

        return super(Detect, self).process(ts)
