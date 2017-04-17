"""Bad/Exceptional visibility values detect.

Inheritance diagram
-------------------

.. inheritance-diagram:: Detect
   :parts: 2

"""

import numpy as np
import timestream_task
from tlpipe.container.raw_timestream import RawTimestream
from tlpipe.container.timestream import Timestream
from caput import mpiutil


class Detect(timestream_task.TimestreamTask):
    """Bad/Exceptional visibility values detect.

    This task does a simple bad/exceptional values detection by mask those
    values that are not finite, and those have non-zero imaginary part of an
    auto-correlation.

    """

    params_init = {}

    prefix = 'bd_'

    def process(self, ts):

        ts.redistribute('baseline')

        vis = ts.local_vis
        vis_mask = ts.local_vis_mask

        # mask non-finite vis values
        vis_mask[:] = np.where(np.isfinite(vis), vis_mask, True)

        # mask where the imaginary part of an auto-correlation is non-zero
        if isinstance(ts, RawTimestream):
            for bi, (fi, fj) in enumerate(ts.local_bl):
                if fi == fj:
                    vis_mask[..., bi] = np.where(vis[..., bi].imag == 0.0, vis_mask[..., bi], True)
        elif isinstance(ts, Timestream):
            for bi, (fi, fj) in enumerate(ts.local_bl):
                for pi, pol in enumerate(ts.local_pol):
                    pol = ts.pol_dict[pol]
                    if fi == fj and pol in ['xx', 'yy']:
                        vis_mask[..., pi, bi] = np.where(vis[..., pi, bi].imag == 0.0, vis_mask[..., pi, bi], True)

        # mask bl that have no signal
        problematic_bls = []
        bad_bls = []
        for bi in xrange(len(ts.local_bl)):
            bl = tuple(ts.local_bl[bi])
            if isinstance(ts, RawTimestream):
                # create a copy of vis for this bi, and fill 0 in masked positions
                vis1 = np.where(ts.local_vis_mask[..., bi], 0, ts.local_vis[..., bi])
                if np.allclose(vis1, 0): # all zeros
                    ts.local_vis_mask[..., bi] = True # mask all
                    bad_bls.append(bl)
                    problematic_bls.append(bl)
                elif np.sum(ts.local_vis_mask[..., bi]) >= 0.5 * np.prod(ts.local_vis[..., bi].shape):
                    problematic_bls.append(bl)
            elif isinstance(ts, Timestream):
                for pi, pol in enumerate(ts.local_pol):
                    pol = ts.pol_dict[pol]
                    # create a copy of vis for this bi, and fill 0 in masked positions
                    vis1 = np.where(ts.local_vis_mask[..., pi, bi], 0, ts.local_vis[..., pi, bi])
                    if np.allclose(vis1, 0): # all zeros
                        ts.local_vis_mask[..., pi, bi] = True # mask all
                        bad_bls.append((bl, pol))
                        problematic_bls.append((bl, pol))
                    elif np.sum(ts.local_vis_mask[..., pi, bi]) >= 0.5 * np.prod(ts.local_vis[..., pi, bi].shape):
                        problematic_bls.append((bl, pol))


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
            for ch, val in bad_chs.iteritems():
                if val > 0.5 * num_chs:
                    bchs.append(ch)
            # exclude those already in bchs
            for ch, val in problematic_chs.iteritems():
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
            for pol, d in bad_feeds.iteritems():
                for fd, val in d.iteritems():
                    if val > 0.5 * num_feeds:
                        bfeeds.append((fd, pol))
            # exclude those already in bfeeds
            for pol, d in problematic_feeds.iteritems():
                for fd, val in d.iteritems():
                    if val > 0.5 * num_feeds and not (fd, pol) in bfeeds:
                        pfeeds.append((fd, pol))

            # set mask for bad feeds and pol
            for bi, (fd1, fd2) in enumerate(ts.local_bl):
                for pi, pol in enumerate(ts.local_pol):
                    pol = ts.pol_dict[pol]
                    if (fd1, pol) in bfeeds or (fd2, pol) in bfeeds:
                        ts.local_vis_mask[..., pi, bi] = True


        if mpiutil.rank0:
            print 'Bad baseline: ', bad_bls
            print 'Problematic baseline: ', [ bl for bl in problematic_bls if not bl in bad_bls ]
            if isinstance(ts, RawTimestream):
                print 'Bad channels: ', bchs
                print 'Problematic channels: ', pchs
            elif isinstance(ts, Timestream):
                print 'Bad feeds: ', bfeeds
                print 'Problematic feeds: ', pfeeds

        return super(Detect, self).process(ts)
