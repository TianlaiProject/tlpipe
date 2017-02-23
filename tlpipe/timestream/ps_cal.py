"""Calibration using a strong point source.

Inheritance diagram
-------------------

.. inheritance-diagram:: PsCal
   :parts: 2

"""

import os
import numpy as np
from scipy.linalg import eigh
import h5py
import ephem
import aipy as a
import tod_task

from caput import mpiutil
from caput import mpiarray
from caput import memh5
from tlpipe.utils.path_util import output_path


def cal(vis, vis_mask, li, gi, pbl, ts, **kwargs):
    tgain = kwargs.get('tgain')
    pol = pbl[0]
    ai, aj = pbl[1]

    feedno = ts['feedno'][:].tolist()
    i = feedno.index(ai)
    j = feedno.index(aj)

    if pol == 'xx':
        pi = 0
    elif pol == 'yy':
        pi = 1
    else:
        return

    bl_gain = tgain[i, pi, :] * tgain[j, pi, :].conj()

    vis[:] = vis / bl_gain


class PsCal(tod_task.TaskTimestream):
    """Calibration using a strong point source.

    The calibration is done by using the Eigen-decomposition method.

    May be more explain to this...

    """

    params_init = {
                    'calibrator': 'cyg',
                    'catalog': 'misc,helm,nvss',
                    'span': 60, # second
                    'save_gain': False,
                    'gain_file': 'gain.hdf5',
                  }

    prefix = 'pc_'

    def process(self, ts):

        calibrator = self.params['calibrator']
        catalog = self.params['catalog']
        span = self.params['span']
        save_gain = self.params['save_gain']
        gain_file = self.params['gain_file']

        ts.redistribute('frequency')

        lfreq = ts.local_freq[:] # local freq

        feedno = ts['feedno'][:].tolist()
        pol = ts['pol'][:].tolist()
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
        eigval = np.empty((nt, nfeed, 2, len(lfreq)), dtype=np.float64)
        eigval[:] = np.nan
        gain = np.empty((nt, nfeed, 2, len(lfreq)), dtype=np.complex128)
        gain[:] = complex(np.nan, np.nan)

        # construct visibility matrix for a single time, pol, freq
        Vmat = np.zeros((nfeed, nfeed), dtype=ts.main_data.dtype)
        for ind, ti in enumerate(range(start_ind, end_ind)):
            # when noise on, just pass
            if 'ns_on' in ts.iterkeys() and ts['ns_on'][ti]:
                continue
            aa.set_jultime(ts['jul_date'][ti])
            s.compute(aa)
            # get fluxes vs. freq of the calibrator
            Sc = s.get_jys()
            # get the topocentric coordinate of the calibrator at the current time
            s_top = s.get_crds('top', ncrd=3)
            aa.sim_cache(cat.get_crds('eq', ncrd=3)) # for compute bm_response and sim
            for pi in [pol.index('xx'), pol.index('yy')]: # xx, yy
                aa.set_active_pol(pol[pi])
                for fi, freq in enumerate(lfreq): # mpi among freq
                    for i, ai in enumerate(feedno):
                        for j, aj in enumerate(feedno):
                            # uij = aa.gen_uvw(i, j, src='z').squeeze() # (rj - ri)/lambda
                            uij = aa.gen_uvw(i, j, src='z')[:, 0, :] # (rj - ri)/lambda
                            # bmij = aa.bm_response(i, j).squeeze() # will get error for only one local freq
                            # import pdb
                            # pdb.set_trace()
                            bmij = aa.bm_response(i, j).reshape(-1)
                            try:
                                bi = bls.index((ai, aj))
                                # Vmat[i, j] = ts.local_vis[ti, fi, pi, bi] / (Sc[fi] * bmij[fi] * np.exp(-2.0J * np.pi * np.dot(s_top, uij[:, fi]))) # xx, yy
                                Vmat[i, j] = ts.local_vis[ti, fi, pi, bi] / (Sc[fi] * bmij[fi] * np.exp(2.0J * np.pi * np.dot(s_top, uij[:, fi]))) # xx, yy
                            except ValueError:
                                bi = bls.index((aj, ai))
                                # Vmat[i, j] = np.conj(ts.local_vis[ti, fi, pi, bi] / (Sc[fi] * bmij[fi] * np.exp(-2.0J * np.pi * np.dot(s_top, uij[:, fi])))) # xx, yy
                                Vmat[i, j] = np.conj(ts.local_vis[ti, fi, pi, bi] / (Sc[fi] * bmij[fi] * np.exp(2.0J * np.pi * np.dot(s_top, uij[:, fi])))) # xx, yy

                    # Eigen decomposition

                    Vmat = np.where(np.isfinite(Vmat), Vmat, 0)

                    e, U = eigh(Vmat)
                    eigval[ind, :, pi, fi] = e[::-1] # descending order
                    # max eigen-val
                    lbd = e[-1] # lambda
                    # the gain vector for this freq
                    gvec = np.sqrt(lbd) * U[:, -1] # only eigen-vector corresponding to the maximum eigen-val
                    gain[ind, :, pi, fi] = gvec

        # apply gain to vis
        # get the time mean gain
        tgain = np.ma.mean(np.ma.masked_invalid(gain), axis=0) # time mean
        tgain = mpiutil.gather_array(tgain, axis=-1, root=None)

        ts.redistribute('baseline')
        ts.pol_and_bl_data_operate(cal, tgain=tgain)

        # save gain if required:
        if save_gain:
            gain_file = output_path(gain_file)
            eigval = mpiarray.MPIArray.wrap(eigval, axis=3)
            gain = mpiarray.MPIArray.wrap(gain, axis=3)
            mem_gain = memh5.MemGroup(distributed=True)
            mem_gain.create_dataset('eigval', data=eigval)
            mem_gain.create_dataset('gain', data=gain)
            # add attris
            mem_gain.attrs['jul_data'] = ts['jul_date'][start_ind:end_ind]
            mem_gain.attrs['feed'] = np.array(feedno)
            mem_gain.attrs['pol'] = np.array(['xx', 'yy'])
            mem_gain.attrs['freq'] = ts.freq[:] # freq should be common

            # save to file
            mem_gain.to_hdf5(gain_file, hints=False)

        ts.add_history(self.history)

        return ts
