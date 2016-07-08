"""Calibration using a strong point source."""

import os
import numpy as np
from scipy.linalg import eigh
import h5py
import aipy as a
import tod_task

from caput import mpiutil
from caput import mpiarray
from caput import memh5
from tlpipe.core import tldishes
from tlpipe.utils.path_util import input_path, output_path


def cal(vis, li, gi, pbl, ts, **kwargs):
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
        return vis

    bl_gain = tgain[i, pi, :] * tgain[j, pi, :].conj()

    return vis / bl_gain


class PsCal(tod_task.SingleTimestream):
    """Calibration using a strong point source."""

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

        if 'Dish' in ts.attrs['telescope']:
            ant_type = 'dish'
        elif 'Cylinder' in ts.attrs['telescope']:
            ant_type = 'cylinder'
        else:
            raise RuntimeError('Unknown antenna type %s' % ts.attrs['telescope'])

        ts.redistribute('frequency')

        lfreq = ts.freq.local_data[:] # local freq

        feedno = ts['feedno'][:].tolist()
        pol = ts['pol'][:].tolist()
        bl = ts.bl[:]
        bls = [ tuple(b) for b in bl ]
        # antpointing = np.radians(ts['antpointing'][-1, :, :]) # radians
        transitsource = ts['transitsource'][:]
        transit_time = transitsource[-1, 0] # second, sec1970
        int_time = ts.attrs['inttime'] # second

        transit_ind = np.searchsorted(ts['sec1970'][:], transit_time)
        print transit_ind
        start_ind = transit_ind - np.int(span / int_time)
        end_ind = transit_ind + np.int(span / int_time)

        # array
        if ant_type == 'dish':
            aa = tldishes.get_aa(1.0e-3 * lfreq) # use GHz
            # make all antennas point to the pointing direction
            for fd in feedno:
                # feedno start from 1
                # aa[fd-1].set_pointing(az=antpointing[fi, 0], alt=antpointing[fi, 1], twist=0)
                aa[fd-1].set_pointing(az=0, alt=np.pi/2, twist=0)
            # for ind, ai in enumerate(aa):
            #     if ind+1 in feedno: # feedno start from 1
            #         fi = feedno.index(ind+1)
            #         # ai.set_pointing(az=antpointing[fi, 0], alt=antpointing[fi, 1], twist=0)
            #         ai.set_pointing(az=0, alt=np.pi/2, twist=0)
        else:
            raise NotImplementedError('ps_cal for cylinder array not implemented yet')

        # calibrator
        srclist, cutoff, catalogs = a.scripting.parse_srcs(calibrator, catalog)
        cat = a.src.get_catalog(srclist, cutoff, catalogs)
        assert(len(cat) == 1), 'Allow only one calibrator'
        s = cat.values()[0]
        if mpiutil.rank0:
            print 'Calibrating for source %s with' % calibrator,
            print 'strength', s._jys,
            print 'measured at', s.mfreq, 'GHz',
            print 'with index', s.index

        nt = end_ind - start_ind
        nfeed = len(feedno)
        eigval = np.empty((nt, nfeed, 2, len(lfreq)), dtype=np.float64)
        eigval[:] = np.nan
        gain = np.empty((nt, nfeed, 2, len(lfreq)), dtype=np.complex128)
        gain[:] = complex(np.nan, np.nan)

        # construct visibility matrix for a single time, pol, freq
        Vmat = np.zeros((nfeed, nfeed), dtype=ts.main_data.dtype)
        for ind, ti in enumerate(range(start_ind, end_ind)):
            # when noise no, just pass
            if ts['ns_on'][ti]:
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
                            uij = aa.gen_uvw(ai-1, aj-1, src='z').squeeze() # (rj - ri)/lambda
                            # import pdb
                            # pdb.set_trace()
                            bmij = aa.bm_response(ai-1, aj-1).squeeze()
                            try:
                                bi = bls.index((ai, aj))
                                # Vmat[i, j] = ts.main_data.local_data[ti, fi, pi, bi] / (Sc[fi] * bmij[fi] * np.exp(-2.0J * np.pi * np.dot(s_top, uij[:, fi]))) # xx, yy
                                Vmat[i, j] = ts.main_data.local_data[ti, fi, pi, bi] / (Sc[fi] * bmij[fi] * np.exp(2.0J * np.pi * np.dot(s_top, uij[:, fi]))) # xx, yy
                            except ValueError:
                                bi = bls.index((aj, ai))
                                # Vmat[i, j] = np.conj(ts.main_data.local_data[ti, fi, pi, bi] / (Sc[fi] * bmij[fi] * np.exp(-2.0J * np.pi * np.dot(s_top, uij[:, fi])))) # xx, yy
                                Vmat[i, j] = np.conj(ts.main_data.local_data[ti, fi, pi, bi] / (Sc[fi] * bmij[fi] * np.exp(2.0J * np.pi * np.dot(s_top, uij[:, fi])))) # xx, yy

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
            gain_dir = os.path.dirname(gain_file)
            try:
                os.makedirs(gain_dir)
            except OSError:
                pass
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
