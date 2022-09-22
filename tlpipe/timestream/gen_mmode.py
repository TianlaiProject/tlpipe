"""Generate m-mode by DFT of the timestream data.

Inheritance diagram
-------------------

.. inheritance-diagram:: GenMmode
   :parts: 2

"""

import os
import numpy as np
import h5py
from . import timestream_task
from tlpipe.container.timestream import Timestream
from tlpipe.core import constants as const

from caput import mpiutil
from tlpipe.utils.path_util import output_path
from tlpipe.map.drift.core import beamtransfer
from tlpipe.map.drift.pipeline import timestream


class GenMmode(timestream_task.TimestreamTask):
    """Generate m-mode by DFT of the timestream data.

    The generated m-mode can be used for map-making.

    """

    params_init = {
                    'tsys': 50.0,
                    'accuracy_boost': 1.0,
                    'l_boost': 1.0,
                    'bl_range': [0.0, 1.0e7],
                    'auto_correlations': False,
                    'lmax': None, # max l to compute
                    'mmax': None, # max m to compute
                    'pol': 'xx', # 'yy' or 'I'
                    'beam_dir': 'map/bt',
                    'noise_weight': True,
                    'ts_dir': 'map/ts',
                    'ts_name': 'ts',
                    'no_m_zero': True,
                  }

    prefix = 'gm_'

    def process(self, ts):

        tsys = self.params['tsys']
        accuracy_boost = self.params['accuracy_boost']
        l_boost = self.params['l_boost']
        bl_range = self.params['bl_range']
        auto_correlations = self.params['auto_correlations']
        lmax = self.params['lmax']
        mmax = self.params['mmax']
        pol = self.params['pol']
        beam_dir = output_path(self.params['beam_dir'])
        noise_weight = self.params['noise_weight']
        ts_dir = output_path(self.params['ts_dir'])
        ts_name = self.params['ts_name']
        no_m_zero = self.params['no_m_zero']


        assert isinstance(ts, Timestream), '%s only works for Timestream object' % self.__class__.__name__

        ts.redistribute('time')

        lat = ts.attrs['sitelat']
        # lon = ts.attrs['sitelon']
        lon = 0.0
        # lon = np.degrees(ts['ra_dec'][0, 0]) # the first ra
        local_origin = False
        freqs = ts.freq[:] # MHz
        nfreq = freqs.shape[0]
        band_width = ts.attrs['freqstep'] # MHz
        try:
            ndays = ts.attrs['ndays']
        except KeyError:
            ndays = 1
        feeds = ts['feedno'][:]
        bls = [ tuple(bl) for bl in ts.bl ]
        az, alt = ts['az_alt'].local_data[0] # assume fixed az, alt during the observation
        az = np.degrees(az)
        alt = np.degrees(alt)
        pointing = [az, alt, 0.0]
        feedpos = ts['feedpos'][:]

        if ts.is_dish:
            from tlpipe.map.drift.telescope import tl_dish

            dish_width = ts.attrs['dishdiam']
            tel = tl_dish.TlUnpolarisedDishArray(lat, lon, freqs, band_width, tsys, ndays, accuracy_boost, l_boost, bl_range, auto_correlations, local_origin, lmax, mmax, dish_width, feedpos, pointing)
        elif ts.is_cylinder:
            from tlpipe.map.drift.telescope import tl_cylinder

            # factor = 1.2 # suppose an illumination efficiency, keep same with that in timestream_common
            factor = 0.79 # for xx
            # factor = 0.88 # for yy
            cyl_width = factor * ts.attrs['cywid']
            tel = tl_cylinder.TlUnpolarisedCylinder(lat, lon, freqs, band_width, tsys, ndays, accuracy_boost, l_boost, bl_range, auto_correlations, local_origin, cyl_width, feedpos, lmax, mmax)
        else:
            raise RuntimeError('Unknown array type %s' % ts.attrs['telescope'])

        allpairs = tel.allpairs
        redundancy = tel.redundancy
        red_bin = np.cumsum(np.insert(redundancy, 0, 0)) # redundancy bin
        unqpairs = tel.uniquepairs
        nuq = len(unqpairs) # number of unique pairs

        # to save m-mode
        if mpiutil.rank0:
            # large array only in rank0 to save memory
            mmode = np.zeros((2*tel.mmax+1, nfreq, nuq), dtype=np.complex128)
            N = np.zeros((nfreq, nuq), dtype=np.int) # number of accumulate terms

        # mmode of a specific unique pair
        mmodeqi = np.zeros((2*tel.mmax+1, nfreq), dtype=np.complex128)
        Nqi = np.zeros((nfreq), dtype=np.int) # number of accumulate terms

        start_ra = ts.vis.attrs['start_ra']
        ra = mpiutil.gather_array(ts['ra_dec'].local_data[:, 0], root=None)
        ra = np.unwrap(ra)
        # find the first index that ra closest to start_ra
        ind = np.searchsorted(ra, start_ra)
        if np.abs(ra[ind] - start_ra) > np.abs(ra[ind+1] - start_ra):
            ind = ind + 1

        # get number of int_time in one sidereal day
        num_int = np.int(np.around(1.0 * const.sday / ts.attrs['inttime']))
        nt = ts.vis.shape[0]
        nt1 = min(num_int, nt-ind)

        inds = np.arange(nt)
        local_inds = mpiutil.scatter_array(inds, root=None)

        local_phi = ts['ra_dec'].local_data[:, 0]
        # the Fourier transfom matrix
        E = np.exp(-1.0J * np.outer(np.arange(-tel.mmax, tel.mmax+1), local_phi)) # e^(- i m phi)

        # pols to consider
        pol_str = [ ts.pol_dict[p] for p in ts['pol'][:] ] # as string
        if pol == 'xx' or pol == 'yy':
            pis = [ pol_str.index(pol) ]
        elif pol == 'I':
            pis = [ pol_str.index('xx'), pol_str.index('yy') ]
        else:
            raise ValueError('Invalid pol: %s' % pol)

        # compute mmodes for each unique pair
        for qi in range(nuq):
            mmodeqi[:] = 0
            Nqi[:] = 0
            this_pairs = allpairs[red_bin[qi]:red_bin[qi+1]]
            for a1, a2 in this_pairs:
                for pi in pis:
                    try:
                        b_ind = bls.index((feeds[a1], feeds[a2]))
                        V = ts.local_vis[:, :, pi, b_ind]
                    except ValueError:
                        b_ind = bls.index((feeds[a2], feeds[a1]))
                        V = ts.local_vis[:, :, pi, b_ind].conj()
                    M = ts.local_vis_mask[:, :, pi, b_ind] # mask
                    # mask time points that are outside of this day
                    M[local_inds<ind, :] = True
                    M[local_inds>=ind+nt1, :] = True
                    V = np.where(M, 0, V) # fill masked values with 0
                    v = np.logical_not(M).astype(np.int) # 1 for valid, 0 for invalid
                    # mmode[:, :, qi] += np.dot(E, V)
                    # N[:, qi] += np.sum(v, axis=0)
                    mmodeqi += np.dot(E, V)
                    Nqi += np.sum(v, axis=0)

            mpiutil.barrier()

            # accumulate mmode from all processes by Reduce
            if mpiutil.size > 1: # more than one processes
                if mpiutil.rank0:
                    # use IN_PLACE to reuse the mmode and N array
                    mpiutil.world.Reduce(mpiutil.IN_PLACE, mmodeqi, op=mpiutil.SUM, root=0)
                    mpiutil.world.Reduce(mpiutil.IN_PLACE, Nqi, op=mpiutil.SUM, root=0)
                else:
                    mpiutil.world.Reduce(mmodeqi, mmodeqi, op=mpiutil.SUM, root=0)
                    mpiutil.world.Reduce(Nqi, Nqi, op=mpiutil.SUM, root=0)

            if mpiutil.rank0:
                mmode[:, :, qi] = mmodeqi
                N[:, qi] = Nqi

        del ts
        del E

        # beamtransfer
        bt = beamtransfer.BeamTransfer(beam_dir, tel, noise_weight, True)
        # timestream
        tstream = timestream.Timestream(ts_dir, ts_name, bt, no_m_zero)

        if mpiutil.rank0:
            # reshape mmode toseparate positive and negative ms
            mmode1 = np.zeros((tel.mmax+1, nfreq, 2, nuq), dtype=mmode.dtype)
            mmode1[0, :, 0] = mmode[tel.mmax]
            for mi in range(1, tel.mmax+1):
                mmode1[mi, :, 0] = mmode[tel.mmax+mi]
                mmode1[mi, :, 1] = mmode[tel.mmax-mi].conj()

            del mmode

            # normalize mmode
            # mmode1 /= N[np.newaxis, :, np.newaxis, :]

            # save mmode to file
            mmode_dir = tstream.output_directory + '/mmodes'
            if os.path.exists(mmode_dir + '/COMPLETED_M'):
                # update the already existing mmodes
                for mi in range(tel.mmax+1):
                    with h5py.File(tstream._mfile(mi), 'r+') as f:
                        f['/mmode'][:] += mmode1[mi]
                with h5py.File(mmode_dir + '/count.hdf5', 'r+') as f:
                    f['count'][:] += N
            else:
                for mi in range(tel.mmax+1):
                    # make directory for each m-mode
                    if not os.path.exists(tstream._mdir(mi)):
                        os.makedirs(tstream._mdir(mi))

                    # create the m-file and save the result.
                    with h5py.File(tstream._mfile(mi), 'w') as f:
                        f.create_dataset('/mmode', data=mmode1[mi])
                        f.attrs['m'] = mi

                with h5py.File(mmode_dir + '/count.hdf5', 'w') as f:
                    f.create_dataset('count', data=N)

                # Make file marker that the m's have been correctly generated:
                open(mmode_dir + '/COMPLETED_M', 'a').close()

                # save the tstream object
                tstream.save()

        mpiutil.barrier()

        return tstream