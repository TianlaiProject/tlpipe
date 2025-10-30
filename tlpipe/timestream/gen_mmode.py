"""Generate m-mode by DFT of the timestream data.

Inheritance diagram
-------------------

.. inheritance-diagram:: GenMmode
   :parts: 2

"""

import os
import shutil
import numpy as np
import h5py
from . import timestream_task
from tlpipe.container.timestream import Timestream
from tlpipe.core import constants as const

from caput import mpiutil
from caput import mpiarray
from tlpipe.utils.path_util import input_path
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
                    'use_beam': None, # use specified beam model, take hightes prioriety if given
                    'use_beam_freq_offset': 0, # freq offset of the used specified beam model
                    'use_fitted_beam_params': True,
                    'use_beam_params_in_file': False, # only when use_fitted_beam_params is False
                    'beam_params_file': 'beam_params.hdf5',
                    'use_feedpos_in_file': True,
                    'bl_range': [0.0, 1.0e7],
                    'auto_correlations': False,
                    'lmax': None, # max l to compute
                    'mmax': None, # max m to compute
                    'pol': 'xx', # or 'yy'
                    'beam_dir': 'map/bt',
                    'noise_weight': True,
                    'skip_svd': True, # set to False if do KL transform
                    'ts_dir': 'map/ts',
                    'ts_name': 'ts',
                    'no_m_zero': True,
                    'backup_ts': False, # backup timestream at each iteration
                    'keep_ts_bk_num': 1, # only keep this number of recent ts backup and remove all old ones, only work when backup_ts is True
                  }

    prefix = 'gm_'

    def process(self, ts):

        via_memmap = self.params['via_memmap']
        tsys = self.params['tsys']
        accuracy_boost = self.params['accuracy_boost']
        l_boost = self.params['l_boost']
        use_beam = self.params['use_beam']
        use_beam_freq_offset = self.params['use_beam_freq_offset']
        use_fitted_beam_params = self.params['use_fitted_beam_params']
        use_beam_params_in_file = self.params['use_beam_params_in_file']
        beam_params_file = self.params['beam_params_file']
        use_feedpos_in_file = self.params['use_feedpos_in_file']
        bl_range = self.params['bl_range']
        auto_correlations = self.params['auto_correlations']
        lmax = self.params['lmax']
        mmax = self.params['mmax']
        pol = self.params['pol']
        beam_dir = output_path(self.params['beam_dir'])
        noise_weight = self.params['noise_weight']
        skip_svd = self.params['skip_svd']
        ts_dir = output_path(self.params['ts_dir'])
        ts_name = self.params['ts_name']
        no_m_zero = self.params['no_m_zero']
        backup_ts = self.params['backup_ts']
        keep_ts_bk_num = self.params['keep_ts_bk_num']


        assert isinstance(ts, Timestream), '%s only works for Timestream object' % self.__class__.__name__

        ts.redistribute('time', via_memmap=via_memmap)

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
        if use_feedpos_in_file:
            feedpos = ts['feedpos'][:]
        else:
            # used the fixed feedpos
            feedpos = ts.feedpos

        # pols to consider
        pol_str = [ ts.pol_dict[p] for p in ts['pol'][:] ] # as string
        if pol == 'xx' or pol == 'yy':
            pis = [ pol_str.index(pol) ]
        elif pol == 'I':
            pis = [ pol_str.index('xx'), pol_str.index('yy') ]
            raise RuntimeError('pol can only be xx or yy now')
        else:
            raise ValueError('Invalid pol: %s' % pol)
        pi = pis[0]

        if ts.is_dish:
            from tlpipe.map.drift.telescope import tl_dish

            dish_width = ts.attrs['dishdiam']
            tel = tl_dish.TlUnpolarisedDishArray(lat, lon, freqs, band_width, tsys, ndays, accuracy_boost, l_boost, bl_range, auto_correlations, local_origin, lmax, mmax, dish_width, feedpos, pointing)
        elif ts.is_cylinder:
            from tlpipe.map.drift.telescope import tl_cylinder

            if 'beam_params' in ts.keys() and use_fitted_beam_params:
                if mpiutil.rank0:
                    print('Use fitted beam params')
                beam_params = ts['beam_params'].local_data[:, pi, :]
                cyl_width = beam_params[:, 0]
                fwhm_x = beam_params[:, 1]
                fwhm_y = beam_params[:, 2]
            elif use_beam_params_in_file:
                beam_params_name = input_path(beam_params_file)
                if mpiutil.rank0:
                    print(f'Use beam params in file {beam_params_name}')
                with h5py.File(beam_params_file, 'r') as f:
                    beam_params = f['beam_params'][:, pi, :]
                    cyl_width = beam_params[:, 0]
                    fwhm_x = beam_params[:, 1]
                    fwhm_y = beam_params[:, 2]
            else:
                # factor = 1.2 # suppose an illumination efficiency, keep same with that in timestream_common
                factor = 0.79 # for xx
                # factor = 0.88 # for yy
                cyl_width = factor * ts.attrs['cywid']
                cyl_width = np.array([cyl_width] * nfreq)
                fwhm_x = np.array([0.7] * nfreq)
                fwhm_y = np.array([1.0] * nfreq)
            tel = tl_cylinder.TlUnpolarisedCylinder(lat, lon, freqs, band_width, tsys, ndays, accuracy_boost, l_boost, bl_range, auto_correlations, local_origin, cyl_width, feedpos, lmax, mmax, True, True, 0.0, False, fwhm_x, fwhm_y, use_beam, use_beam_freq_offset)
        else:
            raise RuntimeError('Unknown array type %s' % ts.attrs['telescope'])

        allpairs = tel.allpairs
        redundancy = tel.redundancy
        red_bin = np.cumsum(np.insert(redundancy, 0, 0)) # redundancy bin
        unqpairs = tel.uniquepairs
        nuq = len(unqpairs) # number of unique pairs

        # to save m-mode
        # create an distributed array mmode to save memory use
        mmode = mpiarray.MPIArray((tel.mmax+1, nfreq, 2, nuq), axis=0, comm=ts.comm, dtype=np.complex128)
        mis = mpiarray.MPIArray.from_numpy_array(np.arange(tel.mmax+1), axis=0, root=None, comm=ts.comm)
        N = np.zeros((nfreq, nuq), dtype=float) # number of accumulate terms

        # mmode of a specific unique pair
        mmodeqi = np.zeros((2*tel.mmax+1, nfreq), dtype=np.complex128)
        Nqi = np.zeros((nfreq), dtype=float) # number of accumulate terms

        start_ra = ts.vis.attrs['start_ra']
        ra = mpiutil.gather_array(ts['ra_dec'].local_data[:, 0], root=None)
        ra = np.unwrap(ra)
        # find the first index that ra closest to start_ra
        ind = np.searchsorted(ra, start_ra)
        if np.abs(ra[ind] - start_ra) > np.abs(ra[ind+1] - start_ra):
            ind = ind + 1

        # get number of int_time in one sidereal day
        num_int = int(np.around(1.0 * const.sday / ts.attrs['inttime']))
        nt = ts.vis.shape[0]
        nt1 = min(num_int, nt-ind)

        inds = np.arange(nt)
        local_inds = mpiutil.scatter_array(inds, root=None, comm=ts.comm)

        local_phi = ts['ra_dec'].local_data[:, 0]
        # the Fourier transfom matrix
        E = np.exp(-1.0J * np.outer(np.arange(-tel.mmax, tel.mmax+1), local_phi)) # e^(- i m phi)

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
                    v = np.logical_not(M).astype(float) # 1 for valid, 0 for invalid
                    if 'local_hour_factor' in ts.keys():
                        a = ts['local_hour_factor'].local_data
                        V *= a[:, np.newaxis]
                        v *= a[:, np.newaxis]
                    mmodeqi += np.dot(E, V)
                    Nqi += np.sum(v, axis=0)

            mpiutil.barrier()

            # accumulate mmode from all processes by AllReduce
            if mpiutil.size > 1: # more than one processes
                # use IN_PLACE to reuse the mmode and N array
                mpiutil.world.Allreduce(mpiutil.IN_PLACE, mmodeqi, op=mpiutil.SUM)
                mpiutil.world.Allreduce(mpiutil.IN_PLACE, Nqi, op=mpiutil.SUM)

            # reshape mmode toseparate positive and negative ms
            mmodeqi1 = np.zeros((tel.mmax+1, nfreq, 2), dtype=mmodeqi.dtype)
            mmodeqi1[0, :, 0] = mmodeqi[tel.mmax]
            for mi in range(1, tel.mmax+1):
                mmodeqi1[mi, :, 0] = mmodeqi[tel.mmax+mi]
                mmodeqi1[mi, :, 1] = mmodeqi[tel.mmax-mi].conj()

            # lmmodeqi1 = mpiutil.scatter_array(mmodeqi1, axis=0, root=None, comm=ts.comm)
            # mmode.local_array[:, :, :, qi] = lmmodeqi1
            mmode.local_array[:, :, :, qi] = mmodeqi1[mis.local_array]
            N[:, qi] = Nqi

        del ts
        del E

        # beamtransfer
        bt = beamtransfer.BeamTransfer(f'{beam_dir}_{pol}', tel, noise_weight, skip_svd)
        # timestream
        tstream = timestream.Timestream(f'{ts_dir}_{pol}', ts_name, bt, no_m_zero)

        # save mmode to file
        mmode_dir = tstream.output_directory + '/mmodes'
        for i, mi in enumerate(mis.local_array[:]):
            if os.path.exists(mmode_dir + '/COMPLETED_M'):
                with h5py.File(tstream._mfile(mi), 'r+') as f:
                    f['/mmode'][:] += mmode.local_array[i]
            else:
                # make directory for each m-mode
                if not os.path.exists(tstream._mdir(mi)):
                    os.makedirs(tstream._mdir(mi))

                # create the m-file and save the result.
                with h5py.File(tstream._mfile(mi), 'w') as f:
                    f.create_dataset('/mmode', data=mmode.local_array[i])
                    f.attrs['m'] = mi

        mpiutil.barrier()

        if mpiutil.rank0:
            if os.path.exists(mmode_dir + '/COMPLETED_M'):
                with h5py.File(mmode_dir + '/count.hdf5', 'r+') as f:
                    f['count'][:] += N
            else:
                with h5py.File(mmode_dir + '/count.hdf5', 'w') as f:
                    f.create_dataset('count', data=N)

                # Make file marker that the m's have been correctly generated:
                open(mmode_dir + '/COMPLETED_M', 'a').close()

            # save the tstream object if there is no one
            if not os.path.isfile(tstream._picklefile):
                tstream.save()

            # backup timestream at each iteratiion
            if backup_ts:
                shutil.copytree(tstream.output_directory, f'{tstream.output_directory}_bk_{self.iteration}')
                # remove old backups
                for i in range(self.iteration - keep_ts_bk_num + 1):
                    shutil.rmtree(f'{tstream.output_directory}_bk_{i}', ignore_errors=True)

        mpiutil.barrier()

        return tstream