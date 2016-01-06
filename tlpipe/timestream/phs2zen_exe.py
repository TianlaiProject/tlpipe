"""Module to phase data to zenith."""

try:
    import cPickle as pickle
except ImportError:
    import pickle

import os
import numpy as np
from scipy.optimize import curve_fit
# from scipy.interpolate import UnivariateSpline
import h5py

from tlpipe.kiyopy import parse_ini
from tlpipe.utils import mpiutil


# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
params_init = {
               'nprocs': mpiutil.size, # number of processes to run this module
               'aprocs': range(mpiutil.size), # list of active process rank no.
               'data_files': ['./data.hdf5'],
              }
prefix = 'ph_'


pol_dict = {0: 'xx', 1: 'yy', 2: 'xy', 3: 'yx'}

def gauss(x, a, x0, sigma, b):
    return a * np.exp(-(x-x0)**2 / (2 * sigma**2)) + b


class Phs2zen(object):
    """Phase the data to zenith."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        # Read in the parameters.
        self.params = parse_ini.parse(parameter_file_or_dict, params_init,
                                 prefix=prefix, feedback=feedback)
        self.feedback = feedback
        nprocs = min(self.params['nprocs'], mpiutil.size)
        procs = set(range(mpiutil.size))
        aprocs = set(self.params['aprocs']) & procs
        self.aprocs = (list(aprocs) + list(set(range(nprocs)) - aprocs))[:nprocs]
        assert 0 in self.aprocs, 'Process 0 must be active'
        self.comm = mpiutil.active_comm(self.aprocs) # communicator consists of active processes

    def execute(self):

        output_dir = os.environ['TL_OUTPUT']
        data_files = self.params['data_files']
        nfiles = len(data_files)
        assert nfiles > 0, 'No input data files'

        # read in ants, freq, time info from data files
        with h5py.File(data_files[0], 'r') as f:
            dataset = f['data']
            data_shp = dataset.shape
            data_type = dataset.dtype
            ants = dataset.attrs['ants']
            freq = dataset.attrs['freq']
            # ts = f['time'] # Julian date for data in this file only

        npol = data_shp[2]
        nfreq = len(freq)
        # cut central 80% of the data
        # data[np.int(0.1*nt):np.int(0.9*nt)] = 0
        # only early 10%
        # data[np.int(0.1*nt):] = 0
        # only late 10%
        # data[:np.int(0.9*nt)] = 0

        nants = len(ants)
        bls = [(ants[i], ants[j]) for i in range(nants) for j in range(i, nants)]
        nbls = len(bls)

        lbls, sbl, ebl = mpiutil.split_local(nbls)
        local_bls = range(sbl, ebl)
        # local data section corresponding to local bls
        local_data = np.array([], dtype=data_type).reshape((0, lbls, npol, nfreq))
        ts = np.array([], dtype=np.float64)
        for ind, data_file in enumerate(data_files):
            with h5py.File(data_file, 'r') as f:
                local_data = np.concatenate((local_data, f['data'][:, sbl:ebl, :, :]), axis=0)
                ts = np.concatenate((ts, f['time'][...])) # Julian date
        local_int_time = np.zeros(local_data.shape[1:], dtype=local_data.dtype)

        nt = len(ts)
        if mpiutil.size == 1:
            data_phs2zen = local_data.view()
            data_int_time = local_int_time.view()
        else:
            if mpiutil.rank0:
                data_phs2zen = np.zeros((nt,) + data_shp[1:], dtype=data_type) # save data that phased to zenith
                data_int_time = np.zeros(data_phs2zen.shape[1:], dtype=data_type) # save data integrate over time
            else:
                data_phs2zen = None
                data_int_time = None

        for pol_ind in range(npol):
            for bi, bl_ind in enumerate(local_bls): # mpi among bls
                # # ignore auto-correlation
                # bl = bls[bl_ind]
                # if bl[0] == bl[1]:
                #     continue

                data_slice = local_data[:, bi, pol_ind, :].copy() # will use local_data to save data_slice_dphs in-place, so here use copy

                # subtract the mean
                # data -= np.mean(data, axis=1)

                # freq fft
                data_slice_fft_freq = np.fft.fft(data_slice, axis=1)
                data_slice_fft_freq = np.fft.fftshift(data_slice_fft_freq, axes=1)
                # time fft
                data_slice_fft_time = np.fft.fft(data_slice, axis=0)
                data_slice_fft_time = np.fft.fftshift(data_slice_fft_time, axes=0)
                # freq and time fft
                data_slice_fft2 = np.fft.fft2(data_slice)
                data_slice_fft2 = np.fft.fftshift(data_slice_fft2)

                ########################
                # find max in time fft
                max_row_ind = np.argmax(np.abs(data_slice_fft_time), axis=0)
                data_slice_fft_time_max = np.zeros_like(data_slice_fft_time)
                for ci in range(nfreq):
                    data_slice_fft_time_max[max_row_ind[ci], ci] = data_slice_fft_time[max_row_ind[ci], ci]

                # ifft for time
                data_slice_new = np.fft.ifft(np.fft.ifftshift(data_slice_fft_time_max, axes=0), axis=0)

                # freq fft
                data_slice_new_fft_freq = np.fft.fft(data_slice_new, axis=1)
                data_slice_new_fft_freq = np.fft.fftshift(data_slice_new_fft_freq, axes=1)
                # time fft
                data_slice_new_fft_time = np.fft.fft(data_slice_new, axis=0)
                data_slice_new_fft_time = np.fft.fftshift(data_slice_new_fft_time, axes=0)
                # freq and time fft
                data_slice_new_fft2 = np.fft.fft2(data_slice_new)
                data_slice_new_fft2 = np.fft.fftshift(data_slice_new_fft2)


                # divide phase
                data_slice_dphs = data_slice / (data_slice_new / np.abs(data_slice_new))
                # fill invalid values (divide 0 or something else) by 0.0
                data_slice_dphs[np.logical_not(np.isfinite(data_slice_dphs))] = 0.0
                # save data after phas2 to zenith
                # data_phs2zen[:, bl_ind, pol_ind, :] = data_slice_dphs
                local_data[:, bi, pol_ind, :] = data_slice_dphs # change local_data to save memory

                # freq fft
                data_slice_dphs_fft_freq = np.fft.fft(data_slice_dphs, axis=1)
                data_slice_dphs_fft_freq = np.fft.fftshift(data_slice_dphs_fft_freq, axes=1)
                # time fft
                data_slice_dphs_fft_time = np.fft.fft(data_slice_dphs, axis=0)
                data_slice_dphs_fft_time = np.fft.fftshift(data_slice_dphs_fft_time, axes=0)
                # freq and time fft
                data_slice_dphs_fft2 = np.fft.fft2(data_slice_dphs)
                data_slice_dphs_fft2 = np.fft.fftshift(data_slice_dphs_fft2)

                # # Fit a Gaussian function to data_slice_dphs
                # data_slice_dphs_gauss_fit = np.zeros_like(data_slice_dphs)
                # data_slice_dphs_xgauss = np.zeros_like(data_slice_dphs)
                # for fi in range(nfreq):
                #     try:
                #         # data_slice_dphs_smooth = UnivariateSpline(ts, data_slice_dphs[:, fi].real, s=1)(ts)
                #         data_slice_dphs_smooth = data_slice_dphs[:, fi].real # maybe should try some smooth
                #         max_val = np.max(data_slice_dphs_smooth)
                #         max_ind = np.argmax(data_slice_dphs_smooth)
                #         # sigma = np.sum(np.sqrt((ts-ts[max_ind])**2 / (2 * np.log(np.abs(max_val / (data_slice_dphs[:, fi].real - 0.0)))))) / nt
                #         sigma = 1.0
                #         popt,pcov = curve_fit(gauss, ts, data_slice_dphs_smooth, p0=[max_val, ts[max_ind], sigma, 0.0]) # now only fit real part
                #         data_slice_dphs_gauss_fit[:, fi] = gauss(ts, *popt)
                #     except RuntimeError:
                #         print 'Error occured while fitting pol: %s, bl: (%d, %d), fi: %d' % (pol_dict[pol_ind], bls[bl_ind][0], bls[bl_ind][1], fi)
                #         # print data_slice_dphs[:, fi].real
                #         plt.figure()
                #         plt.plot(ts, data_slice_dphs[:, fi].real)
                #         plt.plot(ts, data_slice_dphs_smooth)
                #         plt.xlabel('t')
                #         figname = output_dir + 'data_slice_%d_%d_%s_%d.png' % (bls[bl_ind][0], bls[bl_ind][1], pol_dict[pol_ind], fi)
                #         plt.savefig(figname)
                #         # data_slice_dphs_gauss_fit[:, fi] = data_slice_dphs[:, fi].real
                #         data_slice_dphs_gauss_fit[:, fi] = data_slice_dphs_smooth
                # data_slice_dphs_xgauss = data_slice_dphs * data_slice_dphs_gauss_fit # assume int_time = 1 here

                # # integrate over time
                # data_slice_int_time = np.sum(data_slice_dphs_xgauss, axis=0)
                # data_int_time[bl_ind, pol_ind, :] = data_slice_int_time

                # integrate over time
                data_slice_int_time = np.sum(data_slice_dphs, axis=0)
                local_int_time[bi, pol_ind, :] = data_slice_int_time

                # save data to file
                filename = output_dir + 'data_slice_%d_%d_%s.hdf5' % (bls[bl_ind][0], bls[bl_ind][1], pol_dict[pol_ind])
                with h5py.File(filename, 'w') as f:
                    f.create_dataset('data_slice', data=data_slice)
                    f.create_dataset('data_slice_fft_freq', data=data_slice_fft_freq)
                    f.create_dataset('data_slice_fft_time', data=data_slice_fft_time)
                    f.create_dataset('data_slice_fft2', data=data_slice_fft2)
                    f.create_dataset('data_slice_fft_time_max', data=data_slice_fft_time_max)
                    f.create_dataset('data_slice_new', data=data_slice_new)
                    f.create_dataset('data_slice_new_fft_freq', data=data_slice_new_fft_freq)
                    f.create_dataset('data_slice_new_fft_time', data=data_slice_new_fft_time)
                    f.create_dataset('data_slice_new_fft2', data=data_slice_new_fft2)
                    f.create_dataset('data_slice_dphs', data=data_slice_dphs)
                    f.create_dataset('data_slice_dphs_fft_freq', data=data_slice_dphs_fft_freq)
                    f.create_dataset('data_slice_dphs_fft_time', data=data_slice_dphs_fft_time)
                    f.create_dataset('data_slice_dphs_fft2', data=data_slice_dphs_fft2)
                    # f.create_dataset('data_slice_dphs_gauss_fit', data=data_slice_dphs_gauss_fit)
                    # f.create_dataset('data_slice_dphs_xgauss', data=data_slice_dphs_xgauss)
                    f.create_dataset('data_slice_int_time', data=data_slice_int_time)


        # Gather data in separate processes
        if self.comm is not None and self.comm.size > 1: # Reduce only when there are multiple processes
            mpiutil.gather_local(data_phs2zen, local_data, (0, sbl, 0, 0), root=0, comm=self.comm)
            mpiutil.gather_local(data_int_time, local_int_time, (sbl, 0, 0), root=0, comm=self.comm)


        # save data phased to zenith
        if mpiutil.rank0:
            with h5py.File(output_dir + 'data_phs2zen.hdf5', 'w') as f:
                f.create_dataset('time', data=ts)
                f.create_dataset('data_int_time', data=data_int_time)
                dset = f.create_dataset('data', data=data_phs2zen)
                # copy metadata from input file
                with h5py.File(data_files[0], 'r') as fin:
                    for attrs_name, attrs_value in fin['data'].attrs.iteritems():
                        dset.attrs[attrs_name] = attrs_value
                # update some attrs
                dset.attrs['history'] = dset.attrs['history'] + 'Phased data to zenith with parameters %s.\n' % self.params
