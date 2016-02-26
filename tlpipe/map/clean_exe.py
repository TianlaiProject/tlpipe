"""Module to clean the image."""

try:
    import cPickle as pickle
except ImportError:
    import pickle

import os
import numpy as np
import aipy as a
import h5py

from tlpipe.utils import mpiutil
from tlpipe.core.base_exe import Base
# from tlpipe.core import tldishes
from tlpipe.utils.path_util import input_path, output_path


# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
params_init = {
               'nprocs': mpiutil.size, # number of processes to run this module
               'aprocs': range(mpiutil.size), # list of active process rank no.
               'input_file': 'dirty_image.hdf5',
               'output_file': 'clean_image.hdf5',
               'method': ['cln'],
               'model': None, # the model image
               'pol': 'I',
               'gain': 0.1,
               'maxiter': 10000,
               'tol': 1.0e-3,
               'pos_def': False,
               'verbose': False,
               'extra_history': '',
              }
prefix = 'cl_'


pol_dict = {'I': 0, 'Q': 1, 'U': 2, 'V': 3}


class Clean(Base):
    """Clean the image."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(Clean, self).__init__(parameter_file_or_dict, params_init, prefix, feedback)

    def execute(self):

        input_file = input_path(self.params['input_file'])
        output_file = output_path(self.params['output_file'])
        method = self.params['method']
        model = self.params['model']
        gain = self.params['gain']
        maxiter = self.params['maxiter']
        tol = self.params['tol']
        pos_def = self.params['pos_def']
        verbose = self.params['verbose']

        with h5py.File(input_file, 'r') as f:
            dim = f['uv_fft'][...]
            dbm = f['uv_cov_fft'][...]
            # max_wl = f.attrs['max_wl']
            # max_lm = f.attrs['max_lm']

        DIM = dim.shape[0]
        dbm = a.img.recenter(dbm, (DIM/2,DIM/2))
        bm_gain = a.img.beam_gain(dbm)
        if mpiutil.rank0:
            print 'Gain of dirty beam:', bm_gain
        for md in mpiutil.mpilist(method):
            if md == 'cln':
                cim, info = a.deconv.clean(dim, dbm, mdl=model, gain=gain, maxiter=maxiter, stop_if_div=True, verbose=verbose, tol=tol, pos_def=pos_def)
            elif md == 'mem':
                cim, info = a.deconv.maxent_findvar(dim, dbm, mdl=model, f_var0=0.6, maxiter=maxiter, verbose=verbose, tol=tol, maxiterok=True)
            elif md == 'lsq':
                cim, info = a.deconv.lsq(dim, dbm, mdl=model, maxiter=maxiter, verbose=verbose, tol=tol)
            elif md == 'ann':
                cim, info = a.deconv.anneal(dim, dbm, mdl=model, maxiter=maxiter, cooling=lambda i,x: tol*(1-np.cos(i/50.0))*(x**2), verbose=verbose)

            # Fit a 2d Gaussian to the dirty beam and convolve that with the clean components.
            dbm_fit = np.fft.fftshift(dbm)
            DIM = dbm.shape[0]
            lo, hi = (DIM - 30)/2, (DIM + 30)/2
            dbm_fit = dbm_fit[lo:hi, lo:hi]

            cbm = a.twodgauss.twodgaussian(a.twodgauss.moments(dbm_fit), shape=dbm.shape)
            cbm = a.img.recenter(cbm, (np.ceil((DIM+dbm_fit.shape[0])/2), np.ceil((DIM+dbm_fit.shape[0])/2)))
            cbm /= np.sum(cbm)

            cimc = np.fft.fftshift(np.fft.ifft2(np.fft.fft2(cim)*np.fft.fft2(cbm))).real

            rim = info['res']

            bim = rim / bm_gain + cimc

            # save clean image
            output_file = output_file.replace('.hdf5', '_%s.hdf5' % md)
            with h5py.File(output_file, 'w') as f:
                f.create_dataset('bim', data=bim)
                f.create_dataset('cim', data=cim)
                f.create_dataset('cimc', data=cimc)
                f.create_dataset('rim', data=rim)
                # copy metadata from input file
                with h5py.File(input_file, 'r') as fin:
                    for attrs_name, attrs_value in fin.attrs.iteritems():
                        f.attrs[attrs_name] = attrs_value
                    # update some attrs
                    f.attrs['history'] = fin.attrs['history'] + self.history
