"""Imaging by Inverse Fourier Transform visibilities gridded in the uv-plane."""

try:
    import cPickle as pickle
except ImportError:
    import pickle

import os
import numpy as np
import ephem
import aipy as a
import h5py

from tlpipe.utils import mpiutil
from tlpipe.core.base_exe import Base
from tlpipe.utils.path_util import input_path, output_path


# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
params_init = {
               'nprocs': mpiutil.size, # number of processes to run this module
               'aprocs': range(mpiutil.size), # list of active process rank no.
               'input_file': 'uv.hdf5',
               'output_file': 'image.hdf5',
               # 'pol': 'I',
               'extra_history': '',
              }
prefix = 'im_'


class Imaging(Base):
    """Imaging by Inverse Fourier Transform visibilities gridded in the uv-plane."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(Imaging, self).__init__(parameter_file_or_dict, params_init, prefix, feedback)

    def execute(self):

        input_file = input_path(self.params['input_file'])
        output_file = output_path(self.params['output_file'])

        if mpiutil.rank0:
            with h5py.File(input_file, 'r') as f:
                uv = f['uv'][...]
                uv_cov = f['uv_cov'][...]
                uv_res = f.attrs['res']
                max_lm = 0.5 * 1.0 / uv_res
                size = uv.shape[0]
                center = size / 2
                res = max_lm / center # resolution in the lm-plane

                # fourier transform
                uv_cov_fft = np.fft.ifft2(np.fft.ifftshift(uv_cov))
                uv_cov_fft = np.fft.ifftshift(uv_cov_fft)
                uv_fft = np.fft.ifft2(np.fft.ifftshift(uv))
                uv_fft = np.fft.ifftshift(uv_fft)
                uv_imag_fft = np.fft.ifft2(np.fft.ifftshift(1.0J * uv.imag))
                uv_imag_fft = np.fft.ifftshift(uv_imag_fft)

                # imaginary part should be 0
                assert(np.allclose(uv_cov_fft.imag, 0))
                assert(np.allclose(uv_fft.imag, 0))
                assert(np.allclose(uv_imag_fft.imag, 0))

                # save data
                with h5py.File(output_file, 'w') as fout:
                    # save only real part
                    fout.create_dataset('uv_cov_fft', data=uv_cov_fft.real)
                    fout.create_dataset('uv_fft', data=uv_fft.real)
                    fout.create_dataset('uv_imag_fft', data=uv_imag_fft.real)
                    # copy meta data from input file
                    for attrs_name, attrs_value in f.attrs.iteritems():
                        fout.attrs[attrs_name] = attrs_value
                    # update some attrs
                    fout.attrs['res'] = res
                    fout.attrs['max_lm'] = max_lm
                    fout.attrs['d_ra'] = np.degrees(2.0 * max_lm / size)
                    fout.attrs['d_dec'] = np.degrees(2.0 * max_lm / size)
                    fout.attrs['history'] = fout.attrs['history'] + self.history
                    del fout.attrs['max_wl']
