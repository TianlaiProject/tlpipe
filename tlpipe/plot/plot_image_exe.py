"""Plot image."""

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tlpipe.utils import mpiutil
from tlpipe.core.base_exe import Base
from tlpipe.utils.path_util import input_path, output_path


# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
params_init = {
               'nprocs': mpiutil.size, # number of processes to run this module
               'aprocs': range(mpiutil.size), # list of active process rank no.
               'input_file': ['uv_image.hdf5'], # str or a list of str
               'output_file': None, # None, str or a list of str
               'scale': 2,
              }
prefix = 'plti_'



class Plot(Base):
    """Plot image."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(Plot, self).__init__(parameter_file_or_dict, params_init, prefix, feedback)

    def execute(self):

        input_file = input_path(self.params['input_file'])
        output_file = self.params['output_file']
        if output_file is not None:
            output_file = output_path(output_file)
        scale = self.params['scale']

        if type(input_file) is str:
            input_file = [input_file]
        else:
            input_file = list(input_file)

        if output_file is None:
            output_file = [infile.replace('.hdf5', '.png') for infile in input_file]
        elif type(output_file) is str:
            output_file = [output_file]
        else:
            output_file = list(output_file)

        for infile, outfile in zip(mpiutil.mpilist(input_file), mpiutil.mpilist(output_file)):
            with h5py.File(infile, 'r') as f:
                uv_cov = f['uv_cov'][...]
                uv = f['uv'][...]
                uv_cov_fft = f['uv_cov_fft'][...]
                uv_fft = f['uv_fft'][...]
                uv_imag_fft = f['uv_imag_fft'][...]
                max_wl = f.attrs['max_wl']
                max_lm = f.attrs['max_lm']


            plt.figure(figsize=(13, 8))
            plt.subplot(231)
            extent = [-max_wl, max_wl, -max_wl, max_wl]
            plt.imshow(uv_cov.real/np.sqrt(np.abs(uv_cov.real)), origin='lower', aspect='auto', extent=extent, interpolation='nearest')
            plt.xlabel(r'$u$ / $\lambda$')
            plt.ylabel(r'$v$ / $\lambda$')
            plt.colorbar()
            plt.subplot(232)
            plt.imshow(uv.real/np.sqrt(np.abs(uv.real)), origin='lower', aspect='auto', extent=extent, interpolation='nearest')
            plt.xlabel(r'$u$ / $\lambda$')
            plt.ylabel(r'$v$ / $\lambda$')
            plt.colorbar()
            plt.subplot(233)
            plt.imshow(uv.imag/np.sqrt(np.abs(uv.imag)), origin='lower', aspect='auto', extent=extent, interpolation='nearest')
            plt.xlabel(r'$u$ / $\lambda$')
            plt.ylabel(r'$v$ / $\lambda$')
            plt.colorbar()

            plt.subplot(234)
            extent = [-max_lm/scale, max_lm/scale, -max_lm/scale, max_lm/scale]
            shp = uv.shape
            assert shp[0] == shp[1]
            ct = shp[0]/2
            plt_data = uv_cov_fft.real[ct-ct/scale:ct+ct/scale, ct-ct/scale:ct+ct/scale]
            plt.imshow(plt_data/np.sqrt(np.abs(plt_data)), origin='lower', aspect='auto', extent=extent, interpolation='nearest')
            # plt.imshow(plt_data, origin='lower', aspect='auto', extent=extent, interpolation='nearest')
            plt.xlabel(r'$l$')
            plt.ylabel(r'$m$')
            plt.colorbar()
            plt.subplot(235)
            plt_data = uv_fft.real[ct-ct/scale:ct+ct/scale, ct-ct/scale:ct+ct/scale]
            plt.imshow(plt_data/np.sqrt(np.abs(plt_data)), origin='lower', aspect='auto', extent=extent, interpolation='nearest')
            # plt.imshow(plt_data, origin='lower', aspect='auto', extent=extent, interpolation='nearest')
            plt.xlabel(r'$l$')
            plt.ylabel(r'$m$')
            plt.colorbar()
            # plt.subplot(236)
            # plt_data = uv_fft.imag[ct-ct/scale:ct+ct/scale, ct-ct/scale:ct+ct/scale] # should be 0
            # plt.imshow(plt_data/np.sqrt(np.abs(plt_data)), origin='lower', aspect='auto', extent=extent, interpolation='nearest')
            # # plt.imshow(plt_data, origin='lower', aspect='auto', extent=extent, interpolation='nearest')
            # plt.xlabel(r'$l$')
            # plt.ylabel(r'$m$')
            # plt.colorbar()
            plt.subplot(236)
            plt_data = uv_imag_fft.real[ct-ct/scale:ct+ct/scale, ct-ct/scale:ct+ct/scale]
            plt.imshow(plt_data/np.sqrt(np.abs(plt_data)), origin='lower', aspect='auto', extent=extent, interpolation='nearest')
            # plt.imshow(plt_data, origin='lower', aspect='auto', extent=extent, interpolation='nearest')
            plt.xlabel(r'$l$')
            plt.ylabel(r'$m$')
            plt.colorbar()


            plt.savefig(outfile)





# data_dir = '/home/zuoshifan/programming/python/21cmcosmology/dishary/example_cal/tldishes/Cas_20151227/'
# output_dir = data_dir + 'output/'
# input_imag_file = output_dir + 'uv_imag_conv.hdf5'
# # input_imag_file = output_dir + 'uv_imag_noconv.hdf5'
# # input_imag_file = output_dir + 'uv_imag_noshift.hdf5'
# # input_imag_file = output_dir + 'uv_imag_nocal_conv.hdf5'
# # input_imag_file = output_dir + 'uv_imag_nocal_noconv.hdf5'
# # input_imag_file = output_dir + 'uv_imag_nocal_noshift.hdf5'

# # input_imag_file = output_dir + 'uv_imag_conv_c0.hdf5'
# # input_imag_file = output_dir + 'uv_imag_noconv_c0.hdf5'
# # input_imag_file = output_dir + 'uv_imag_noshift_c0.hdf5'
# # input_imag_file = output_dir + 'uv_imag_nocal_conv_c0.hdf5'
# # input_imag_file = output_dir + 'uv_imag_nocal_noconv_c01.hdf5'
# # input_imag_file = output_dir + 'uv_imag_nocal_noshift_c0.hdf5'
# output_imag_file = input_imag_file.replace('.hdf5', '.png')

# with h5py.File(input_imag_file, 'r') as f:
#     uv_cov = f['uv_cov'][...]
#     uv = f['uv'][...]
#     uv_cov_fft = f['uv_cov_fft'][...]
#     uv_fft = f['uv_fft'][...]
#     uv_imag_fft = f['uv_imag_fft'][...]
#     max_wl = f.attrs['max_wl']
#     max_lm = f.attrs['max_lm']


# scale = 2

# plt.figure(figsize=(13, 8))
# plt.subplot(231)
# extent = [-max_wl, max_wl, -max_wl, max_wl]
# plt.imshow(uv_cov.real/np.sqrt(np.abs(uv_cov.real)), origin='lower', aspect='auto', extent=extent, interpolation='nearest')
# plt.xlabel(r'$u$ / $\lambda$')
# plt.ylabel(r'$v$ / $\lambda$')
# plt.colorbar()
# plt.subplot(232)
# plt.imshow(uv.real/np.sqrt(np.abs(uv.real)), origin='lower', aspect='auto', extent=extent, interpolation='nearest')
# plt.xlabel(r'$u$ / $\lambda$')
# plt.ylabel(r'$v$ / $\lambda$')
# plt.colorbar()
# plt.subplot(233)
# plt.imshow(uv.imag/np.sqrt(np.abs(uv.imag)), origin='lower', aspect='auto', extent=extent, interpolation='nearest')
# plt.xlabel(r'$u$ / $\lambda$')
# plt.ylabel(r'$v$ / $\lambda$')
# plt.colorbar()

# plt.subplot(234)
# extent = [-max_lm/scale, max_lm/scale, -max_lm/scale, max_lm/scale]
# shp = uv.shape
# assert shp[0] == shp[1]
# ct = shp[0]/2
# plt_data = uv_cov_fft.real[ct-ct/scale:ct+ct/scale, ct-ct/scale:ct+ct/scale]
# plt.imshow(plt_data/np.sqrt(np.abs(plt_data)), origin='lower', aspect='auto', extent=extent, interpolation='nearest')
# # plt.imshow(plt_data, origin='lower', aspect='auto', extent=extent, interpolation='nearest')
# plt.xlabel(r'$l$')
# plt.ylabel(r'$m$')
# plt.colorbar()
# plt.subplot(235)
# plt_data = uv_fft.real[ct-ct/scale:ct+ct/scale, ct-ct/scale:ct+ct/scale]
# plt.imshow(plt_data/np.sqrt(np.abs(plt_data)), origin='lower', aspect='auto', extent=extent, interpolation='nearest')
# # plt.imshow(plt_data, origin='lower', aspect='auto', extent=extent, interpolation='nearest')
# plt.xlabel(r'$l$')
# plt.ylabel(r'$m$')
# plt.colorbar()
# # plt.subplot(236)
# # plt_data = uv_fft.imag[ct-ct/scale:ct+ct/scale, ct-ct/scale:ct+ct/scale] # should be 0
# # plt.imshow(plt_data/np.sqrt(np.abs(plt_data)), origin='lower', aspect='auto', extent=extent, interpolation='nearest')
# # # plt.imshow(plt_data, origin='lower', aspect='auto', extent=extent, interpolation='nearest')
# # plt.xlabel(r'$l$')
# # plt.ylabel(r'$m$')
# # plt.colorbar()
# plt.subplot(236)
# plt_data = uv_imag_fft.real[ct-ct/scale:ct+ct/scale, ct-ct/scale:ct+ct/scale]
# plt.imshow(plt_data/np.sqrt(np.abs(plt_data)), origin='lower', aspect='auto', extent=extent, interpolation='nearest')
# # plt.imshow(plt_data, origin='lower', aspect='auto', extent=extent, interpolation='nearest')
# plt.xlabel(r'$l$')
# plt.ylabel(r'$m$')
# plt.colorbar()


# plt.savefig(output_imag_file)
