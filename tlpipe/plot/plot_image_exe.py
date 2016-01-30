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
               'input_file': [('uv.hdf5', 'image.hdf5')], # str or a list of str
               'output_file': None, # None, str or a list of str
               'scale': 2,
               'plot_sqrt': False,
              }
prefix = 'plti_'



class Plot(Base):
    """Plot image."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(Plot, self).__init__(parameter_file_or_dict, params_init, prefix, feedback)

    def execute(self):

        input_file = self.params['input_file']
        # input_file = input_path(self.params['input_file'])
        output_file = self.params['output_file']
        if output_file is not None:
            output_file = output_path(output_file)
        scale = self.params['scale']
        plot_sqrt = self.params['plot_sqrt']

        if type(input_file) is not list:
            input_file = [input_file]
        else:
            input_file = list(input_file)

        if output_file is None:
            output_file = [infile[1].replace('.hdf5', '.png') for infile in input_file]
        elif type(output_file) is str:
            output_file = [output_file]
        else:
            output_file = list(output_file)

        for infile, outfile in zip(mpiutil.mpilist(input_file), mpiutil.mpilist(output_file)):
            uv_file, im_file = infile
            uv_file = input_path(uv_file)
            im_file = input_path(im_file)
            outfile = output_path(outfile)
            with h5py.File(uv_file, 'r') as fuv, h5py.File(im_file) as fim:
                uv_cov = fuv['uv_cov'][...]
                uv = fuv['uv'][...]
                max_wl = fuv.attrs['max_wl']

                uv_cov_fft = fim['uv_cov_fft'][...]
                uv_fft = fim['uv_fft'][...]
                uv_imag_fft = fim['uv_imag_fft'][...]
                max_lm = fim.attrs['max_lm']


            plt.figure(figsize=(13, 8))
            plt.subplot(231)
            extent = [-max_wl, max_wl, -max_wl, max_wl]
            plt_data = uv_cov.real
            if plot_sqrt:
                plt.imshow(plt_data/np.sqrt(np.abs(plt_data)), origin='lower', aspect='auto', extent=extent, interpolation='nearest')
            else:
                plt.imshow(plt_data, origin='lower', aspect='auto', extent=extent, interpolation='nearest')
            plt.xlabel(r'$u$ / $\lambda$')
            plt.ylabel(r'$v$ / $\lambda$')
            plt.colorbar()
            plt.subplot(232)
            plt_data = uv.real
            if plot_sqrt:
                plt.imshow(plt_data/np.sqrt(np.abs(plt_data)), origin='lower', aspect='auto', extent=extent, interpolation='nearest')
            else:
                plt.imshow(plt_data, origin='lower', aspect='auto', extent=extent, interpolation='nearest')
            plt.xlabel(r'$u$ / $\lambda$')
            plt.ylabel(r'$v$ / $\lambda$')
            plt.colorbar()
            plt.subplot(233)
            plt_data = uv.imag
            if plot_sqrt:
                plt.imshow(plt_data/np.sqrt(np.abs(plt_data)), origin='lower', aspect='auto', extent=extent, interpolation='nearest')
            else:
                plt.imshow(plt_data, origin='lower', aspect='auto', extent=extent, interpolation='nearest')
            plt.xlabel(r'$u$ / $\lambda$')
            plt.ylabel(r'$v$ / $\lambda$')
            plt.colorbar()

            plt.subplot(234)
            extent = [-max_lm/scale, max_lm/scale, -max_lm/scale, max_lm/scale]
            shp = uv.shape
            assert shp[0] == shp[1]
            ct = shp[0]/2
            plt_data = uv_cov_fft.real[ct-ct/scale:ct+ct/scale, ct-ct/scale:ct+ct/scale]
            if plot_sqrt:
                plt.imshow(plt_data/np.sqrt(np.abs(plt_data)), origin='lower', aspect='auto', extent=extent, interpolation='nearest')
            else:
                plt.imshow(plt_data, origin='lower', aspect='auto', extent=extent, interpolation='nearest')
            plt.xlabel(r'$l$')
            plt.ylabel(r'$m$')
            plt.colorbar()
            plt.subplot(235)
            plt_data = uv_fft.real[ct-ct/scale:ct+ct/scale, ct-ct/scale:ct+ct/scale]
            if plot_sqrt:
                plt.imshow(plt_data/np.sqrt(np.abs(plt_data)), origin='lower', aspect='auto', extent=extent, interpolation='nearest')
            else:
                plt.imshow(plt_data, origin='lower', aspect='auto', extent=extent, interpolation='nearest')
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
            if plot_sqrt:
                plt.imshow(plt_data/np.sqrt(np.abs(plt_data)), origin='lower', aspect='auto', extent=extent, interpolation='nearest')
            else:
                plt.imshow(plt_data, origin='lower', aspect='auto', extent=extent, interpolation='nearest')
            plt.xlabel(r'$l$')
            plt.ylabel(r'$m$')
            plt.colorbar()


            plt.savefig(outfile)
