"""Plot gain."""

import numpy as np
import h5py
import matplotlib.pyplot as plt

from tlpipe.utils import mpiutil
from tlpipe.core.base_exe import Base
from tlpipe.utils.path_util import input_path, output_path


# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
params_init = {
               'nprocs': mpiutil.size, # number of processes to run this module
               'aprocs': range(mpiutil.size), # list of active process rank no.
               'input_file': 'gain.hdf5',
               'output_file': None, # None, str or a list of str
               'plot_type': 'amp_phs', # or 'real_imag' or 'amp_phs,real_imag'
               'sub_mean': False,
              }
prefix = 'pltg_'



class Plot(Base):
    """Plot gain."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(Plot, self).__init__(parameter_file_or_dict, params_init, prefix, feedback)

    def execute(self):

        input_file = input_path(self.params['input_file'])
        output_file = self.params['output_file']
        plot_type = self.params['plot_type']
        sub_mean = self.params['sub_mean']

        if output_file is None:
            output_file = output_path('gain.png')
        else:
            output_file = output_path(output_file)
        suffix = output_file.split('.')[-1]

        with h5py.File(input_file, 'r') as f:
            dset = f['gain']
            gain = dset[...]
            time = dset.attrs['time']
            ants = dset.attrs['ants']
            freq = dset.attrs['freq']

        ant_inds = range(len(ants))
        extent = [time[0], time[-1], freq[0], freq[-1]]

        for ant_ind in mpiutil.mpilist(ant_inds):
            for plt_type in plot_type.split(','):
                if plt_type == 'amp_phs':
                    outfile = output_file.replace('.'+suffix, '_'+plt_type+'_%d.'+suffix)
                    plt.figure()
                    plt.subplot(411)
                    plt_data = np.abs(gain[:, ant_ind, 0, :])
                    if sub_mean:
                        plt_data -= np.mean(plt_data, axis=0)
                    plt.imshow(plt_data.T, origin='lower', extent=extent, aspect='auto')
                    plt.xlabel(r'$t$')
                    plt.ylabel(r'X Amp $\nu$ / MHz')
                    plt.colorbar()
                    plt.subplot(412)
                    plt_data = np.angle(gain[:, ant_ind, 0, :])
                    if sub_mean:
                        plt_data -= np.mean(plt_data, axis=0)
                    plt.imshow(plt_data.T, origin='lower', extent=extent, aspect='auto')
                    plt.xlabel(r'$t$')
                    plt.ylabel(r'X Phs $\nu$ / MHz')
                    plt.colorbar()
                    plt.subplot(413)
                    plt_data = np.abs(gain[:, ant_ind, 1, :])
                    if sub_mean:
                        plt_data -= np.mean(plt_data, axis=0)
                    plt.imshow(plt_data.T, origin='lower', extent=extent, aspect='auto')
                    plt.xlabel(r'$t$')
                    plt.ylabel(r'Y Amp $\nu$ / MHz')
                    plt.colorbar()
                    plt.subplot(414)
                    plt_data = np.angle(gain[:, ant_ind, 1, :])
                    if sub_mean:
                        plt_data -= np.mean(plt_data, axis=0)
                    plt.imshow(plt_data.T, origin='lower', extent=extent, aspect='auto')
                    plt.xlabel(r'$t$')
                    plt.ylabel(r'Y Phs $\nu$ / MHz')
                    plt.colorbar()
                    plt.savefig(outfile % ant_ind)
                elif plt_type == 'real_imag':
                    outfile = output_file.replace('.'+suffix, '_'+plt_type+'_%d.'+suffix)
                    plt.figure()
                    plt.subplot(411)
                    plt_data = gain[:, ant_ind, 0, :]
                    if sub_mean:
                        plt_data -= np.mean(plt_data, axis=0)
                    plt.imshow(plt_data.T.real, origin='lower', extent=extent, aspect='auto')
                    plt.xlabel(r'$t$')
                    plt.ylabel(r'X Real $\nu$ / MHz')
                    plt.colorbar()
                    plt.subplot(412)
                    plt.imshow(plt_data.T.imag, origin='lower', extent=extent, aspect='auto')
                    plt.xlabel(r'$t$')
                    plt.ylabel(r'X Imag $\nu$ / MHz')
                    plt.colorbar()
                    plt.subplot(413)
                    plt_data = gain[:, ant_ind, 1, :]
                    if sub_mean:
                        plt_data -= np.mean(plt_data, axis=0)
                    plt.imshow(plt_data.T.real, origin='lower', extent=extent, aspect='auto')
                    plt.xlabel(r'$t$')
                    plt.ylabel(r'Y Real $\nu$ / MHz')
                    plt.colorbar()
                    plt.subplot(414)
                    plt.imshow(plt_data.T.imag, origin='lower', extent=extent, aspect='auto')
                    plt.xlabel(r'$t$')
                    plt.ylabel(r'Y Imag $\nu$ / MHz')
                    plt.colorbar()
                    plt.savefig(outfile % ant_ind)
                else:
                    raise ValueError('Unknown plot type: %s' % plt_type)
