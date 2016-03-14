"""Plot visibility."""

import itertools
import numpy as np
import h5py
import matplotlib.pyplot as plt

from tlpipe.utils import mpiutil
from tlpipe.core.base_exe import Base
from tlpipe.utils.pickle_util import get_value
from tlpipe.utils.path_util import input_path, output_path


# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
params_init = {
               'nprocs': mpiutil.size, # number of processes to run this module
               'aprocs': range(mpiutil.size), # list of active process rank no.
               'input_file': 'data.hdf5',
               'output_file': None, # None ork str
               'bl_index': None, # None or list, None for all baselines
               'pol_index': [0, 1, 2, 3],
               'cut': [None, None],
               'vmin': None,
               'vmax': None,
              }
prefix = 'pltv_'



class Plot(Base):
    """Plot visibility."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(Plot, self).__init__(parameter_file_or_dict, params_init, prefix, feedback)

    def execute(self):

        input_file = input_path(self.params['input_file'])
        output_file = self.params['output_file']
        bl_index = self.params['bl_index']
        pol_index = self.params['pol_index']
        cut = self.params['cut']
        vmin = self.params['vmin']
        vmax = self.params['vmax']

        if output_file is None:
            output_file = output_path('vis.png')
        else:
            output_file = output_path(output_file)
        suffix = output_file.split('.')[-1]

        with h5py.File(input_file, 'r') as f:
            dset = f['data']
            nt = dset.shape[0]
            ants = get_value(dset.attrs['ants'])
            nant = len(ants)
            bls = [(ants[i], ants[j]) for i in range(nant) for j in range(i, nant)]
            nbl  = len(bls)
            st = int(cut[0] * nt) if cut[0] is not None else None
            et = int(cut[1] * nt) if cut[1] is not None else None
            ts = f['time'][st:et]
            freq = dset.attrs['freq']
            extent = [freq[0], freq[-1], ts[0], ts[-1]]
            if bl_index is None:
                bl_index = range(nbl)
            lst = list(itertools.product(bl_index, pol_index))
            for (bi, pi) in mpiutil.mpilist(lst):
                outfile = output_file.replace('.'+suffix, '_%d_%d_%d.'+suffix)
                plt_data = dset[st:et, bi, pi, :]
                plt.figure(figsize=(13, 8))
                plt.subplot(121)
                plt.imshow(plt_data.real, origin='lower', aspect='auto', extent=extent, vmin=vmin, vmax=vmax)
                plt.xlabel(r'$\nu$ / MHz')
                plt.ylabel(r'$t$ / Julian Date')
                plt.colorbar()
                plt.subplot(122)
                plt.imshow(plt_data.imag, origin='lower', aspect='auto', extent=extent, vmin=vmin, vmax=vmax)
                plt.xlabel(r'$\nu$ / MHz')
                plt.ylabel(r'$t$ / Julian Date')
                plt.colorbar()
                plt.savefig(outfile % (bls[bi][0], bls[bi][1], pi))
