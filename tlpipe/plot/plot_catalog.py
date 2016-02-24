""" Plot Catalog """

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
               'input_file': 'uv_image.hdf5',  # str or a list of str
               'output_file': '', # None, str or a list of str
               'ra_range' : [320, 359],
               'dec_range': [50, 65],
              }
prefix = 'pltcat_'



class Plot_Catalog(Base):
    """Plot image."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(Plot_Catalog, self).__init__(parameter_file_or_dict, params_init, prefix, feedback)

    def execute(self):

        input_file = input_path(self.params['input_file'])
        output_file = output_path(self.params['output_file'])

        data = h5py.File(input_file, 'r')

        obj_list = data['data'].attrs['obj_list']

        arg_sort = np.argsort(obj_list[:,-1])
        obj_list = obj_list[arg_sort, :]
        #obj_list = obj_list[::-1, :]

        fig = plt.figure(figsize=(7, 5))
        ax1 = fig.add_axes([0.12, 0.12, 0.75, 0.75], axisbg='none')

        cmap = plt.get_cmap('Blues')
        #norm = plt.normalize(
        #        np.log10(obj_list[:,-1].min()), np.log10(obj_list[:,-1].max()))
        norm = plt.normalize(obj_list[:,-1].min(), obj_list[:,-1].max())

        ax1.scatter(obj_list[:,0]*180./np.pi, obj_list[:,1]*180./np.pi, s=60,
                c=cmap(norm(obj_list[:,-1])), 
                edgecolor='k', alpha=0.5)

        ax1.set_xlim(xmin=self.params['ra_range'][0], xmax=self.params['ra_range'][1])
        ax1.set_ylim(ymin=self.params['dec_range'][0], ymax=self.params['dec_range'][1])

        ax1.set_ylabel('Dec')
        ax1.set_xlabel('RA')
        ax1.minorticks_on()
        ax1.tick_params(length=4, width=1., direction='out')
        ax1.tick_params(which='minor', length=2, width=1., direction='out')
        ax1.set_aspect('equal')

        plt.savefig(output_file + '_Cat.png', format='png')

        plt.show()
