""" Plot Fringe """

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
              }
prefix = 'pltf_'



class Plot(Base):
    """Plot image."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(Plot, self).__init__(parameter_file_or_dict, params_init, prefix, feedback)

    def execute(self):

        input_file = input_path(self.params['input_file'])
        output_file = output_path(self.params['output_file'])

        data = h5py.File(input_file, 'r')

        vis_list = data['data']
        ant_list = vis_list.attrs['ants']

        time = data['time'].value
        freq = vis_list.attrs['freq']

        time0 = int(time.min())
        time -= time0

        indx = 0
        for i in ant_list:
            for j in ant_list:

                if i > j: continue

                vis = vis_list[:,indx,0,:]
                ant_pair = '%02d_%02d'%(i, j)


                fig = plt.figure(figsize=(7, 5))
                ax1 = fig.add_axes([0.12, 0.53, 0.75, 0.40])
                ax2 = fig.add_axes([0.12, 0.10, 0.75, 0.40])
                cax1 = fig.add_axes([0.875, 0.55, 0.02, 0.34])
                cax2 = fig.add_axes([0.875, 0.13, 0.02, 0.34])


                X, Y = np.meshgrid(time, freq)
                #im1 = ax1.pcolormesh(X, Y, np.abs(vis).T)
                #im2 = ax2.pcolormesh(X, Y, np.angle(vis).T)
                im1 = ax1.pcolormesh(X, Y, vis.real.T)
                im2 = ax2.pcolormesh(X, Y, vis.imag.T)

                fig.colorbar(im1, cax=cax1)
                fig.colorbar(im2, cax=cax2)
                cax1.minorticks_on()
                cax1.tick_params(length=2, width=1, direction='out')
                cax1.tick_params(which='minor', length=1, width=1, direction='out')
                cax2.minorticks_on()
                cax2.tick_params(length=2, width=1, direction='out')
                cax2.tick_params(which='minor', length=1, width=1, direction='out')


                ax1.set_title('Antenna Pair ' + ant_pair)
    
                ax1.set_xticklabels([])
                ax1.set_ylabel('Freq Real')
                #ax1.set_ylabel('Freq Amp')
                ax1.minorticks_on()
                ax1.set_xlim(xmin=time.min(), xmax=time.max())
                ax1.set_ylim(ymin=freq.min(), ymax=freq.max())
                ax1.tick_params(length=4, width=1., direction='out')
                ax1.tick_params(which='minor', length=2, width=1., direction='out')
    
                
                ax2.set_ylabel('Freq Imag')
                #ax2.set_ylabel('Freq Pha')
                ax2.set_xlabel('Time + %f [JD]'%time0)
                ax2.minorticks_on()
                ax2.set_xlim(xmin=time.min(), xmax=time.max())
                ax2.set_ylim(ymin=freq.min(), ymax=freq.max())
                ax2.tick_params(length=4, width=1., direction='out')
                ax2.tick_params(which='minor', length=2, width=1., direction='out')

                plt.savefig(output_file + '_AntPir' + ant_pair + '.png', format='png')

                plt.show()

                indx += 1




