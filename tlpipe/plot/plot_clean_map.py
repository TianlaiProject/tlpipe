"""Plot clean image."""

import numpy as np
import h5py
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from tlpipe.utils import mpiutil
from tlpipe.core.base_exe import Base
from tlpipe.utils.path_util import input_path, output_path


# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
params_init = {
               'nprocs': mpiutil.size, # number of processes to run this module
               'aprocs': range(mpiutil.size), # list of active process rank no.
               'input_file': 'clean_image.hdf5', # str or a list of str
               'output_file': None, # None, str or a list of str
               'uv_input_file' : 'uv_image.hdf5',
               'catalog' : None,
               'fov': 10, # degree
               'cmap': 'jet',
               'vmin': None,
               'vmax': None,
               #'ra_range': [0, 360],
               #'dec_range': [-90, 90],
              }
prefix = 'pcm_'



class Plot(Base):
    """Plot clean image."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(Plot, self).__init__(parameter_file_or_dict, params_init, prefix, feedback)

    def execute(self):

        if self.params['catalog'] != None:
            catalog_file = h5py.File(input_path(self.params['catalog']), 'r')
            catalog = catalog_file['data'].attrs['obj_list']
        else:
            catalog = None

        input_file = input_path(self.params['input_file'])
        uv_input_file = input_path(self.params['uv_input_file'])
        output_file = self.params['output_file']
        fov = self.params['fov']
        #ra_min,  ra_max  = self.params['ra_range']
        #dec_min, dec_max = self.params['dec_range']
        cmap = self.params['cmap']
        vmin = self.params['vmin']
        vmax = self.params['vmax']

        if output_file is None:
            output_file = input_file.replace('.hdf5', '.png')
        else:
            output_file = output_path(output_file)

        with h5py.File(input_file, 'r') as f: 
            img = f['cimc'][...]
            ra = f.attrs['ra']
            dec = f.attrs['dec']
            d_ra = f.attrs['d_ra']
            d_dec = f.attrs['d_dec']

        with h5py.File(uv_input_file, 'r') as uvf:
            max_wl = uvf.attrs['max_lm']
            uv_fft = uvf['uv_fft'][...]


        shape = img.shape

        space = np.linspace(-max_wl, max_wl, shape[0]+1)
        space = np.arcsin(space) * 180./np.pi
        RA  = space + ra
        DEC = space + dec

        #RA  = ( np.arange(shape[0] + 1) - shape[0]//2 ) * d_ra  + ra
        #DEC = ( np.arange(shape[1] + 1) - shape[1]//2 ) * d_dec + dec 

        #RA  -= 0.5 * d_ra
        #DEC -= 0.5 * d_dec

        fig = plt.figure(figsize=(5, 5))
        ax  = fig.add_axes([0.15, 0.15, 0.65, 0.8])
        cax = fig.add_axes([0.83, 0.25, 0.03, 0.6])
        #ax.pcolormesh(RA[None, :], DEC[:, None], img.real)
        im = ax.pcolormesh(RA[None, :], DEC[:, None], uv_fft.real, cmap='Greys')
        fig.colorbar(im, ax=ax, cax=cax)

        ax.set_xlim(xmin=ra-0.5*fov, xmax=ra+0.5*fov)
        ax.set_ylim(ymin=dec-0.5*fov, ymax=dec+0.5*fov)

        ax.set_ylabel('Dec')
        ax.set_xlabel('RA')
        ax.minorticks_on()
        ax.tick_params(length=4, width=1., direction='out')
        ax.tick_params(which='minor', length=2, width=1., direction='out')
        ax.set_aspect('equal')

        cax.minorticks_on()
        cax.tick_params(length=4, width=1., direction='out')
        cax.tick_params(which='minor', length=2, width=1., direction='out')

        if catalog != None:
            #ax.scatter(catalog[:,0]*180./np.pi, catalog[:,1]*180./np.pi, 
            #        s=30, c='none', edgecolor='g', mew=0.5)
            ax.plot(catalog[:,0]*180./np.pi, catalog[:,1]*180./np.pi, 'go', 
                    mec='g', mew=0.5, mfc='none', markersize=5)

        plt.savefig(output_file.replace('.png', '_dirty.png'))

        
        vmax = None
        vmin = None
        #mean = np.mean(img.real)
        #std  = np.std(img.real)
        #print std, mean
        #vmax = mean + 30 * std
        #vmin = mean - 30 * std

        fig = plt.figure(figsize=(5, 5))
        ax  = fig.add_axes([0.15, 0.15, 0.65, 0.8])
        cax = fig.add_axes([0.83, 0.25, 0.03, 0.6])
        im = ax.pcolormesh(RA[None, :], DEC[:, None], img.real, 
                cmap='Greys', vmax=vmax, vmin=vmin)
        #im = ax.pcolormesh(RA[None, :], DEC[:, None], uv_fft.real, cmap='Greys')
        fig.colorbar(im, ax=ax, cax=cax)

        ax.set_xlim(xmin=ra-0.5*fov, xmax=ra+0.5*fov)
        ax.set_ylim(ymin=dec-0.5*fov, ymax=dec+0.5*fov)

        ax.set_ylabel('Dec')
        ax.set_xlabel('RA')
        ax.minorticks_on()
        ax.tick_params(length=4, width=1., direction='out')
        ax.tick_params(which='minor', length=2, width=1., direction='out')
        ax.set_aspect('equal')

        cax.minorticks_on()
        cax.tick_params(length=4, width=1., direction='out')
        cax.tick_params(which='minor', length=2, width=1., direction='out')

        if catalog != None:
            #ax.scatter(catalog[:,0]*180./np.pi, catalog[:,1]*180./np.pi, 
            #        s=30, c='none', edgecolor='g', mew=0.5)
            ax.plot(catalog[:,0]*180./np.pi, catalog[:,1]*180./np.pi, 'go', 
                    mec='g', mew=0.5, mfc='none', markersize=5)

        plt.savefig(output_file.replace('.png', '_clean.png'))
