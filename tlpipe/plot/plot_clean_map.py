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
               'catalog' : None,
               'fov': 10, # degree
               'cmap': 'jet',
               'vmin': None,
               'vmax': None,
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
        output_file = self.params['output_file']
        fov = self.params['fov']
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

        shape = img.shape
        RA  = ( np.arange(shape[0] + 1) - shape[0]//2 ) * d_ra  + ra
        DEC = ( np.arange(shape[1] + 1) - shape[1]//2 ) * d_dec + dec 

        RA  -= 0.5 * d_ra
        DEC -= 0.5 * d_dec

        fig = plt.figure(figsize=(5, 5))
        ax  = fig.add_axes([0.15, 0.15, 0.8, 0.8])
        ax.pcolormesh(RA[None, :], DEC[:, None], img.real)
        ax.set_xlim(xmin=ra-0.5*fov, xmax=ra+0.5*fov)
        ax.set_ylim(ymin=dec-0.5*fov, ymax=dec+0.5*fov)

        ax.set_ylabel('Dec')
        ax.set_xlabel('RA')
        ax.minorticks_on()
        ax.tick_params(length=4, width=1., direction='out')
        ax.tick_params(which='minor', length=2, width=1., direction='out')
        ax.set_aspect('equal')

        if catalog != None:
            ax.scatter(catalog[:,0]*180./np.pi, catalog[:,1]*180./np.pi, 
                    s=60, c='none', edgecolor='w')

        plt.savefig(output_file)

        #xshp, yshp = img.shape
        #nx = min(xshp, fov / np.abs(d_ra))
        #ny = min(yshp, fov / np.abs(d_dec))
        #img = img[(xshp - nx)/2:(xshp + nx)/2, (yshp - ny)/2:(yshp + ny)/2]

        #xpx, ypx = img.shape
        #dx1 = -(xpx/2 + .5) * np.radians(d_ra)
        #dx2 = (xpx/2 - .5) * np.radians(d_ra)
        #dy1 = -(ypx/2 + .5) * np.radians(d_dec)
        #dy2 = (ypx/2 - .5) * np.radians(d_dec)

        #plt.figure(figsize=(8, 6))
        #map = Basemap(projection='ortho', lon_0=ra, lat_0=dec, rsphere=1, llcrnrx=dx1, llcrnry=dy1, urcrnrx=dx2, urcrnry=dy2)
        #map.drawmeridians(np.arange(0, 360, 2), latmax=90.0, labels=[0, 0, 0, 1])
        #map.drawparallels(np.arange(-90, 90, 2), labels=[1, 0, 0, 0])
        #map.drawmapboundary()
        #map.imshow(img.real, vmin=vmin, vmax=vmax, cmap=cmap, interpolation='nearest')
        #plt.colorbar()
