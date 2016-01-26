"""Plot clean image."""

import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
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
               'fov': 10, # degree
               'cmap': 'jet',
               'vmin': None,
               'vmax': None,
              }
prefix = 'pltc_'



class Plot(Base):
    """Plot clean image."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(Plot, self).__init__(parameter_file_or_dict, params_init, prefix, feedback)

    def execute(self):

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


        xshp, yshp = img.shape
        nx = min(xshp, fov / np.abs(d_ra))
        ny = min(yshp, fov / np.abs(d_dec))
        img = img[(xshp - nx)/2:(xshp + nx)/2, (yshp - ny)/2:(yshp + ny)/2]

        xpx, ypx = img.shape
        dx1 = -(xpx/2 + .5) * np.radians(d_ra)
        dx2 = (xpx/2 - .5) * np.radians(d_ra)
        dy1 = -(ypx/2 + .5) * np.radians(d_dec)
        dy2 = (ypx/2 - .5) * np.radians(d_dec)

        plt.figure(figsize=(8, 6))
        map = Basemap(projection='ortho', lon_0=ra, lat_0=dec, rsphere=1, llcrnrx=dx1, llcrnry=dy1, urcrnrx=dx2, urcrnry=dy2)
        map.drawmeridians(np.arange(0, 360, 2), latmax=90.0, labels=[0, 0, 0, 1])
        map.drawparallels(np.arange(-90, 90, 2), labels=[1, 0, 0, 0])
        map.drawmapboundary()
        map.imshow(img.real, vmin=vmin, vmax=vmax, cmap=cmap, interpolation='nearest')
        plt.colorbar()
        plt.savefig(output_file)
