"""Plot celestial sources on lm-plane using simple sin-projection."""

import numpy as np
import h5py
import aipy as a
import matplotlib.pyplot as plt

from tlpipe.utils import mpiutil
from tlpipe.core.base_exe import Base
from tlpipe.utils.path_util import input_path, output_path


# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
params_init = {
               'nprocs': mpiutil.size, # number of processes to run this module
               'aprocs': range(mpiutil.size), # list of active process rank no.
               'output_file': 'nvss_sources.png',
               'phase_center': 'cas', # <src_name> or <ra XX[:XX:xx]>_<dec XX[:XX:xx]> or <time y/m/d h:m:s> (array pointing of this local time)
               'catalog': 'nvss',
               'flux': 1.0, # Jy
               'frequency': 750, # MHz
               'lm_range': [-0.5, 0.5],
              }
prefix = 'pltss_'



class Plot(Base):
    """Plot celestial sources on lm-plane using simple sin-projection."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(Plot, self).__init__(parameter_file_or_dict, params_init, prefix, feedback)

    def execute(self):

        output_file = self.params['output_file']
        if output_file is not None:
            output_file = output_path(output_file)
        else:
            output_file = output_path('nvss_sources.png')
        phase_center = self.params['phase_center']
        catalog = self.params['catalog']
        flux = self.params['flux']
        frequency = self.params['frequency']
        lm_range = self.params['lm_range']

        if mpiutil.rank0:
            # phase center
            srclist, cutoff, catalogs = a.scripting.parse_srcs(phase_center, 'misc,helm,nvss')
            cat = a.src.get_catalog(srclist, cutoff, catalogs)
            assert(len(cat) == 1), 'Allow only one phase center'
            pc = cat.values()[0] # the phase center
            pc_ra, pc_dec = pc._ra, pc._dec # in radians
            if mpiutil.rank0:
                print 'Plot sources relative to phase center %s.' % phase_center

            src = '%f/%f' % (flux, frequency / 1.0e3)
            srclist, cutoff, catalogs = a.scripting.parse_srcs(src, catalog)
            cat = a.src.get_catalog(srclist, cutoff, catalogs)
            nsrc = len(cat) # number of sources in cat
            ras = [cat.values()[i]._ra for i in range(nsrc)]
            decs = [cat.values()[i]._dec for i in range(nsrc)]
            jys = [cat.values()[i].get_jys() for i in range(nsrc)]

            ls = [ -np.sin(ra - pc_ra) for ra in ras]
            ms = [ np.sin(dec - pc_dec) for dec in decs]

            # plot
            plt.figure()
            plt.scatter(ls, ms, s=jys, c=jys, alpha=0.5)
            plt.xlabel(r'$l$')
            plt.ylabel(r'$m$')
            plt.xlim(lm_range)
            plt.ylim(lm_range)
            plt.savefig(output_file)
