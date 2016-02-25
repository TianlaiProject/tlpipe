"""Plot celestial sources on lm-plane."""

import numpy as np
import h5py
import aipy as a
import matplotlib.pyplot as plt

from tlpipe.utils import mpiutil
from tlpipe.core.base_exe import Base
from tlpipe.core import tldishes
from tlpipe.utils.date_util import get_juldate
from tlpipe.utils.path_util import input_path, output_path
from tlpipe.timestream.gridding_exe import get_uvvec


# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
params_init = {
               'nprocs': mpiutil.size, # number of processes to run this module
               'aprocs': range(mpiutil.size), # list of active process rank no.
               'output_file': 'nvss_sources.png',
               'obs_time': '2016/01/03 22:06:59',
               'time_zone': 'UTC+08',
               'phase_center': 'cas', # <src_name> or <ra XX[:XX:xx]>_<dec XX[:XX:xx]>
               'catalog': 'nvss',
               'flux': 1.0, # Jy
               'frequency': 750, # MHz
               'lm_range': [-0.5, 0.5],
              }
prefix = 'plts_'



class Plot(Base):
    """Plot celestial sources on lm-plane."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(Plot, self).__init__(parameter_file_or_dict, params_init, prefix, feedback)

    def execute(self):

        output_file = self.params['output_file']
        if output_file is not None:
            output_file = output_path(output_file)
        else:
            output_file = output_path('nvss_sources.png')
        obs_time = self.params['obs_time']
        time_zone = self.params['time_zone']
        phase_center = self.params['phase_center']
        catalog = self.params['catalog']
        flux = self.params['flux']
        frequency = self.params['frequency']
        lm_range = self.params['lm_range']

        if mpiutil.rank0:
            # array
            aa = tldishes.get_aa(1.0e-3 * np.array(frequency)) # use GHz
            # make all antennas point to the pointing direction
            # for ai in aa:
            #     ai.set_pointing(az=az, alt=alt, twist=0)
            t = get_juldate(obs_time, time_zone)
            aa.set_jultime(t)

            # phase center
            srclist, cutoff, catalogs = a.scripting.parse_srcs(phase_center, 'misc,helm,nvss')
            cat = a.src.get_catalog(srclist, cutoff, catalogs)
            assert(len(cat) == 1), 'Allow only one phase center'
            pc = cat.values()[0] # the phase center
            if mpiutil.rank0:
                print 'Plot sources relative to phase center %s.' % phase_center
            pc.compute(aa)
            # get the topocentric coordinate of the phase center at the current time
            pc_top = pc.get_crds('top', ncrd=3)

            src = '%f/%f' % (flux, frequency / 1.0e3)
            srclist, cutoff, catalogs = a.scripting.parse_srcs(src, catalog)
            cat = a.src.get_catalog(srclist, cutoff, catalogs)
            nsrc = len(cat) # number of sources in cat
            srcs = [ cat.values()[i] for i in range(nsrc) ]
            for i in range(nsrc):
                srcs[i].compute(aa)
            # get the topocentric coordinate of each src in cat at the current time
            srcs_top = [ s.get_crds('top', ncrd=3) for s in srcs ]

            # the north celestial pole
            NP = a.phs.RadioFixedBody(0.0, np.pi/2.0, name='north pole', epoch=str(aa.epoch))
            # get the topocentric coordinate of the north celestial pole at the current time
            NP.compute(aa)
            n_top = NP.get_crds('top', ncrd=3)

            # unit vector in u,v direction in topocentric coordinate at current time relative to the phase center
            uvec, vvec = get_uvvec(pc_top, n_top)

            # l,m of srcs in cat relative to phase center
            ls = [ np.dot(src_top, uvec) for src_top in srcs_top ]
            ms = [ np.dot(src_top, vvec) for src_top in srcs_top ]

            # flux of each src in cat
            jys = [ s.get_jys() for s in srcs ]

            # plot
            plt.figure()
            plt.scatter(ls, ms, s=jys, c=jys, alpha=0.5)
            plt.xlabel(r'$l$')
            plt.ylabel(r'$m$')
            plt.xlim(lm_range)
            plt.ylim(lm_range)
            plt.savefig(output_file)