"""Phase the source-phased visibility data to the zenith, this is just the inverse operation of phs2src_exe.py."""

try:
    import cPickle as pickle
except ImportError:
    import pickle

import os
import numpy as np
from scipy.linalg import eigh
import h5py
import ephem
import aipy as a

from tlpipe.utils import mpiutil
from tlpipe.core.base_exe import Base
from tlpipe.core import tldishes
from tlpipe.utils.pickle_util import get_value
from tlpipe.utils.date_util import get_ephdate
from tlpipe.utils.path_util import input_path, output_path


# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
params_init = {
               'nprocs': mpiutil.size, # number of processes to run this module
               'aprocs': range(mpiutil.size), # list of active process rank no.
               'input_file': 'data_phs2src.hdf5',
               'output_file': 'data_phs2zen.hdf5',
               'source': 'cas', # <src_name> or <ra XX[:XX:xx]>_<dec XX[:XX:xx]> or <time y/m/d h:m:s> (array pointing of this local time)
               'catalog': 'misc,helm,nvss',
               'extra_history': '',
              }
prefix = 'p2z_'


class Phs2zen(Base):
    """Phase the source-phased visibility data to the zenith, this is just the inverse operation of phs2src_exe.py."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(Phs2zen, self).__init__(parameter_file_or_dict, params_init, prefix, feedback)

    def execute(self):

        input_file = input_path(self.params['input_file'])
        output_file = output_path(self.params['output_file'])
        source = self.params['source']
        catalog = self.params['catalog']

        with h5py.File(input_file, 'r') as f:
            dset = f['data']
            data_shp = dset.shape
            data_type = dset.dtype
            ants = dset.attrs['ants']
            ts = f['time']
            freq = dset.attrs['freq']
            az = np.radians(dset.attrs['az_alt'][0][0])
            alt = np.radians(dset.attrs['az_alt'][0][1])
            time_zone = dset.attrs['timezone']

            npol = dset.shape[2]
            nt = len(ts)
            nfreq = len(freq)
            nants = len(ants)
            bls = [(ants[i], ants[j]) for i in range(nants) for j in range(i, nants)] # start from 1
            nbls = len(bls)

            lt, st, et = mpiutil.split_local(nt)
            # data an time local to this process
            local_data = dset[st:et]
            local_ts = ts[st:et]

        if mpiutil.rank0:
            data_phs2zen = np.zeros(data_shp, dtype=data_type) # save data phased to src
        else:
            data_phs2zen = None


        # array
        aa = tldishes.get_aa(1.0e-3 * freq) # use GHz
        # make all antennas point to the pointing direction
        for ai in aa:
            ai.set_pointing(az=az, alt=alt, twist=0)

        try:
            # convert an observing time to the ra_dec of the array pointing of that time
            src_time = get_ephdate(source, tzone=time_zone) # utc time
            aa.date = str(ephem.Date(src_time)) # utc time
            # print 'date:', aa.date
            az, alt = ephem.degrees(az), ephem.degrees(alt)
            src_ra, src_dec = aa.radec_of(az, alt)
            source = '%s_%s' % (src_ra, src_dec)
            # print 'source:', source
        except ValueError:
            pass

        # source
        srclist, cutoff, catalogs = a.scripting.parse_srcs(source, catalog)
        cat = a.src.get_catalog(srclist, cutoff, catalogs)
        assert(len(cat) == 1), 'Allow only one source'
        s = cat.values()[0]
        if mpiutil.rank0:
            print 'Undo the source-phase %s to phase to the zenith.' % source


        for ti, t in enumerate(local_ts): # mpi among time
            aa.set_jultime(t)
            s.compute(aa)
            # get fluxes vs. freq of the calibrator
            # Sc = s.get_jys()
            # get the topocentric coordinate of the calibrator at the current time
            s_top = s.get_crds('top', ncrd=3)
            # aa.sim_cache(cat.get_crds('eq', ncrd=3)) # for compute bm_response and sim
            for bi, (ai, aj) in enumerate(bls):
                uij = aa.gen_uvw(ai-1, aj-1, src='z').squeeze() # (rj - ri)/lambda
                # phase the data to src
                local_data[ti, bi, :, :] *= np.exp(-2.0J * np.pi * np.dot(s_top, uij))
                # local_data[ti, bi, :, :] *= np.exp(2.0J * np.pi * np.dot(s_top, uij))

        # Gather data in separate processes
        mpiutil.gather_local(data_phs2zen, local_data, (st, 0, 0, 0), root=0, comm=self.comm)

        # save data after phased to src
        if mpiutil.rank0:
            with h5py.File(output_file, 'w') as f:
                dset = f.create_dataset('data', data=data_phs2zen)
                # copy metadata from input file
                with h5py.File(input_file, 'r') as fin:
                    f.create_dataset('time', data=fin['time'])
                    for attrs_name, attrs_value in fin['data'].attrs.iteritems():
                        dset.attrs[attrs_name] = attrs_value
                # update some attrs
                dset.attrs['history'] = dset.attrs['history'] + self.history
