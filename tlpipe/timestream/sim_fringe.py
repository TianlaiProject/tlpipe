"""Convert data to 4 dimentianal (time, bl, pol, freq)."""

try:
    import cPickle as pickle
except ImportError:
    import pickle

import os
import numpy as np
import h5py
import ephem
import pyfits
import aipy

from tlpipe.utils import mpiutil
from tlpipe.core.base_exe import Base
from tlpipe.utils.pickle_util import get_value
from tlpipe.utils.date_util import get_ephdate
from tlpipe.utils.date_util import get_juldate
from tlpipe.utils.path_util import input_path, output_path
from tlpipe.core import tldishes


# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
params_init = {
               'nprocs': mpiutil.size, # number of processes to run this module
               'aprocs': range(mpiutil.size), # list of active process rank no.
               'temp_file': 'temp.hdf5',
               'output_file': 'sim_file1.hdf5',
               'exclude_ant': [],         # a list of ants to exclude
               'ant_diameter': 6,
               'extra_history': '',
               'duration' : 2*60*60.,       # obervation time
               'int_time' : 10.,            # integration time
               'source'   : [['23:24:11.93', '58:54:16.7', 100], ],

               #'freq0' : 685.,            # freq of bin 0, in MHz
               #'freq_delta': 0.2441,      # freq delta
               #'ra_center': 351.0497083,  # target ra, will roll to center
               ##'source'   : ['23:23:26.0000', '58:48:0.0000'],
               #'phs_ref'  : ['0:0:0.0', '64:54:16.7']
              }
prefix = 'simf_'



class Sim(Base):
    """Convert data."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(Sim, self).__init__(parameter_file_or_dict, params_init, prefix, feedback)

    def execute(self):

        print '-'*10

        temp_file = input_path(self.params['temp_file'])
        output_file = output_path(self.params['output_file'])

        # load the temp file
        print "Load the temp file", temp_file
        tmp = h5py.File(temp_file, 'r')
        ants_list = tmp['data'].attrs['ants']
        print "Antenna used: ", ants_list

        #time_axis  = tmp['time'].value
        freq_axis  = tmp['data'].attrs['freq']
        pols_list  = tmp['data'].attrs['pol']

        # load transit time
        transit_time = tmp['data'].attrs["transit_time"][0]
        transit_time = get_ephdate(transit_time, tzone='UTC+00')
        t_n = int(self.params['duration'] / self.params['int_time'])
        time_axis  = np.arange(t_n).astype('float') - t_n//2
        time_axis *= self.params['int_time'] * ephem.second
        time_axis += transit_time
        time_axis  = np.array([get_juldate(x) for x in time_axis])

        # load pointing direction
        pointing = tmp['data'].attrs["az_alt"][0]
        # load cite information

        tl = tldishes.get_aa(freq_axis*1.e-3)
        tl.set_active_pol(pols_list[0])

        #t_n = len(time_axis)
        f_n = len(freq_axis)
        a_n = len(ants_list)
        b_n = a_n * (a_n + 1) / 2
        data_sim = np.zeros([t_n, b_n, f_n]) + 1.J * np.zeros([t_n, b_n, f_n])

        for src_crd in self.params['source']:

            if len(src_crd) == 3:
                data = src_crd[2]*np.ones([t_n, b_n, f_n]) + 1.J*np.zeros([t_n, b_n, f_n])
                src_crd = src_crd[0] + '_' + src_crd[1]
                srclist, cutoff, catalogs = aipy.scripting.parse_srcs(src_crd, 'misc')
                obj = aipy.src.get_catalog(srclist, cutoff, catalogs).values()[0]
            elif len(src_crd) == 2:
                data = src_crd[1]*np.ones([t_n, b_n, f_n]) + 1.J*np.zeros([t_n, b_n, f_n])
                srclist, cutoff, catalogs = aipy.scripting.parse_srcs(src_crd[0], 'misc')
                obj = aipy.src.get_catalog(srclist, cutoff, catalogs).values()[0]


            for ti, t in enumerate(time_axis):
                tl.set_jultime(t)
                obj.compute(tl)
                gain = self.get_gain(
                        pointing*np.pi/180., obj.az, obj.alt, freq_axis*1.e-3)
                bi = 0
                for i in ants_list:
                    for j in ants_list:
                        if i > j: continue
                        #data[ti, bi, :] *= tl.gen_phs(obj, i-1, j-1).conj()
                        data[ti, bi, :] *= tl.gen_phs(obj, i-1, j-1)
                        data[ti, bi, :] *= gain
                        bi += 1
            data_sim += data

        data = data_sim[:, :, None, :] * np.array([1,1,0,0])[None, None, :, None]
        #print data.real.max(), data.real.min()

        fout = h5py.File(output_file, 'w')
        fout.create_dataset('time', data=time_axis)
        data = fout.create_dataset('data', data=data)
        data.attrs['axes'] = ['time', 'bls', 'pol', 'freq']
        data.attrs['pol'] = pols_list
        data.attrs['ants'] = ants_list
        data.attrs['freq'] = freq_axis
        data.attrs['history'] = self.history
        data.attrs['transit_time'] = transit_time 
        data.attrs['az_alt'] = [pointing,]
        data.attrs['int_time'] = time_axis[1] - time_axis[0]

        fout.close()

    def get_gain(self, pointing, source_az, source_alt, freq_axis):
        ''' pointing is antenna direction by azimuth and altitude
        '''

        pointing_az = pointing[0]
        pointing_alt= pointing[1]

        delta_y = source_alt - pointing_alt
        delta_az= source_az - pointing_az
        delta_x = 2.*np.arcsin(np.cos(source_alt)*np.sin(0.5*delta_az))

        delta = np.sqrt( delta_x**2 + delta_y**2 )

        beam  = aipy.phs.Beam(freq_axis)
        wavelenght = aipy.const.c / 100. / (freq_axis*1.e9)
        d_ill = np.pi * self.params['ant_diameter'] * 0.9 / wavelenght
        beam_pattern = lambda x: ( np.sin( d_ill * x ) / ( d_ill * x ) ) ** 2

        return beam_pattern(delta)

if __name__=="__main__":

    pass
