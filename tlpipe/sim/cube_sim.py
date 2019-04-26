#! 
import numpy as np
import healpy as hp
import h5py
import copy
from numpy import random
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation

from tlpipe.pipeline import pipeline
from tlpipe.utils.path_util import output_path
from caput import mpiutil
from tlpipe.sim import corr21cm
from tlpipe.map import algebra as al
from tlpipe.map import mapbase
import beam
import units
#from mpi4py import MPI

from pipeline.Observatory.Receivers import Noise

class CubeSim(mapbase.MapBase, pipeline.TaskBase):

    params_init = {
            'prefix'        : 'MeerKAT3',

            'freq' : np.linspace(950, 1350, 32), 

            'mock_n' : 10,
            'scenario': 'str',
            'refinement': 2,

            'field_centre' : (12., 0.,),
            'pixel_spacing' : 0.5,
            'map_shape'     : (10, 10),
            'map_pad'       : 5,

            #'outfile_raw'     : True,
            #'outfile_physical': True,
            'outfiles' : ['raw', 'delta', 'withbeam'],

            }

    prefix = 'csim_'

    def setup(self):

        freq  = self.params['freq'] #* 1.e6
        freq_d = freq[1] - freq[0]
        freq_n = freq.shape[0]
        freq_c = freq[freq_n//2]

        self.refinement = self.params['refinement']
        self.scenario = self.params['scenario']

        field_centre = self.params['field_centre']
        spacing = self.params['pixel_spacing']
        dec_spacing = spacing
        ra_spacing  = - spacing / np.cos(field_centre[1] * np.pi / 180.)

        axis_names = ['freq', 'ra', 'dec']
        map_pad = self.params['map_pad']
        map_shp = [x + map_pad for x in self.params['map_shape']]
        map_tmp = np.zeros([freq_n, ] + map_shp)
        map_tmp = al.make_vect(map_tmp, axis_names=axis_names)
        map_tmp.set_axis_info('freq', freq_c, freq_d)
        map_tmp.set_axis_info('ra',   field_centre[0], ra_spacing)
        map_tmp.set_axis_info('dec',  field_centre[1], dec_spacing)
        self.map_tmp = map_tmp

        # here we use 300 h km/s from WiggleZ for streaming dispersion
        self.streaming_dispersion = 300.*0.72

        self.beam_data = np.array([1., 1., 1.])
        self.beam_freq = np.array([900, 1100, 1400]) #* 1.e6

        random.seed(3936650408)
        seeds = random.random_integers(100000000, 1000000000, mpiutil.size)
        self.seed = seeds[mpiutil.rank]
        print "RANK: %02d with random seed [%d]"%(mpiutil.rank, self.seed)
        random.seed(self.seed)


        self.outfiles = self.params['outfiles']
        self.open_outputfiles()

        self.iter_list = mpiutil.mpirange(self.params['mock_n'])
        self.iter      = 0
        self.iter_num  = len(self.iter_list)

    def next(self):

        if self.iter == self.iter_num:
            mpiutil.barrier()
            #self.close_outputfiles()
            super(CubeSim, self).next()

        print "rank %03d, %03d"%(mpiutil.rank, self.iter_list[self.iter])

        self.realize_simulation()
        if 'delta' in self.outfiles:
            self.make_delta_sim()
        if 'withbeam' in self.outfiles:
            self.convolve_by_beam()

        self.write_to_file()

        self.iter += 1


    def realize_simulation(self):
        """do basic handling to call Richard's simulation code
        this produces self.sim_map and self.sim_map_phys
        """
        if self.scenario == "nostr":
            print "running dd+vv and no streaming case"
            simobj = corr21cm.Corr21cm.like_kiyo_map(self.map_tmp)
            maps = simobj.get_kiyo_field_physical(refinement=self.refinement)

        else:
            if self.scenario == "str":
                print "running dd+vv and streaming simulation"
                simobj = corr21cm.Corr21cm.like_kiyo_map(self.map_tmp,
                                           sigma_v=self.streaming_dispersion)

                maps = simobj.get_kiyo_field_physical(refinement=self.refinement)

            if self.scenario == "ideal":
                print "running dd-only and no mean simulation"
                simobj = corr21cm.Corr21cm.like_kiyo_map(self.map_tmp)
                maps = simobj.get_kiyo_field_physical(
                                            refinement=self.refinement,
                                            density_only=True,
                                            no_mean=True,
                                            no_evolution=True)

        (gbtsim, gbtphys, physdim) = maps

        # process the physical-space map
        self.sim_map_phys = al.make_vect(gbtphys, axis_names=('freq', 'ra', 'dec'))
        pshp = self.sim_map_phys.shape

        # define the axes of the physical map; several alternatives are commented
        info = {}
        info['axes'] = ('freq', 'ra', 'dec')
        info['type'] = 'vect'
        info['freq_delta'] = abs(physdim[0] - physdim[1]) / float(pshp[0] - 1)
        info['freq_centre'] = physdim[0] + info['freq_delta'] * float(pshp[0] // 2)
        #        'freq_centre': abs(physdim[0] + physdim[1]) / 2.,

        info['ra_delta'] = abs(physdim[2]) / float(pshp[1] - 1)
        #info['ra_centre'] = info['ra_delta'] * float(pshp[1] // 2)
        #        'ra_centre': abs(physdim[2]) / 2.,
        info['ra_centre'] = 0.

        info['dec_delta'] = abs(physdim[3]) / float(pshp[2] - 1)
        #info['dec_centre'] = info['dec_delta'] * float(pshp[2] // 2)
        #        'dec_centre': abs(physdim[3]) / 2.,
        info['dec_centre'] = 0.

        self.sim_map_phys.info = info

        # process the map in observation coordinates
        self.sim_map = al.make_vect(gbtsim, axis_names=('freq', 'ra', 'dec'))
        self.sim_map.copy_axis_info(self.map_tmp)
        self.sim_map_raw = self.sim_map

    def make_delta_sim(self):
        r"""this produces self.sim_map_delta"""
        print "making sim in units of overdensity"
        freq_axis = self.sim_map.get_axis('freq') # / 1.e6
        z_axis = units.nu21 / freq_axis - 1.0

        simobj = corr21cm.Corr21cm()
        T_b = simobj.T_b(z_axis) * 1e-3

        self.sim_map_delta = copy.deepcopy(self.sim_map)
        self.sim_map_delta /= T_b[:, np.newaxis, np.newaxis]

    def convolve_by_beam(self):
        r"""this produces self.sim_map_withbeam"""
        print "convolving simulation by beam"
        beamobj = beam.GaussianBeam(self.beam_data, self.beam_freq)
        self.sim_map_withbeam = beamobj.apply(self.sim_map)


    def open_outputfiles(self):

        output_prefix = '/%s_cube_%s'%(self.params['prefix'], self.params['scenario'])
        output_file = output_prefix + '.h5'
        output_file = output_path(output_file, relative=True)
        self.allocate_output(output_file, 'w')

        dset_shp  = (self.params['mock_n'], ) + self.map_tmp.shape
        dset_info = {}
        dset_info['axes'] = ('mock', 'freq', 'ra', 'dec')
        dset_info['type'] = 'vect'
        dset_info['mock_delta']  = 1
        dset_info['mock_centre'] = self.params['mock_n']//2
        dset_info['freq_delta']  = self.map_tmp.info['freq_delta']
        dset_info['freq_centre'] = self.map_tmp.info['freq_centre']
        dset_info['ra_delta']    = self.map_tmp.info['ra_delta']
        dset_info['ra_centre']   = self.map_tmp.info['ra_centre']
        dset_info['dec_delta']   = self.map_tmp.info['dec_delta']
        dset_info['dec_centre']  = self.map_tmp.info['dec_centre']
        #if self.params['outfile_raw']:
        for outfile in self.outfiles:
            self.create_dataset(outfile, dset_shp, dset_info)

    def write_to_file(self):

        df = self.df

        for outfile in self.outfiles:
            df[outfile][self.iter_list[self.iter], ...] = getattr(self, 'sim_map_'+outfile)


    #def close_outputfiles(self):

    #    self.df.close()


class ScanMode(object):

    def __init__(self, site_lon, site_lat, params=None):

        self.location = EarthLocation.from_geodetic(site_lon, site_lat)
        self.alt_list = None
        self.az_list  = None
        self.t_list   = None
        self.ra_list  = None
        self.dec_list = None

        self.params = params

    def generate_altaz(self):

        pass

    def radec_list(self):

        _alt = self.alt_list
        _az  = self.az_list
        _t_list = self.t_list
        _obs_len = len(_alt)

        #radec_list = np.zeros([int(_obs_len), 2])
        #for i in mpiutil.mpirange(_obs_len): #[rank::size]:
            #print mpiutil.rank, i
        pp = SkyCoord(alt=_alt, az=_az,  frame='altaz', 
                location=self.location, obstime=_t_list)
        pp = pp.transform_to('icrs')
        #radec_list[:, 0] = pp.ra.deg
        #radec_list[:, 1] = pp.dec.deg

        #radec_list =  mpiutil.allreduce(radec_list)
        self.ra_list  = pp.ra.deg
        self.dec_list = pp.dec.deg

    @property
    def ra(self):
        return self.ra_list

    @property
    def dec(self):
        return self.dec_list

class AzDrift(ScanMode):

    def generate_altaz(self):

        obs_speed = self.params['obs_speed']
        obs_int   = self.params['obs_int']
        obs_tot   = self.params['obs_tot']
        obs_len   = int((obs_tot / obs_int).decompose().value) 

        alt_list = []
        az_list  = []
        t_list   = []
        starttime_list = self.params['starttime']
        startpointing_list = self.params['startpointing']
        for ii in range(len(starttime_list)):
            starttime = Time(starttime_list[ii])
            alt_start, az_start  = startpointing_list[ii]
            alt_start *= u.deg
            az_start  *= u.deg

            _alt_list = ((np.ones(obs_len) * alt_start)/u.deg).value
            _az_list  = (((np.arange(obs_len) * obs_speed * obs_int)\
                    + az_start)/u.deg).value

            alt_list.append(_alt_list)
            az_list.append(_az_list)
            t_list.append(np.arange(obs_len) * obs_int + starttime)

        self.alt_list = np.concatenate(alt_list) * u.deg
        self.az_list  = np.concatenate(az_list) * u.deg
        self.t_list   = Time(np.concatenate(t_list))

class HorizonRasterDrift(ScanMode):

    def generate_altaz(self):

        obs_speed = self.params['obs_speed']
        obs_int   = self.params['obs_int']
        #obs_tot   = self.params['obs_tot']
        #obs_len   = int((obs_tot / obs_int).decompose().value) 
        block_time = self.params['block_time']
        obs_len   = int((block_time / obs_int).decompose().value) 
        obs_az_range = self.params['obs_az_range']

        alt_list = []
        az_list  = []
        t_list   = []
        starttime_list = self.params['starttime']
        startpointing_list = self.params['startpointing']
        for ii in range(len(starttime_list)):
            starttime = Time(starttime_list[ii])
            alt_start, az_start  = startpointing_list[ii]
            alt_start *= u.deg
            az_start  *= u.deg

            _alt_list = ((np.ones(obs_len) * alt_start)/u.deg).value
            alt_list.append(_alt_list)

            t_list.append((np.arange(obs_len) - 0.5 * obs_len) * obs_int + starttime)

            _az_space = obs_speed * obs_int
            _one_way_npoints = (obs_az_range / obs_speed / obs_int).decompose()
            _az_list = np.arange(_one_way_npoints) - 0.5 * _one_way_npoints
            _az_list = np.append(_az_list, -_az_list)
            _az_list *= _az_space
            _az_list += az_start
            _az_list = (_az_list / u.deg).value
            _az_list = [_az_list[i%int(2.*_one_way_npoints)] for i in range(obs_len)]
            az_list.append(_az_list)

        self.alt_list = np.concatenate(alt_list) * u.deg
        self.az_list  = np.concatenate(az_list) * u.deg
        self.t_list   = Time(np.concatenate(t_list))

