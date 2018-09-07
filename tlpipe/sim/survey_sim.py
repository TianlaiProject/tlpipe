#! 
import numpy as np
import healpy as hp
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation

from tlpipe.pipeline import pipeline
from caput import mpiutil
#from mpi4py import MPI

from pipeline.Observatory.Receivers import Noise

ants = ['m017', 'm021', 'm036']
ants_pos = [
        [ 200., 123., 0.],
        [-296., -93., 0.],
        [ 388., -57., 0.],
        ]

ant_Lon =  (21. + 26./60. + 37.69/3600.) * u.deg 
ant_Lat = -(30. + 42./60. + 46.53/3600.) * u.deg  

class SurveySim(pipeline.TaskBase):

    params_init = {
            'survey_mode'   : 'AzDrift',
            'starttime'     : '2018-09-15 21:06:00.000', #UTC
            'startpointing' : [55.0, 180.], #[Alt, Az]

            'obs_speed' : 2. * u.deg / u.second,
            'obs_int'   : 100. * u.second,
            'obs_tot'   : 5. * u.hour,

            'block_time' : 1. * u.hour,

            'freq' : np.linspace(950, 1350, 32), 

            'fg_syn_model' : None,
            'HI_model' : None,

            'fnoise'   : True,
            'noiseRatio' : True,
            'noisePower' : 1.,
            'noiseFreq'  : 0.1, 
            'alpha'      : 1.,
            'beta'       : 0.5,
            'cutoff'     : 1200000.,
            'sim_wnoise' : False,
            'filterscale': 3600,
            }
    def setup(self):

        freq  = self.params['freq']
        dfreq = freq[1] - freq[0]
        freq_n = freq.shape[0]

        starttime = Time(self.params['starttime'])
        startalt, startaz  = self.params['startpointing']
        startalt *= u.deg
        startaz  *= u.deg
        obs_speed = self.params['obs_speed']
        obs_int   = self.params['obs_int']
        obs_tot   = self.params['obs_tot']
        obs_len   = int((obs_tot / obs_int).decompose().value) 
        samplerate = ((1./obs_int/u.Hz).decompose()).value

        block_time = self.params['block_time']
        self.block_len  = int((block_time / obs_int).decompose().value)
        block_num  = obs_tot / self.block_len


        if self.params['fg_syn_model'] is not None:
            self.syn_model = hp.read_map(
                    self.params['fg_syn_model'], range(freq.shape[0]))
            self.syn_model = self.syn_model.T

        if self.params['fnoise']:
            self.FNoiseObject = Noise.FNoise(
                    self.params['noiseRatio'], 
                    self.params['noisePower'],
                    self.params['noiseFreq'],
                    self.params['alpha'],
                    self.params['cutoff'],
                    self.params['beta'],
                    samplerate,
                    dfreq,
                    freq_n,
                    block_len,
                    'scipy',       #fftMode=simInfo.Parameters['FNoise']['fftLib'],
                    'numpy',       #random=simInfo.Parameters['FNoise']['randomLib'],
                    'mt19937',     #GSL_RNG_TYPE=simInfo.Parameters['FNoise']['gsl_rng_type'],
                    'FFTW_MEASURE',#planner=simInfo.Parameters['FNoise']['planner'],
                    1,             #threads = simInfo.Parameters['FNoise']['threads'],
                    self.params['sim_wnoise'],
                    self.params['filterscale'],
                    )

        
        self.SM = globals()[self.params['survey_mode']](ant_Lon, ant_Lat)
        self.SM.generate_altaz(startalt, startaz, starttime, obs_len, obs_speed, obs_int)
        self.SM.radec_list()

        self.iter_list =  mpiutil.mpirange(0, obs_len, self.block_len)
        self.iter = 0
        self.iter_num = len(self.iter_list)

    def next(self):

        if self.iter == self.iter_num:
            mpiutil.barrier()
            super(SurveySim, self).next()

        freq   = self.params['freq']
        dfreq  = freq[1] - freq[0]
        freq_n = freq.shape[0]

        idx_st   = self.iter_list[self.iter]
        idx_ed   = idx_st + self.block_len
        t_list   = self.SM.t_list[idx_st:idx_ed]
        ra_list  = self.SM.ra[idx_st:idx_ed]
        dec_list = self.SM.dec[idx_st:idx_ed]

        print mpiutil.rank, t_list[0]

        if self.params['fg_syn_model'] is not None:
            syn_model_nside = hp.npix2nside(self.syn_model.shape[0])
            _idx_pix = hp.ang2pix(syn_model_nside, ra_list, dec_list, lonlat=True)
            _syn = self.syn_model[_idx_pix, :]
        else:
            _syn = 1.

        rvis = np.empty(block_n, freq_n, 2, len(ants))
        for i in range(len(ants)):
            if self.params['fnoise']:
                rvis[..., 0, i] = _syn * (self.FNoiseObject.Realisation(freq_n, block_n).T + 1.)
                rvis[..., 1, i] = _syn * (self.FNoiseObject.Realisation(freq_n, block_n).T + 1.)


        self.iter += 1

class ScanMode(object):

    def __init__(self, site_lon, site_lat):

        self.location = EarthLocation.from_geodetic(site_lon, site_lat)
        self.alt_list = None
        self.az_list  = None
        self.t_list   = None
        self.ra_list  = None
        self.dec_list = None

    def generate_altaz(self):

        pass

    def radec_list(self):

        _alt = self.alt_list
        _az  = self.az_list
        _t_list = self.t_list
        _obs_len = len(_alt)

        radec_list = np.zeros([int(_obs_len), 2])
        for i in mpiutil.mpirange(_obs_len): #[rank::size]:
            #print mpiutil.rank, i
            pp = SkyCoord(alt=_alt[i], az=_az[i],  frame='altaz', 
                    location=self.location,
                    obstime=_t_list[i]).transform_to('fk5')
            radec_list[i, 0] = pp.ra.deg
            radec_list[i, 1] = pp.dec.deg

        radec_list =  mpiutil.allreduce(radec_list)
        self.ra_list  = radec_list[:,0]
        self.dec_list = radec_list[:,1]

class AzDrift(ScanMode):

    def generate_altaz(self,  alt_start, az_start, starttime, obs_len, obs_speed, obs_int):

        self.alt_list = np.ones(obs_len) * alt_start
        self.az_list  = (np.arange(obs_len) * obs_speed * obs_int) + az_start
        self.t_list   = np.arange(obs_len) * obs_int + starttime


