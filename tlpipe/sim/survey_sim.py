#! 
import numpy as np
import healpy as hp
import h5py
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation

from tlpipe.pipeline import pipeline
from tlpipe.utils.path_util import output_path
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
            'prefix'        : 'MeerKAT3',
            'survey_mode'   : 'AzDrift',
            'starttime'     : ['2018-09-15 21:06:00.000',], #UTC
            'startpointing' : [[55.0, 180.], ],#[Alt, Az]

            'obs_speed' : 2. * u.deg / u.second,
            'obs_int'   : 100. * u.second,
            'obs_tot'   : 5. * u.hour,
            'obs_az_range' : 15. * u.deg, # for HorizonRasterDrift

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

    prefix = 'ssim_'

    def setup(self):

        freq  = self.params['freq']
        dfreq = freq[1] - freq[0]
        freq_n = freq.shape[0]

        self.SM = globals()[self.params['survey_mode']](ant_Lon, ant_Lat, self.params)
        #self.SM.generate_altaz(startalt, startaz, starttime, obs_len, obs_speed, obs_int)
        self.SM.generate_altaz()
        self.SM.radec_list()


        #starttime = Time(self.params['starttime'])
        #startalt, startaz  = self.params['startpointing']
        #startalt *= u.deg
        #startaz  *= u.deg
        #obs_speed = self.params['obs_speed']
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
            self.FN = Noise.FNoise(
                    self.params['noiseRatio'], 
                    self.params['noisePower'],
                    self.params['noiseFreq'],
                    self.params['alpha'],
                    self.params['cutoff'],
                    self.params['beta'],
                    samplerate,
                    dfreq,
                    freq_n,
                    self.block_len,
                    'scipy',       #fftMode=simInfo.Parameters['FNoise']['fftLib'],
                    'numpy',       #random=simInfo.Parameters['FNoise']['randomLib'],
                    'mt19937',     #GSL_RNG_TYPE=simInfo.Parameters['FNoise']['gsl_rng_type'],
                    'FFTW_MEASURE',#planner=simInfo.Parameters['FNoise']['planner'],
                    1,             #threads = simInfo.Parameters['FNoise']['threads'],
                    self.params['sim_wnoise'],
                    self.params['filterscale'],
                    )

        
        self.get_blorder()

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

        block_n = self.block_len
        idx_st   = self.iter_list[self.iter]
        idx_ed   = idx_st + block_n
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

        rvis = np.empty([block_n, freq_n, 2, len(ants)])
        for i in range(len(ants)):
            if self.params['fnoise']:
                rvis[..., 0, i] = _syn * (self.FN.Realisation(freq_n, block_n).T + 1.)
                rvis[..., 1, i] = _syn * (self.FN.Realisation(freq_n, block_n).T + 1.)

        shp = (block_n, freq_n, 4, len(self.blorder))
        vis = np.empty(shp, dtype=np.complex)
        vis[:, :, 0, self.auto_idx] = rvis[:, :, 0, :] + 0. * 1j
        vis[:, :, 1, self.auto_idx] = rvis[:, :, 1, :] + 0. * 1j


        self.write_to_file(vis)

        self.iter += 1

    def get_blorder(self):

        feedno = []
        channo = []
        feedpos = []
        for ii in range(len(ants)):
            ant = ants[ii]
            antno = int(ant[1:]) + 1
            feedno.append(antno)
            channo.append([2 * antno - 1, 2 * antno])
            feedpos.append(ants_pos[ii])

        feedno = np.array(feedno)
        channo = np.array(channo)
        feedpos = np.array(feedpos)

        antn   = len(feedno)

        blorder = [[feedno[i], feedno[j]] for i in range(antn) for j in range(i, antn)]
        auto_idx = [blorder.index([feedno[i], feedno[i]]) for i in range(antn)]

        self.blorder = blorder
        self.auto_idx = auto_idx
        self.feedno = feedno
        self.channo = channo
        self.feedpos = feedpos


    def write_to_file(self, vis=None):

        block_n = self.block_len
        idx_st   = self.iter_list[self.iter]
        idx_ed   = idx_st + block_n
        t_list   = self.SM.t_list[idx_st:idx_ed]
        ra_list  = self.SM.ra[idx_st:idx_ed]
        dec_list = self.SM.dec[idx_st:idx_ed]


        output_prefix = '/sim/%s_%s'%(self.params['prefix'], self.params['survey_mode'])
        output_file = output_prefix + '_%s.h5'%t_list[0].datetime.strftime('%Y%m%d%H%M%S')
        output_file = output_path(output_file, relative=True)
        print mpiutil.rank, output_file
        with h5py.File(output_file, 'w') as df:
            df.attrs['nickname'] = output_prefix
            df.attrs['comment'] = 'just a simulation'
            df.attrs['observer'] = 'Robot'
            history = 'Here is the beginning of the history'
            df.attrs['history'] = history
            df.attrs['keywordver'] = '0.0' # Keyword version.

            # Type B Keywords
            df.attrs['sitename'] = 'MeerKAT'
            df.attrs['sitelat'] = self.SM.location.lat.deg #-(30. + 42./60. + 47.41/3600.)
            df.attrs['sitelon'] = self.SM.location.lon.deg #  21. + 26./60. + 38.00/3600. 
            df.attrs['siteelev'] = 1000.0    # Not precise
            df.attrs['timezone'] = 'UTC+02'  # 
            df.attrs['epoch'] = 2000.0  # year

            df.attrs['telescope'] = 'MeerKAT-Dish-I' # 
            df.attrs['dishdiam'] = 13.5
            df.attrs['nants'] = 3
            df.attrs['npols'] = 2
            df.attrs['cylen'] = -1 # For dish: -1
            df.attrs['cywid'] = -1 # For dish: -1

            df.attrs['recvver'] = '0.0'    # Receiver version.
            df.attrs['lofreq'] = 935.0  # MHz; Local Oscillator Frequency.

            df.attrs['corrver'] = '0.0'    # Correlator version.
            df.attrs['samplingbits'] = 8 # ADC sampling bits.
            df.attrs['corrmode'] = 1 # 2, 3

            obstime = '%s'%t_list[0].isot
            obstime = obstime.replace('-', '/')
            obs_int   = self.params['obs_int']
            inttime = (obs_int / u.second).value
            df.attrs['inttime'] = inttime
            df.attrs['obstime'] = obstime
            #df.attrs['sec1970'] = _t_list[0].unix

            df['sec1970'] = t_list.unix
            df['sec1970'].attrs['dimname'] = 'Time,'
            df['jul_date'] = t_list.jd
            df['jul_date'].attrs['dimname'] = 'Time,'

            df['ra']  = ra_list
            df['ra'].attrs['dimname'] = 'Time,'

            df['dec'] = dec_list
            df['dec'].attrs['dimname'] = 'Time,'

            freq = self.params['freq']
            df.attrs['nfreq'] = freq.shape[0] # Number of Frequency Points
            df.attrs['freqstart'] = freq[0] # MHz; Frequency starts.
            df.attrs['freqstep'] = freq[1] - freq[0] # MHz; Frequency step.

            # Data Array
            #df.create_dataset('vis', chunks = (10, 1024, 1, 4), data=vis,
            df.create_dataset('vis', data=vis, dtype = vis.dtype, shape = vis.shape)
            df['vis'].attrs['dimname'] = 'Time, Frequency, Polarization, Baseline'

            df['pol'] = np.array(['hh', 'vv', 'hv', 'vh'])
            df['pol'].attrs['pol_type'] = 'linear'
            
            df['feedno'] = self.feedno
            df['channo'] = self.channo
            df['channo'].attrs['dimname'] = 'Feed No., (HPolarization VPolarization)'
            
            df['blorder'] = self.blorder
            df['blorder'].attrs['dimname'] = 'Baselines, BaselineName'

            
            df['feedpos'] = self.feedpos
            df['feedpos'].attrs['dimname'] = 'Feed No., (X,Y,Z) coordinate' ###
            df['feedpos'].attrs['unit'] = 'm'
            
            #df['antpointing'] = antpointing(16)
            #df['antpointing'].attrs['dimname'] = 'Feed No., (Az,Alt,AzErr,AltErr)'
            #df['antpointing'].attrs['unit'] = 'degree'


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

