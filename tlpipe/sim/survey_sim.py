#! 
import numpy as np
import healpy as hp
import h5py
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation

from tlpipe.map import algebra as al
from tlpipe.pipeline import pipeline
from tlpipe.utils.path_util import output_path
from caput import mpiutil
#from mpi4py import MPI

import gc

from pipeline.Observatory.Receivers import Noise

#ants = ['m017', 'm021', 'm036']
#ants_pos = [
#        [ 200., 123., 0.],
#        [-296., -93., 0.],
#        [ 388., -57., 0.],
#        ]

ant_dat = np.genfromtxt('/users/ycli/code/tlpipe/tlpipe/sim/data/meerKAT.dat', 
        dtype=[('name', 'S4'), ('X', 'f8'), ('Y', 'f8'), ('Z', 'f8')])

ants = ant_dat['name'] 

ants_pos = [ant_dat['X'][:, None], ant_dat['Y'][:, None], ant_dat['Z'][:, None]]
ants_pos = np.concatenate(ants_pos, axis=1)


ants = ants[:60]
ants_pos = ants_pos[:60, :]


ant_Lon =  (21. + 26./60. + 37.69/3600.) * u.deg 
ant_Lat = -(30. + 42./60. + 46.53/3600.) * u.deg  

class SurveySim(pipeline.TaskBase):

    params_init = {
            'prefix'        : 'MeerKAT3',
            'survey_mode'   : 'AzDrift',
            'schedule_file' : None,
            #'starttime'     : ['2018-09-15 21:06:00.000',], #UTC
            #'startpointing' : [[55.0, 180.], ],#[Alt, Az]

            #'obs_speed' : 2. * u.deg / u.second,
            #'obs_int'   : 100. * u.second,
            #'obs_tot'   : 5. * u.hour,
            #'obs_az_range' : 15. * u.deg, # for HorizonRasterDrift

            ##'beam_size' : 1. * u.deg, # FWHM

            #'block_time' : 1. * u.hour,

            'T_rec' : 25., # K

            'freq' : np.linspace(950, 1350, 32), 

            'fg_syn_model' : None,
            'HI_model' : None,
            'HI_model_type' : 'delta', # withbeam, raw, delta
            'HI_scenario'   : 'ideal',

            'mock_n'   : 10,

            'fnoise'   : True,
            #'noiseRatio' : True,
            #'noisePower' : 1.,
            'noiseFreq'  : 0.1, 
            'alpha'      : 1.,
            'beta'       : 0.5,
            'cutoff'     : 1200000.,
            #'sim_wnoise' : False,
            'filterscale': 3600,
            }

    prefix = 'ssim_'

    def setup(self):

        freq  = self.params['freq']
        dfreq = freq[1] - freq[0]
        freq_n = freq.shape[0]

        self.SM = globals()[self.params['survey_mode']](self.params['schedule_file'])
        #self.SM.generate_altaz(startalt, startaz, starttime, obs_len, obs_speed, obs_int)
        self.SM.generate_altaz()
        self.SM.radec_list()


        #starttime = Time(self.params['starttime'])
        #startalt, startaz  = self.params['startpointing']
        #startalt *= u.deg
        #startaz  *= u.deg
        #obs_speed = self.params['obs_speed']

        obs_int   = self.SM.obs_int #self.params['obs_int']
        self.obs_int = obs_int
        samplerate = ((1./obs_int).to(u.Hz)).value

        #obs_tot   = self.SM.obs_tot # self.params['obs_tot']
        #obs_len   = int((obs_tot / obs_int).decompose().value) 

        self.block_time = self.SM.sche['block_time'] #self.params['block_time']
        #self.block_len  = int((block_time / obs_int).decompose().value)
        max_block_len = int(np.max(self.block_time) / obs_int.to(u.s).value)
        block_num  = self.block_time.shape[0]


        _obs_int = (obs_int.to(u.second)).value
        self._RMS = self.params['T_rec'] / np.sqrt(_obs_int * dfreq * 1.e6)

        if self.params['fg_syn_model'] is not None:
            self.syn_model = hp.read_map(
                    self.params['fg_syn_model'], range(freq.shape[0]))
            self.syn_model = self.syn_model.T

        if self.params['HI_model'] is not None:
            #self.HI_model = hp.read_map(
            #        self.params['HI_model'], range(freq.shape[0]))
            #self.HI_model = self.HI_model.T

            #print "use fake frquency for testing cosmic var"
            #self.HI_model = hp.read_map( self.params['HI_model'], 0)
            #self.HI_model = self.HI_model[:, None] * np.ones(freq.shape[0])[None, :]
            #SEED = 3936650408
            #np.random.seed(SEED)
            #dt = np.random.rand(freq.shape[0]) * 360.
            #dp = (np.random.rand(freq.shape[0]) - 0.5) * 180.
            #npixl = self.HI_model.shape[0]
            #nside = hp.npix2nside(npixl)
            #t, p = hp.pix2ang(nside, np.arange(npixl), lonlat=True)
            #for i in range(1, freq.shape[0]):
            #    r = hp.Rotator(deg=True, rot=[dt[i], dp[i]])
            #    t_rot, p_rot = r(t, p, lonlat=True)
            #    self.HI_model[:, i] = hp.get_interp_val(self.HI_model[:, i], 
            #            t_rot, p_rot, lonlat=True)
            with h5py.File(self.params['HI_model'], 'r') as fhi:
                self.HI_model = al.make_vect(al.load_h5(fhi, self.params['HI_model_type']))
                self.mock_n = self.HI_model.shape[0]
        else:
            self.mock_n = self.params['mock_n']

        if self.params['fnoise']:
            self.FN = Noise.FNoise(
                    ratio  = False,               # if true, dG is the ratio of FN to WN
                    dG     = 1. / (dfreq * 1.e6), # rms fluctuations at f_k, multply T_sys later
                    dGFreq = self.params['noiseFreq'], # f_k
                    alpha  = self.params['alpha'],     # alpha
                    cutoff = self.params['cutoff'],
                    beta   = self.params['beta'],      # freq corr factor
                    sampleRate = samplerate,
                    dv     = dfreq,
                    _nFreq = freq_n,
                    _nSamples = max_block_len,
                    filterScale = self.params['filterscale'],
                    fftMode = 'scipy',
                    random  = 'numpy',
                    GSL_RNG_TYPE = 'mt19937', 
                    threads = 1,
                    planner = 'FFTW_MEASURE',
                    whiteNoise = False,
                    )

        
        self.get_blorder()

        #self.iter_list =  mpiutil.mpirange(0, obs_len, self.block_len)
        self.iter_list =  mpiutil.mpirange(0, block_num)
        self.iter = 0
        self.iter_num = len(self.iter_list)

    def next(self):

        if self.iter == self.iter_num:
            mpiutil.barrier()
            super(SurveySim, self).next()

        mock_n = self.mock_n

        freq   = self.params['freq']
        dfreq  = freq[1] - freq[0]
        freq_n = freq.shape[0]

        block_time = self.block_time[:self.iter+1]

        block_n = int(block_time[-1] / self.obs_int.to(u.s).value)
        idx_st   = int(np.sum(block_time[:-1]) / self.obs_int.to(u.s).value)
        idx_ed   = idx_st + block_n
        t_list   = self.SM.t_list[idx_st:idx_ed]
        ra_list  = self.SM.ra[idx_st:idx_ed]
        dec_list = self.SM.dec[idx_st:idx_ed]

        print mpiutil.rank, t_list[0]

        _sky = np.zeros((mock_n, block_n, freq_n)) + self.params['T_rec']

        if self.params['fg_syn_model'] is not None:
            print "add syn"
            syn_model_nside = hp.npix2nside(self.syn_model.shape[0])
            _idx_pix = hp.ang2pix(syn_model_nside, ra_list, dec_list, lonlat=True)
            _sky += self.syn_model[_idx_pix, :][None, ...]

        if self.params['HI_model'] is not None:
            print "add HI"
            #HI_model_nside = hp.npix2nside(self.HI_model.shape[0])
            #_idx_pix = hp.ang2pix(HI_model_nside, ra_list, dec_list, lonlat=True)
            #_HI = self.HI_model[_idx_pix, :]
            HI_model_ra  = self.HI_model.get_axis('ra')
            HI_model_dec = self.HI_model.get_axis('dec')
            on_map_indx  = (ra_list > min(HI_model_ra)) * (ra_list < max(HI_model_ra))\
                    * (dec_list > min(HI_model_dec)) * (dec_list < max(HI_model_dec))
            #_HI = np.zeros(self.HI_model.shape[:2] + (block_n, ))
            #_HI = np.zeros((mock_n, block_n, freq_n))
            for jj in range(block_n):
                if on_map_indx[jj]:
                    _sky[:, jj, :] += self.HI_model.slice_interpolate([2, 3], 
                            [ra_list[jj], dec_list[jj]], kind='linear')
                    #_HI[:, jj, :] = 

        rvis = np.empty([mock_n, block_n, freq_n, 2, len(ants)])
        if self.params['fnoise']:
            print "add 1/f"
            for i in range(len(ants)):
                for j in range(mock_n):
                    rvis[j,...,0,i] = _sky[j,...] * (self.FN.Realisation(freq_n, block_n).T + 1.)
                    rvis[j,...,1,i] = _sky[j, ...]* (self.FN.Realisation(freq_n, block_n).T + 1.)
        else:
            print "no 1/f"
            rvis = _sky[..., None, None]

        del _sky
        gc.collect()


        WN = self._RMS * np.random.randn(mock_n, block_n, freq_n, 2, len(ants))
        print "    %f K(%f K^2)" % (np.std(WN), np.var(WN))
        rvis = rvis + WN

        del WN
        gc.collect()

        #shp = (mock_n, block_n, freq_n, 4, len(self.blorder))
        #vis = np.empty(shp, dtype=np.complex)
        #vis[:, :, :, 0, self.auto_idx] = rvis[:, :, :, 0, :] + 0. * 1j
        #vis[:, :, :, 1, self.auto_idx] = rvis[:, :, :, 1, :] + 0. * 1j

        for ii in range(mock_n):

            #shp = (block_n, freq_n, 4, len(self.blorder))
            #vis = np.empty(shp, dtype=np.complex)
            #vis[:, :, 0, self.auto_idx] = rvis[ii, :, :, 0, :] + 0. * 1j
            #vis[:, :, 1, self.auto_idx] = rvis[ii, :, :, 1, :] + 0. * 1j

            output_prefix = '/sim_mock%03d/%s_%s_%s_%s'%(
                    ii, self.params['prefix'], self.params['survey_mode'], 
                    self.params['HI_scenario'], self.params['HI_model_type'])
            #self.write_to_file(rvis[ii, ...] + 0. * 1j, output_prefix=output_prefix)
            self.write_to_file(rvis[ii, ...], output_prefix=output_prefix)
            #del vis
            #gc.collect()

        del rvis
        gc.collect()


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

        #blorder = [[feedno[i], feedno[j]] for i in range(antn) for j in range(i, antn)]
        blorder = [[feedno[i], feedno[i]] for i in range(antn)]
        auto_idx = [blorder.index([feedno[i], feedno[i]]) for i in range(antn)]

        self.blorder = blorder
        self.auto_idx = auto_idx
        self.feedno = feedno
        self.channo = channo
        self.feedpos = feedpos


    def write_to_file(self, vis=None, output_prefix='sim'):

        block_time = self.block_time[:self.iter+1]

        block_n = int(block_time[-1] / self.obs_int.to(u.s).value)
        idx_st   = int(np.sum(block_time[:-1]) / self.obs_int.to(u.s).value)
        idx_ed   = idx_st + block_n

        #block_n  = self.block_len
        #idx_st   = self.iter_list[self.iter]
        #idx_ed   = idx_st + block_n
        t_list   = self.SM.t_list[idx_st:idx_ed]
        ra_list  = self.SM.ra[idx_st:idx_ed]
        dec_list = self.SM.dec[idx_st:idx_ed]


        #output_prefix = '/sim/%s_%s'%(self.params['prefix'], self.params['survey_mode'])
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
            obs_int = self.obs_int
            inttime = obs_int.to(u.second).value
            df.attrs['inttime'] = inttime
            df.attrs['obstime'] = obstime
            #df.attrs['sec1970'] = _t_list[0].unix

            df['sec1970'] = t_list.unix
            df['sec1970'].attrs['dimname'] = 'Time,'
            df['jul_date'] = t_list.jd
            df['jul_date'].attrs['dimname'] = 'Time,'

            df['ra']  = ra_list[:, None]  * np.ones(len(self.blorder))[None, :]
            df['ra'].attrs['dimname'] = 'Time, BaseLines'

            df['dec'] = dec_list[:, None] * np.ones(len(self.blorder))[None, :]
            df['dec'].attrs['dimname'] = 'Time, BaseLines'

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

    def __init__(self, schedule_file):

        self.read_schedule(schedule_file)

        self.location = EarthLocation.from_geodetic(self.site_lon, self.site_lat)
        self.alt_list = None
        self.az_list  = None
        self.t_list   = None
        self.ra_list  = None
        self.dec_list = None

        #self.params = params

    def read_schedule(self, schedule_file):

        with open(schedule_file) as f:
            for l in f.readlines():
                l = l.split()
                if l[0] != '#': continue
                if l[1] == 'Log.Lat.':
                    self.site_lon = float(l[2]) * u.deg
                    self.site_lat = float(l[3]) * u.deg
                if l[1] == 'AZRange':
                    self.obs_az_range = float(l[2]) * getattr(u, l[3])
                if l[1] == 'ScanSpeed':
                    self.obs_speed = float(l[2]) * getattr(u, l[3]) / getattr(u, l[5])
                if l[1] == 'Int.Time':
                    self.obs_int = float(l[2]) * getattr(u, l[3])
                if l[1] == 'SlewTime':
                    self.obs_slow = float(l[2]) * getattr(u, l[3])

        self.sche = np.genfromtxt(schedule_file,
                     names = ['scan', 'UTC', 'LST', 'AZ', 'Alt', 'block_time'],
                     dtype = ['S1', 'S23', 'f8', 'f8', 'f8', 'f8'],
                     delimiter=', ')


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

        obs_speed    = self.obs_speed #self.params['obs_speed']
        obs_int      = self.obs_int #self.params['obs_int']
        block_time   = self.sche['block_time'] * u.s #self.params['block_time']
        obs_az_range = self.obs_az_range #self.params['obs_az_range']

        alt_list = []
        az_list  = []
        t_list   = []
        starttime_list = self.sche['UTC'] #self.params['starttime']
        #print starttime_list
        #startpointing_list = self.params['startpointing']
        start_az_list = self.sche['AZ'] * u.deg
        start_alt_list = self.sche['Alt'] * u.deg
        for ii in range(len(starttime_list)):
            #print starttime_list[ii]
            starttime = Time(starttime_list[ii])
            alt_start = start_alt_list[ii]
            az_start  = start_az_list[ii]

            obs_len   = int((block_time[ii] / obs_int).decompose().value) 

            _alt_list = (np.ones(obs_len) * alt_start).value
            alt_list.append(_alt_list)

            #t_list.append((np.arange(obs_len) - 0.5 * obs_len) * obs_int + starttime)
            t_list.append(np.arange(obs_len) * obs_int + starttime)

            _az_space = (obs_speed * obs_int).to(u.deg)
            _one_way_npoints = (obs_az_range / obs_speed / obs_int).decompose()
            #_az_list = np.arange(_one_way_npoints) - 0.5 * _one_way_npoints
            #_az_list = np.append(_az_list, -_az_list)
            _az_list = np.arange(_one_way_npoints)
            _az_list = np.append(_az_list, _az_list[::-1])
            _az_list = _az_list * _az_space
            _az_list += az_start
            _az_list = _az_list.value
            _az_list = [_az_list[i%int(2.*_one_way_npoints)] for i in range(obs_len)]
            az_list.append(_az_list)

        self.alt_list = np.concatenate(alt_list) * u.deg
        self.az_list  = np.concatenate(az_list) * u.deg
        self.t_list   = Time(np.concatenate(t_list))

