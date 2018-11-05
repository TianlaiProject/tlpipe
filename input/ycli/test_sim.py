# -*- mode: python; -*-
#

import os
from astropy import units as u
import numpy as np

from tlpipe.timestream import convertion
from tlpipe.timestream import tod_noise
from tlpipe.timestream import tod_svd
from tlpipe.sim import survey_sim
from tlpipe.plot import plot_svd
from tlpipe.plot import plot_ps
from tlpipe.plot import plot_waterfall


data_base = os.getenv('DATA_BASE')

#sim_type = 'fnoise'
#sim_type = 'fnoise_beta0.9'
#sim_type = 'fnoise_beta0.0'
#sim_type = 'fnoise_beta0.1'
#sim_type = 'syn_x_fnoise'
sim_type = 'fnoise_test'

pipe_tasks = []
pipe_outdir = data_base + 'meerkat/sim_%s/'%sim_type
pipe_logging = 'info'
pipe_copy = False

data_path = data_base + '/meerkat/sim/'

#file_name = '1474580092'
bad_time_list = None 
bad_freq_list = None 

n_bins    = 20 
mode_list = [0, 1, 2, 5]
ps_method = 'lombscargle' #'fft'
prewhiten = True

input_files = [ ]

pipe_tasks.append(survey_sim.SurveySim)

ssim_prefix       = 'MeerKAT3'

#ssim_survey_mode  = 'AzDrift'
#ssim_starttime    = ['2018-09-15 21:06:00.000', ] #UTC
#ssim_startpointing= [[55.0, 180.], ] #[Alt, Az]

ssim_survey_mode  = 'HorizonRasterDrift'
ssim_starttime    = [
        '2018-09-15 21:06:00.000', 
        '2018-09-16 02:30:00.000', 
        '2018-09-16 21:00:00.000',
        '2018-09-17 02:24:00.000',] #UTC
ssim_startpointing= [
        [40.92286411431906, 57.36757999053849], 
        [39.33195768664754, 300.6946709895161],
        [40.54643878089189, 57.83808773292882],
        [39.71437947016491, 301.1485185840776]] #[Alt, Az]

ssim_obs_speed    = 5. * u.arcmin / u.second
ssim_obs_int      = 1. * u.second
ssim_obs_tot      = 1. * u.hour
ssim_block_time   = 1. * u.hour
ssim_obs_az_range = 15. * u.deg
ssim_freq         = np.linspace(950, 1350, 32)
ssim_fg_syn_model = None
ssim_HI_model     = None
ssim_fnoise       = True
ssim_noiseRatio   = True
ssim_noisePower   = 1.
ssim_noiseFreq    = 0.1
ssim_alpha        = 1.
ssim_beta         = 0.5
ssim_cutoff       = 1200000.
ssim_sim_wnoise   = False
ssim_filterscale  = 3600


