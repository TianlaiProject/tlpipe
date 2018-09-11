# -*- mode: python; -*-
#

import os

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
sim_type = 'HI_tf'

pipe_tasks = []
pipe_outdir = data_base + 'meerkat/sim_%s/'%sim_type
pipe_logging = 'info'
pipe_copy = False

data_path = data_base + '/meerkat/sim/'

#file_name = '1474580092'
bad_time_list = [[3437, 3520]]
bad_freq_list = [[2473-2190, 2548-2190], ] #[2696-2190, 2704-2190], [2900, None]]

n_bins    = 20 
mode_list = [0, 1, 2, 5]
ps_method = 'lombscargle' #'fft'
prewhiten = True

input_files = [
        'sim_meerKAT3_AZdrift_20180915210600.h5',
        'sim_meerKAT3_AZdrift_20180915230605.h5',
        'sim_meerKAT3_AZdrift_20180916010610.h5',
        'sim_meerKAT3_AZdrift_20180916030615.h5',
        'sim_meerKAT3_AZdrift_20180916050620.h5',
        'sim_meerKAT3_AZdrift_20180916070625.h5',
        'sim_meerKAT3_AZdrift_20180916090630.h5',
        'sim_meerKAT3_AZdrift_20180916110635.h5',
        'sim_meerKAT3_AZdrift_20180916130640.h5',
        'sim_meerKAT3_AZdrift_20180916150645.h5',
        'sim_meerKAT3_AZdrift_20180916170650.h5',
        'sim_meerKAT3_AZdrift_20180916190655.h5',
    ]
#input_files = input_files[:2]

method = 'time' #'freq'

#pipe_tasks.append(survey_sim.SurveySim)

pipe_tasks.append(tod_noise.DataEdit)

pipe_tasks.append(tod_svd.SVD)

#pipe_tasks.append((tod_noise.PinkNoisePS_1DTC, 'pnpsvis_'))

#pipe_tasks.append((tod_noise.PinkNoisePS_1DTC, 'pnps1dtccln_'))

#pipe_tasks.append((tod_noise.PinkNoisePS_1DFC, 'pnps1dfccln_'))

#pipe_tasks.append((tod_noise.PinkNoisePS_1DTC, 'pnpsmod_'))


# ===================================================================
# ==                         for plot                              ==
# ===================================================================

file_name = 'sim_meerKAT3_AZdrift_20180915210600'
ant = 'm017' #'m021', 'm017', 'm036'
file_middle = '_avgNoneF_tcorrps_%s'%ps_method
#file_middle = '_avgNoneT_tcorrps_%s'%ps_method
file_suffix = file_middle + '_%s_x_%s'%(ant, ant)

#pipe_tasks.append((plot_svd.PlotSVD, 'psvd_'))

#pipe_tasks.append((plot_ps.PlotPS, 'pps1dtcsvd_'))

#pipe_tasks.append((plot_ps.PlotPS, 'pps1dfcsvd_'))

#pipe_tasks.append((plot_ps.PlotPS, 'ppsmod_'))

#pipe_tasks.append((plot_ps.PlotPS, 'ppsall_'))

#pipe_tasks.append(plot_waterfall.PlotMeerKAT)


# ===================================================================
# ==             details for parameter defination                  ==
# ===================================================================

# paramerer for DataEdit
pned_input_files  = ['raw/%s'%f for f in input_files]
pned_corr = 'auto'
pned_pol_select = (0, 2) # ignore the cross pol
pned_out = 'pned'
pned_bandpass_cal = False
pned_bad_time_list = bad_time_list
pned_bad_freq_list = bad_freq_list
pned_output_files  = ['pned/%s'%f for f in input_files]

#------------------------------------------------------------------------------

# parameter for SVD
#todsvd_in = pned_out
todsvd_input_files = pned_output_files
todsvd_mode_list = mode_list
todsvd_svd_path = '/data/users/ycli/meerkat/svd/1474580580_svdmodes_m017_x_m017.h5'
todsvd_method = method
todsvd_output_files = ['svd_%s/%s'%(method, f) for f in input_files]
todsvd_prewhiten = prewhiten
#todsvd_out = 'todsvd'

#------------------------------------------------------------------------------

# parameter for Power spectrum estimation for raw vis
pnpsvis_input_files = todsvd_output_files
pnpsvis_output_files = ['pnps_1dtc/%s'%f for f in input_files]
pnpsvis_data_sets = 'vis'
pnpsvis_n_bins = n_bins
pnpsvis_method =  ps_method
#pnpsvis_avg_len = 100
pnpsvis_avg_len = None

#------------------------------------------------------------------------------

# parameter for Power spectrum estimation for cleaned vis time corr
pnps1dtccln_input_files = todsvd_output_files
pnps1dtccln_output_files = ['pnps_%s_1dtc/%s'%(method, f) for f in input_files]
pnps1dtccln_data_sets = 'cleaned_vis'
pnps1dtccln_n_bins = n_bins
pnps1dtccln_f_min  = 1.e-4
pnps1dtccln_f_max  = 10.
pnps1dtccln_method =  ps_method
#pnps1dtccln_avg_len = 100
pnps1dtccln_avg_len = None

#------------------------------------------------------------------------------

# parameter for Power spectrum estimation for cleaned vis freq corr
pnps1dfccln_input_files = todsvd_output_files
pnps1dfccln_output_files = ['pnps_1dfc/%s'%f for f in input_files]
pnps1dfccln_data_sets = 'cleaned_vis'
pnps1dfccln_n_bins = n_bins
pnps1dfccln_w_min  = None #1.e-4
pnps1dfccln_w_max  = None #10.
pnps1dfccln_method =  ps_method
pnps1dfccln_avg_len = None
#pnps1dfccln_avg_len = None

#------------------------------------------------------------------------------

# parameter for Power spectrum estimation for svd modes
pnpsmod_input_files = todsvd_output_files
pnpsmod_output_files = ['pnps_1dtc/%s'%f for f in input_files]
pnpsmod_data_sets = 'modes'
pnpsmod_n_bins = n_bins
pnpsmod_method =  ps_method
#pnpsmod_avg_len = 100
pnpsmod_avg_len = None

#------------------------------------------------------------------------------

# parameter for plot SVD modes
psvd_input_files = ['svd/%s_svdmodes_%s_x_%s.h5'%(file_name, ant, ant) ]
psvd_output_files = ['svd/%s_svdmodes_%s_x_%s'%(file_name, ant, ant) ]
psvd_mode_n = 6

#------------------------------------------------------------------------------

# parameter for plot ps with differenct svd modes subtraction
file_name_temp = 'pnps_%s_1dtc/'%method + file_name + '_vis_sub%02dmodes' + file_suffix + '.h5'
pps1dtcsvd_input_files = [file_name_temp%m for m in mode_list]
pps1dtcsvd_labels = [
    'without mode subtracted',
    'subtract the first mode',
    'subtract the first 2 modes',
    'subtract the first 5 modes',
    ]
pps1dtcsvd_label_title = 'MeerKAT Ant. %s'%ant
pps1dtcsvd_vmin = None #1.e-4 #1.e-9
pps1dtcsvd_vmax = None #5.e1 # 1.e-3
pps1dtcsvd_output_files = ['ps_%s_1dtc/%s_submodes%s'%(method, file_name, file_suffix),]

#------------------------------------------------------------------------------

# parameter for plot ps with differenct svd modes subtraction
file_name_temp = 'pnps_1dfc/' + file_name + '_vis_sub%02dmodes' + file_suffix + '.h5'
pps1dfcsvd_input_files = [file_name_temp%m for m in mode_list]
pps1dfcsvd_labels = [
    'without mode subtracted',
    'subtract the first mode',
    'subtract the first 2 modes',
    'subtract the first 5 modes',
    ]
pps1dfcsvd_label_title = 'MeerKAT Ant. %s'%ant
pps1dfcsvd_vmin = 1.e-18 #1.e-9
pps1dfcsvd_vmax = 5.e-9 # 1.e-3
pps1dfcsvd_corr_type = 'fcorr'
pps1dfcsvd_output_files = ['ps_1dfc/%s_submodes%s'%(file_name, file_suffix),]

#------------------------------------------------------------------------------

# parameter for plot ps with differenct svd modes subtraction
file_name_temp = 'pnps/' + file_name + '_modes%02d' + file_suffix + '.h5'
ppsmod_input_files = [file_name_temp%(m + 1) for m in range(4)]
ppsmod_labels = [
    'the 1st SVD mode',
    'the 2nd SVD mode',
    'the 3rd SVD mode',
    'the 4th SVD mode',
    ]
ppsmod_label_title = 'MeerKAT Ant. %s'%ant
ppsmod_vmin = 1.e-4
ppsmod_vmax = 5.e1
ppsmod_output_files = ['ps/%s_SVDmodes%s'%(file_name, file_suffix),]


#------------------------------------------------------------------------------

# parameter for plot raw ps with differenct ants
ppsall_input_files = [
    'pnps/%s_vis_avg0100_tcorrps_%s_m017_x_m017.h5'%(file_name, ps_method),
    'pnps/%s_vis_avg0100_tcorrps_%s_m021_x_m021.h5'%(file_name, ps_method),
    'pnps/%s_vis_avg0100_tcorrps_%s_m036_x_m036.h5'%(file_name, ps_method),
    ]
ppsall_labels = [
    'MeerKAT Ant. m017',
    'MeerKAT Ant. m021',
    'MeerKAT Ant. m036',
    ]
ppsall_vmin = 1.e-9
ppsall_vmax = 1.e-3
ppsall_output_files = ['ps/%s_allants_%s'%(file_name, file_middle)]


#------------------------------------------------------------------------------

# waterfall plots
#pkat_input_files = m2t_output_files
pkat_input_files = pned_output_files
#pkat_input_files = todsvd_output_files
pkat_re_scale = 2.
#pkat_fig_name = 'wf/%s_raw'%file_name
#pkat_fig_name = 'wf/%s_flagbad'%file_name
pkat_main_data = 'vis'
#pkat_main_data = 'vis_sub00modes'
pkat_fig_name = 'wf/%s'%file_name
pkat_flag_mask = True


