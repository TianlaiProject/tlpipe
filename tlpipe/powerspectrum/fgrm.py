"""Module to remove fg with SVD"""
import numpy as np
import h5py as h5
import gc
import copy
import numpy.ma as ma

from caput import mpiutil
from tlpipe.pipeline import pipeline
from tlpipe.utils.path_util import output_path
from tlpipe.map import mapbase
from tlpipe.map import algebra as al
from tlpipe.map import beam
from tlpipe.powerspectrum import find_modes


class FGRM_SVD(mapbase.MultiMapBase, pipeline.OneAndOne):
    """Module to estimate the power spectrum."""

    params_init = {
            'mode_list': [0, 1],
            'output_combined' : None,
            }

    prefix = 'fg_'

    def setup(self):

        super(FGRM_SVD, self).setup()

        input_files = self.input_files
        input_files_num = len(input_files)
        self.input_files_num = input_files_num

        self.init_task_list()


    def init_output(self):

        for output_file in self.output_files:
            output_file = output_path(output_file, 
                relative= not output_file.startswith('/'))
            self.allocate_output(output_file, 'w')
            #self.create_dataset('binavg_1d', self.dset_shp + (self.knum,))

            self.df_out[-1]['mode_list'] = self.params['mode_list']

    def combine_results(self):

        output_combined = self.params['output_combined']
        output_combined  = output_path(output_combined, 
                relative= not output_combined.startswith('/'))
        self.allocate_output(output_combined, 'w')
        self.df_out[-1]['mode_list'] = self.params['mode_list']

        for _m in self.params['mode_list']:
            mode_key = 'cleaned_%02dmode'%_m
            self.create_dataset(-1, mode_key, dset_shp = self.dset_shp,
                                dset_info = self.map_info)
            self.create_dataset(-1, mode_key + '_weight', dset_shp = self.dset_shp, 
                                dset_info = self.map_info)
            mask = np.ones(self.dset_shp[:1]).astype('bool')
            for ii in range(len(self.output_files)):
                for key in self.df_out[ii][mode_key].keys():
                    self.df_out[-1][mode_key][:]\
                            += self.df_out[ii]['weight'][:]\
                            *  self.df_out[ii]['%s/%s'%(mode_key, key)][:]
                    self.df_out[-1][mode_key + '_weight'][:]\
                            += self.df_out[ii]['weight'][:]
                mask *= ~(self.df_out[ii]['mask'][:].astype('bool'))
            weight = self.df_out[-1][mode_key + '_weight'][:]
            weight[weight==0] = np.inf
            self.df_out[-1][mode_key][:] /= weight
            self.df_out[-1][mode_key + '_mask'] = (~mask).astype('int')




    def init_task_list(self):

        self.init_output()

        with h5.File(self.input_files[0], 'r') as f:
            map_tmp = al.make_vect(al.load_h5(f, 'clean_map'))
            self.map_info = map_tmp.info

        task_list = []
        for ii in range(self.input_files_num):
            input_file_name_ii = self.input_files[ii].split('/')[-1]
            input_file_name_ii = input_file_name_ii.replace('.h5', '')
            for jj in range(ii + 1, self.input_files_num):
                input_file_name_jj = self.input_files[jj].split('/')[-1]
                input_file_name_jj = input_file_name_jj.replace('.h5', '')
                tind_l = (ii, )
                tind_r = (jj, )
                tind_o = [input_file_name_jj, input_file_name_ii]
                task_list.append([tind_l, tind_r, tind_o])

                for kk in self.params['mode_list']:
                    self.create_dataset(ii, 'cleaned_%02dmode/'%kk + input_file_name_jj, 
                            dset_shp = map_tmp.shape, dset_info = map_tmp.info)
                    self.create_dataset(jj, 'cleaned_%02dmode/'%kk + input_file_name_ii, 
                            dset_shp = map_tmp.shape, dset_info = map_tmp.info)

            self.create_dataset(ii, 'weight', dset_shp = map_tmp.shape,
                                dset_info = map_tmp.info)
            self.create_dataset(ii, 'mask', dset_shp = map_tmp.shape[:1])

        self.task_list = task_list
        self.dset_shp  = map_tmp.shape

    def read_input(self):

        input = []
        for ii in range(self.input_files_num):
            input.append(h5.File(self.input_files[ii], 'r', 
                                 driver='mpio', comm=mpiutil._comm))

        return input

    def process(self, input):

        task_list = self.task_list
        for task_ind in mpiutil.mpirange(len(task_list)):
            tind_l, tind_r, tind_o = task_list[task_ind]
            tind_l = tuple(tind_l)
            tind_r = tuple(tind_r)
            tind_o = tind_o
            print ("RANK %03d est. ps.\n(" + "%03d,"*len(tind_l) + ") x ("\
                    + "%03d,"*len(tind_r) + ")\n")%((mpiutil.rank, ) + tind_l + tind_r)

            tind_list = [tind_l, tind_r]
            maps    = []
            weights = []
            freq_mask = np.ones(self.dset_shp[0]).astype('bool')
            for i in range(2):
                tind = tind_list[i]

                map_key = 'clean_map'
                input_map = input[tind[0]][map_key][:]
                input_map = al.make_vect(input_map, axis_names = ['freq', 'ra', 'dec'])
                for key in input_map.info['axes']:
                    input_map.set_axis_info(key,
                                            self.map_info[key+'_centre'],
                                            self.map_info[key+'_delta'])
                maps.append(input_map)

                weight_key = 'noise_diag'
                weight = input[tind[0]][weight_key][:]
                weight = make_noise_factorizable(weight)

                weight = al.make_vect(weight, axis_names = ['freq', 'ra', 'dec'])
                weight.info = input_map.info

                freq_mask *= ~(input[tind[0]]['mask'][:]).astype('bool')

                weights.append(weight)

            maps, weights = degrade_resolution(maps, weights, conv_factor=1.,
                    mode='nearest')
            freq_cov, counts = find_modes.freq_covariance(maps[0], maps[1], 
                    weights[0], weights[1], freq_mask, freq_mask)
            svd_info = find_modes.get_freq_svd_modes(freq_cov, np.sum(freq_mask))

            mode_list_ed = self.params['mode_list']
            mode_list_st = copy.deepcopy(self.params['mode_list'])
            mode_list_st[1:] = mode_list_st[:-1]

            dset_key = tind_o[0] + '_sigvalu'
            self.df_out[tind_l[0]][dset_key] = svd_info[0]
            dset_key = tind_o[0] + '_sigvect'
            self.df_out[tind_l[0]][dset_key] = svd_info[1]
            self.df_out[tind_l[0]]['weight'][:] = weights[0]
            self.df_out[tind_l[0]]['mask'][:]   = (~freq_mask).astype('int')


            dset_key = tind_o[1] + '_sigvalu'
            self.df_out[tind_r[0]][dset_key] = svd_info[0]
            dset_key = tind_o[1] + '_sigvect'
            self.df_out[tind_r[0]][dset_key] = svd_info[2]
            self.df_out[tind_r[0]]['weight'][:] = weights[1]
            self.df_out[tind_r[0]]['mask'][:]   = (~freq_mask).astype('int')

            for (n_modes_st, n_modes_ed) in zip(mode_list_st, mode_list_ed):
                svd_modes = svd_info[1][n_modes_st:n_modes_ed]
                maps[0], amp = find_modes.subtract_frequency_modes( maps[0], svd_modes, 
                                                                    weights[0], freq_mask)
                dset_key = 'cleaned_%02dmode/'%n_modes_ed + tind_o[0]
                self.df_out[tind_l[0]][dset_key][:] = copy.deepcopy(maps[0])

                svd_modes = svd_info[2][n_modes_st:n_modes_ed]
                maps[1], amp = find_modes.subtract_frequency_modes( maps[1], svd_modes, 
                                                                    weights[1], freq_mask)
                dset_key = 'cleaned_%02dmode/'%n_modes_ed + tind_o[1]
                self.df_out[tind_r[0]][dset_key][:] = copy.deepcopy(maps[1])


        if self.params['output_combined'] is not None:
            self.combine_results()

        for ii in range(self.input_files_num):
            input[ii].close()


    def finish(self):
        #if mpiutil.rank0:
        print 'RANK %03d Finishing Ps.'%(mpiutil.rank)

        mpiutil.barrier()
        for df in self.df_out:
            df.close()

def noise_diag_2_weight(weight):

    weight[weight==0] = np.inf
    weight = 1./weight
    
    mask = (weight != 0).astype('int')
    spat_w = np.sum(weight, axis=0)
    norm = np.sum(mask, axis=0) * 1.
    norm[norm==0] = np.inf
    spat_w /= norm
    
    freq_w = np.sum(weight, axis=(1, 2))
    norm = np.sum(mask, axis=(1, 2)) * 1.
    norm[norm==0] = np.inf
    freq_w /= norm
    
    weight = freq_w[:, None, None] * spat_w[None, :, :]
    weight = weight ** 0.5
    cut  = np.percentile(weight, 10)
    cut2 = np.percentile(weight, 80)
    #weight[weight>cut] = cut
    weight[weight<cut] = 0.

    weight[weight>cut2] = cut2

    return weight

def make_noise_factorizable(noise, weight_prior=2):
    r"""Convert noise diag such that the factor into a function a
    frequency times a function of pixel by taking means over the original
    weights.
    
    input noise_diag;
    output weight
    
    weight_prior used to be 10^-30 before prior applied
    """
    print "making the noise factorizable"
    
    #noise[noise < weight_prior] = 1.e-30
    #noise = 1. / noise
    noise[noise > 1./weight_prior] = 1.e30
    noise = ma.array(noise)
    # Get the freqency averaged noise per pixel.  Propagate mask in any
    # frequency to all frequencies.
    for noise_index in range(ma.shape(noise)[0]):
        if np.all(noise[noise_index, ...] > 1.e20):
            noise[noise_index, ...] = ma.masked
    noise_fmean = ma.mean(noise, 0)
    noise_fmean[noise_fmean > 1.e20] = ma.masked
    # Get the pixel averaged noise in each frequency.
    noise[noise > 1.e20] = ma.masked
    noise /= noise_fmean
    noise_pmean = ma.mean(ma.mean(noise, 1), 1)
    # Combine.
    noise = noise_pmean[:, None, None] * noise_fmean[None, :, :]
    noise[noise == 0] = np.inf
    weight = (1. / noise).filled(0)

    cut_l  = np.percentile(weight, 10)
    cut_h = np.percentile(weight, 80)
    weight[weight<cut_l] = cut_l
    weight[weight>cut_h] = cut_h

    return weight

def degrade_resolution(maps, noises, conv_factor=1.2, mode="constant", fwhm1400=0.9):
    r"""Convolves the maps down to the lowest resolution.

    Also convolves the noise, making sure to deweight pixels near the edge
    as well.  Converts noise to factorizable form by averaging.

    mode is the ndimage.convolve flag for behavior at the edge
    """
    print "degrading the resolution to a common beam: ", conv_factor
    noise1, noise2 = noises
    map1, map2 = maps

    # Get the beam data.
    freq_data = np.linspace(800., 1600., 50).astype('float')
    beam_data = fwhm1400 * 1400. / freq_data
    beam_diff = np.sqrt(max(conv_factor * beam_data) ** 2 - (beam_data) ** 2)
    common_resolution = beam.GaussianBeam(beam_diff, freq_data)
    # Convolve to a common resolution.
    map2 = common_resolution.apply(map2)
    map1 = common_resolution.apply(map1)

    # This block of code needs to be split off into a function and applied
    # twice (so we are sure to do the same thing to each).
    #good = np.isfinite(noise1)
    #noise1[~good] = 0.
    #noise1[noise1 == 0] = np.inf # 1.e-30
    #noise1 = 1. / noise1
    noise1 = common_resolution.apply(noise1, mode=mode, cval=0)
    noise1 = common_resolution.apply(noise1, mode=mode, cval=0)
    #noise1[noise1 == 0] = np.inf
    #noise1[noise1 < 1.e-5] = np.inf
    #noise1 = 1. / noise1
    #noise1[noise1 < 1.e-20] = 0.

    #good = np.isfinite(noise2)
    #noise2[~good] = 0.
    #noise2[noise2 == 0] = np.inf # 1.e-30
    #noise2 = 1 / noise2
    noise2 = common_resolution.apply(noise2, mode=mode, cval=0)
    noise2 = common_resolution.apply(noise2, mode=mode, cval=0)
    #noise2[noise2 == 0] = np.inf # 1.e-30
    #noise2[noise2 < 1.e-5] = np.inf
    #noise2 = 1. / noise2
    #noise2[noise2 < 1.e-20] = 0.

    #noise_inv1 = algebra.as_alg_like(noise1, self.noise_inv1)
    #noise_inv2 = algebra.as_alg_like(noise2, self.noise_inv2)

    return [map1, map2], [noise1, noise2]
