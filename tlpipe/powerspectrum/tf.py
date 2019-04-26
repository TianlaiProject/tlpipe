"""Module to estimate the power spectrum."""
import numpy as np
import h5py as h5
import gc
import copy

from caput import mpiutil
from tlpipe.pipeline import pipeline
from tlpipe.utils.path_util import output_path
from tlpipe.map import mapbase
from tlpipe.map import algebra as al
from tlpipe.map import physical_gridding as gridding
from tlpipe.powerspectrum import pwrspec_estimator as pse
from tlpipe.powerspectrum import binning
#from tlpipe.powerspectrum import power_spectrum

class FG_clean(mapbase.MapBase, pipeline.OneAndOne):

    prefix = 'fgc_'

    params_init = {
            #'prefix'   : 'MeerKAT3',
            'map_key'  : 'clean_map',
            'svd_file' : '/users/ycli/data/ska/svd_mode.hdf5',
            'svd_keys' : 'svd_mode_eor_p_noise_p_fg_10',
            'svd_mode' : [10, ],
            }

    def setup(self):

        super(FG_clean, self).setup()

        input_files = self.input_files
        input_files_num = len(input_files)
        self.input_files_num = input_files_num

        self.init_task_list()

        self.init_output()

    def init_output(self):

        output_file = self.output_files[0]
        output_file = output_path(output_file, 
                relative= not output_file.startswith('/'))
        self.allocate_output(output_file, 'w')
        for i in self.params['svd_mode']:
            self.create_dataset('clean_map_cln%02dm'%(i), self.map_shp, self.map_info)
        for i in range(max(self.params['svd_mode'])):
            self.create_dataset('the_%02dm'%(i+1), self.map_shp, self.map_info)

    def init_task_list(self):

        #print self.input_files
        with h5.File(self.input_files[0], 'r') as f:
            map_tmp = al.make_vect(al.load_h5(f, self.params['map_key']))
            self.map_shp = map_tmp.shape
            #ant_n, pol_n = map_tmp.shape[:2]
            self.map_info = map_tmp.info

        self.task_list = []

    def read_input(self):

        input = []
        for ii in range(self.input_files_num):
            input.append(h5.File(self.input_files[ii], 'r', 
                                 driver='mpio', comm=mpiutil._comm))
            #input = al.make_vect(al.load_h5(f, 'clean_map'))

        with h5.File(self.params['svd_file'], 'r') as f:
            print self.params['svd_keys']
            self.modes = f[self.params['svd_keys']][:]

        return input

    def process(self, input):

        task_list = self.task_list
        for task_ind in mpiutil.mpirange(len(task_list)):
            tind_l, tind_o = task_list[task_ind]
            tind_l = tuple(tind_l)
            tind_o = tuple(tind_o)
            print ("RANK %03d est. ps.\n(" + "%03d,"*len(tind_l) + ") x ("\
                    + "%03d,"*len(tind_l) + ")\n")%((mpiutil.rank, ) + tind_l + tind_l)

            cube = []
            cube_w = []
            tind = tind_l

            map_key = self.params['map_key']
            input_map = input[tind[0]][map_key][tind[1:] + (slice(None), )]
            input_map_shp = input_map.shape
            input_map.shape = [input_map_shp[0], -1]
            mode_list = list(self.params['svd_mode'])
            c_map, c_mod = self.clean_f_mode(input_map, self.modes, mode_list)
            for i in range(c_map.shape[0]):
                ii = self.params['svd_mode'][i]
                self.df['clean_map_cln%02dm'%ii][tind_o + (slice(None), )] =\
                        c_map[i].reshape(input_map_shp)
            for i in range(c_mod.shape[0]):
                self.df['the_%02dm'%(i+1)][tind_o + (slice(None), )] =\
                        c_mod[i].reshape(input_map_shp)

        for ii in range(self.input_files_num):
            input[ii].close()

    def clean_f_mode(self, input_data, v, mode_list = [1, ]):
    
        i_map = input_data
    
        c_map = np.zeros((len(mode_list), ) + input_data.shape)
        c_mod = np.zeros((max(mode_list), ) + input_data.shape)
    
        #mode_list = [0, ] + mode_list
    
        st_list = [0, ] + mode_list[:-1]
        ed_list = mode_list
    
        for i in range(len(mode_list)):
    
            st = st_list[i]
            ed = ed_list[i]
    
            print st, ed
    
            for j in range(st, ed):
                print j
    
                vec = v[:, j]
                amp = np.dot(i_map.T, vec)
                fit = amp[None, :] * vec[:, None]
                c_mod[j, :, :] = fit
                i_map[:, :]   -= fit
    
            c_map[i, ...] = i_map
    
        return c_map, c_mod

    def finish(self):
        #if mpiutil.rank0:
        print 'RANK %03d Finishing Ps.'%(mpiutil.rank)

        mpiutil.barrier()
        self.df.close()

class FG_clean_CubeFile(FG_clean):

    prefix = 'fgccube_'

    def init_task_list(self):

        super(FG_clean_CubeFile, self).init_task_list()

        aps_num = self.map_shp[0]

        task_list = []
        for ii in range(aps_num):
            #for jj in range(ant_n):
            #    for kk in range(pol_n):
            tind_l = (0, ii)
            tind_o = (ii,)
            task_list.append([tind_l, tind_o])
        self.task_list = task_list
