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


class PowerSpectrum(mapbase.MapBase, pipeline.OneAndOne):
    """Module to estimate the power spectrum."""

    params_init = {
            'prefix'   : 'MeerKAT3',
            'kmin'     : 1.e-2,
            'kmax'     : 1.,
            'knum'     : 10,
            'kbin_x'   : None,
            'kbin_y'   : None,
            'logk'     : True,
            'unitless' : False, 
            'map_key'     : ['clean_map', 'clean_map'],
            'weight_key'  : ['noise_diag', 'noise_diag'],
            'nonorm'      : True, 
            }

    prefix = 'ps_'

    #def __init__(self, parameter_file_or_dict=None, feedback=2):
    #    super(PowerSpectrum, self).__init__(parameter_file_or_dict, feedback)

    def setup(self):

        super(PowerSpectrum, self).setup()

        input_files = self.input_files
        input_files_num = len(input_files)
        self.input_files_num = input_files_num

        self.init_kbins()

        self.init_task_list()

        self.init_output()

    def init_output(self):

        output_file = self.output_files[0]
        output_file = output_path(output_file, 
                relative= not output_file.startswith('/'))
        self.allocate_output(output_file, 'w')
        # load one file to get the ant_n and pol_n
        self.create_dataset('binavg_1d', self.dset_shp + (self.knum,))
        self.create_dataset('counts_1d', self.dset_shp + (self.knum,))

        self.create_dataset('binavg_2d', self.dset_shp + (self.knum_x, self.knum_y))
        self.create_dataset('counts_2d', self.dset_shp + (self.knum_x, self.knum_y))

        self.df['kbin'] = self.kbin
        self.df['kbin_x'] = self.kbin_x
        self.df['kbin_y'] = self.kbin_y
        self.df['kbin_edges'] = self.kbin_edges
        self.df['kbin_x_edges'] = self.kbin_x_edges
        self.df['kbin_y_edges'] = self.kbin_y_edges

    def init_kbins(self):

        logk = self.params['logk']
        kmin = self.params['kmin']
        kmax = self.params['kmax']
        knum = self.params['knum']
        if logk:
            kbin = np.logspace(np.log10(kmin), np.log10(kmax), knum)
            #dk = kbin[1] / kbin[0]
            #kbin_edges = kbin / (dk ** 0.5)
            #kbin_edges = np.append(kbin_edges, kbin_edges[-1]*dk)
        else:
            kbin = np.linspace(kmin, kmax, knum)
            #dk = kbin[1] - kbin[0]
            #kbin_edges = kbin - (dk * 0.5)
            #kbin_edges = np.append(kbin_edges, kbin_edges[-1] + dk)
        kbin_edges = binning.find_edges(kbin, logk=logk)

        self.knum = knum
        self.kbin = kbin
        self.kbin_edges = kbin_edges

        kbin_x = self.params['kbin_x']
        if kbin_x is None:
            self.kbin_x = kbin 
            self.kbin_x_edges = kbin_edges 
        else:
            self.kbin_x = kbin_x
            self.kbin_x_edges = binning.find_edges(kbin_x, logk=logk)
        self.knum_x = len(self.kbin_x)

        kbin_y = self.params['kbin_y']
        if kbin_y is None:
            self.kbin_y = kbin 
            self.kbin_y_edges = kbin_edges 
        else:
            self.kbin_y = kbin_y
            self.kbin_y_edges = binning.find_edges(kbin_y, logk=logk)
        self.knum_y = len(self.kbin_y)



    def init_task_list(self):

        with h5.File(self.input_files[0], 'r') as f:
            map_tmp = al.make_vect(al.load_h5(f, 'clean_map'))
            ant_n, pol_n = map_tmp.shape[:2]
            self.map_info = map_tmp.info

        task_list = []
        for ii in range(self.input_files_num):
            for jj in range(ant_n):
                for kk in range(pol_n):
                    tind_l = (ii, jj, kk)
                    tind_r = tind_l
                    tind_o = tind_l
                    task_list.append([tind_l, tind_r, tind_o])
        self.task_list = task_list
        self.dset_shp = (self.input_files_num, ant_n, pol_n)

    def read_input(self):

        input = []
        for ii in range(self.input_files_num):
            input.append(h5.File(self.input_files[ii], 'r', 
                                 driver='mpio', comm=mpiutil._comm))
            #input = al.make_vect(al.load_h5(f, 'clean_map'))

        return input

    def process(self, input):

        task_list = self.task_list
        for task_ind in mpiutil.mpirange(len(task_list)):
            tind_l, tind_r, tind_o = task_list[task_ind]
            tind_l = tuple(tind_l)
            tind_r = tuple(tind_r)
            tind_o = tuple(tind_o)
            #print "RANK %03d est. ps.\n(file %03d, ant %03d, pol %d)"\
            #        " x (file %03d, ant %03d, pol %d)\n"%(
            #        (mpiutil.rank, ) + tind_l + tind_r)
            print ("RANK %03d est. ps.\n(" + "%03d,"*len(tind_l) + ") x ("\
                    + "%03d,"*len(tind_r) + ")\n")%((mpiutil.rank, ) + tind_l + tind_r)

            cube = []
            cube_w = []
            tind_list = [tind_l, tind_r]
            for i in range(2):
                tind = tind_list[i]

                map_key = self.params['map_key'][i]
                input_map = input[tind[0]][map_key][tind[1:] + (slice(None), )]
                input_map_mask = ~np.isfinite(input_map)
                input_map[input_map_mask] = 0.
                # for testing
                #_std = np.var(input_map[input_map != 0]) ** 0.5
                #input_map *= 0.
                #input_map += np.random.standard_normal(input_map.shape) * _std 
                #print "var %f "% np.var(input_map)
                input_map = al.make_vect(input_map, axis_names = ['freq', 'ra', 'dec'])
                #input_map *= 1.e3
                for key in input_map.info['axes']:
                    input_map.set_axis_info(key,
                                            self.map_info[key+'_centre'],
                                            self.map_info[key+'_delta'])
                # for testeing
                #input_map.info['freq_centre'] -= 300.
                #input_map.info['freq_delta'] /= 2.

                c, c_info = gridding.physical_grid(input_map)
                cube.append(c)

                weight_key = self.params['weight_key'][i]
                if weight_key is not None:
                    weight = input[tind[0]][weight_key][tind[1:] + (slice(None), )]
                    weight[input_map_mask] = 0.
                    if weight_key == 'noise_diag':
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
                        cut = np.percentile(weight, 10)
                        #weight[weight>cut] = cut
                        weight[weight<cut] = 0.

                        #weight /= weight.max()

                    weight = al.make_vect(weight, axis_names = ['freq', 'ra', 'dec'])
                    weight.info = input_map.info

                    cw, cw_info = gridding.physical_grid(weight)
                    cube_w.append(cw)
                    del weight
                else:
                    cw = al.ones_like(c)
                    cw[c==0] = 0.
                    cube_w.append(cw)


                del c, c_info, cw, input_map

                if tind_l == tind_r:
                    cube.append(cube[0])
                    cube_w.append(cube_w[0])
                    break

            #weight = al.ones_like(cube[0])

            ### for testing
            #_m = cube[0] == 0.
            #cube[0] *= 0.
            #cube[0] += np.random.standard_normal(cube[0].shape) * 0.39
            #cube[0][_m] = 0.
            #cube[1] = cube[0]
            print "var %e "% np.var(cube[0][cube[0] != 0.])
            print "mean %e "% np.mean(cube[0][cube[0] != 0.])
            print "var %e "% np.var(cube[1][cube[1] != 0.])
            print "mean %e "% np.mean(cube[1][cube[1] != 0.])
            ps2d, ps1d = pse.calculate_xspec(cube[0], cube[1], cube_w[0], cube_w[1],
                    window=None, #'blackman', #None, 
                    bins=self.kbin_edges, bins_x = self.kbin_x_edges, 
                    bins_y = self.kbin_y_edges,
                    logbins=self.params['logk'],
                    unitless=self.params['unitless'],
                    nonorm = self.params['nonorm'])

            self.df['binavg_1d'][tind_o + (slice(None), )] = ps1d['binavg']
            self.df['counts_1d'][tind_o + (slice(None), )] = ps1d['counts_histo']

            self.df['binavg_2d'][tind_o + (slice(None), )] = ps2d['binavg']
            self.df['counts_2d'][tind_o + (slice(None), )] = ps2d['counts_histo']

            del ps2d, ps1d, cube, cube_w
            gc.collect()

        for ii in range(self.input_files_num):
            input[ii].close()


    def finish(self):
        #if mpiutil.rank0:
        print 'RANK %03d Finishing Ps.'%(mpiutil.rank)

        mpiutil.barrier()
        self.df.close()

class AutoPS_CubeFile(PowerSpectrum):

    prefix = 'apscube_'

    def init_task_list(self):

        with h5.File(self.input_files[0], 'r') as f:
            map_tmp = al.make_vect(al.load_h5(f, self.params['map_key'][0]))
            #ant_n, pol_n = map_tmp.shape[:2]
            self.map_info = map_tmp.info

        aps_num = map_tmp.shape[0]

        task_list = []
        for ii in range(aps_num):
            #for jj in range(ant_n):
            #    for kk in range(pol_n):
            tind_l = (0, ii)
            tind_r = tind_l
            tind_o = (ii, )
            task_list.append([tind_l, tind_r, tind_o])
        self.task_list = task_list
        self.dset_shp = (aps_num, )

class AutoPS_OneByOne(PowerSpectrum):

    prefix = 'aps1b1_'

    def init_task_list(self):

        with h5.File(self.input_files[0], 'r') as f:
            map_tmp = al.make_vect(al.load_h5(f, 'clean_map'))
            #ant_n, pol_n = map_tmp.shape[:2]
            self.map_info = map_tmp.info

        aps_num = self.input_files_num

        task_list = []
        for ii in range(aps_num):
            #for jj in range(ant_n):
            #    for kk in range(pol_n):
            tind_l = (ii, )
            tind_r = tind_l
            tind_o = tind_l
            task_list.append([tind_l, tind_r, tind_o])
        self.task_list = task_list
        self.dset_shp = (aps_num, )

class CrossPS_OneByOne(PowerSpectrum):

    '''
    Est. the Cross power spectrum between input_files and input_files2

    input_files[0][indx] x input_files2[0][indx]
    input_files[1][indx] x input_files2[1][indx]
    ...
    input_files[N][indx] x input_files2[N][indx]

    '''

    params_init = {
            'input_files2' : [],
            }

    prefix = 'xps1b1_'

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(CrossPS_OneByOne, self).__init__(parameter_file_or_dict, feedback)

        input_files  = self.params['input_files']
        input_files2 = self.params['input_files2']
        if len(input_files) != len(input_files2):
            msg = "input_files and intput_files2 should have the same length."
            raise ValueError(msg)
        self.params['input_files'] = input_files + input_files2
        super(CrossPS_OneByOne, self)._init_input_files()




    def init_task_list(self):

        with h5.File(self.input_files[0], 'r') as f:
            map_tmp = al.make_vect(al.load_h5(f, self.params['map_key'][0]))
            ant_n, pol_n = map_tmp.shape[:2]
            self.map_info = map_tmp.info

        xps_num = self.input_files_num / 2

        task_list = []
        for ii in range(xps_num):
            #for jj in range(ant_n):
            #    for kk in range(pol_n):
            tind_l = (ii,           )
            tind_r = (ii + xps_num, )
            tind_o = tind_l
            task_list.append([tind_l, tind_r, tind_o])
        self.task_list = task_list
        self.dset_shp = (xps_num, )

class AutoPS_Opt(PowerSpectrum):

    params_init = {
            'map_key'     : ['delta',     'delta'],
            'weight_key'  : ['separable', 'separable'],
            }

    prefix = 'apsopt_'

    def init_task_list(self):

        with h5.File(self.input_files[0], 'r') as f:
            map_tmp = al.make_vect(al.load_h5(f, 'delta'))
            self.map_info = map_tmp.info

        xps_num = self.input_files_num

        task_list = []
        for ii in range(xps_num):
            tind_l = (0, )
            tind_r = (ii,)
            tind_o = (ii,)
            task_list.append([tind_l, tind_r, tind_o])
        self.task_list = task_list
        self.dset_shp = (xps_num, )




