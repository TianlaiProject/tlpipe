import numpy as np
import timestream_task
import h5py
from caput import mpiutil
from tlpipe.utils.path_util import output_path

class SVD(timestream_task.TimestreamTask):
    """
    nothing but testing
    """

    params_init = {
            'mode_list' : [0],
            'prewhiten' : False,
            }

    prefix = 'todsvd_'

    def process(self, ts):

        print ts.vis.shape

        func = ts.bl_data_operate

        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']
        mode_list     = self.params['mode_list']

        self.cleaned_data_list = []
        self.cleaned_mode_list = []
        func(self.find_and_clean_modes, full_data=True, copy_data=True, 
                show_progress=show_progress, 
                progress_step=progress_step, keep_dist_axis=False)

        cleaned_data_list = np.array(self.cleaned_data_list)
        for mi in range(len(mode_list)):
            cleaned_data = cleaned_data_list[:, mi, ...]
            cleaned_data = np.rollaxis(cleaned_data, 0, 4)

            dset_name = 'vis_sub%02dmodes'%mode_list[mi]
            ts.create_main_time_ordered_dataset(dset_name, cleaned_data)
            ts[dset_name].attrs['dimname'] = ts.vis.attrs['dimname']

            #dset_name = 'vis_sub%02dmodes_mask'%mode_list[mi]
            #ts.create_main_time_ordered_dataset(dset_name, ts.vis_mask)
            #ts[dset_name].attrs['dimname'] = ts.vis.attrs['dimname']
        
        cleaned_mode_list = np.array(self.cleaned_mode_list)
        for mi in range(max(mode_list)):
            cleaned_mode = cleaned_mode_list[:, mi, ...]
            cleaned_mode = np.rollaxis(cleaned_mode, 0, 4)

            dset_name = 'modes%02d'%(mi+1)
            ts.create_main_time_ordered_dataset(dset_name, cleaned_mode)
            ts[dset_name].attrs['dimname'] = ts.vis.attrs['dimname']

            #dset_name = 'modes%02d_mask'%(mi+1)
            #ts.create_main_time_ordered_dataset(dset_name, ts.vis_mask)
            #ts[dset_name].attrs['dimname'] = ts.vis.attrs['dimname']
        #ts['mode_list'] = mode_list
        ts.create_dataset('mode_list', data=np.array(mode_list))
        
        return super(SVD, self).process(ts)

    def find_and_clean_modes(self, vis, vis_mask, li, gi, bl, ts, **kwargs):

        mode_list = self.params['mode_list']

        print vis.dtype
        #vis = np.abs(vis) - 1.
        #vis = np.abs(vis)
        vis_rawshp = vis.shape

        if len(self.output_files) > 1:
            raise 
        elif len(self.output_files) == 0:
            svd_name = None
        else:
            svd_suffix = '_svdmodes_m%03d_x_m%03d.h5'%(bl[0]-1, bl[1]-1)
            svd_name = self.output_files[0].replace('.h5', svd_suffix)
            svd_name = output_path(svd_name, relative= not svd_name.startswith('/'))

        mode_list = self.params['mode_list']

        bad_time = np.all(vis_mask, axis=(1, 2))
        bad_freq = np.all(vis_mask, axis=(0, 2))
        good = (~bad_time)[:, None] * (~bad_freq)[None, :]
        good = good[:, :, None] * np.ones_like(vis).astype('bool')

        vis = vis[~bad_time, ...][:, ~bad_freq, ...]
        if self.params['prewhiten']:
            print "pre-whiten the vis"
            vis = vis - np.mean(vis, axis=0)[None, :, :]
        t_shp, f_shp, npol = vis.shape
        k_shp = min(t_shp, f_shp)

        u = np.zeros([t_shp, k_shp, npol])
        v = np.zeros([k_shp, f_shp, npol])
        s = np.zeros([k_shp, npol])

        for i in range(npol):
            u[...,i], s[:,i], v[...,i] =\
                    np.linalg.svd(vis[..., i], full_matrices=False)

        if svd_name is not None:
            print svd_name
            with h5py.File(svd_name, 'w') as f:
                f['u'] = u
                f['v'] = v
                f['s'] = s
                if bad_freq is not None:
                    f['bad_freq'] = bad_freq
                if bad_time is not None:
                    f['bad_time'] = bad_time
                f['t'] = ts['sec1970']
                f['f'] = ts.freq[:] * 1.e-3

        mode_list = list(mode_list)
        cleaned_data = np.zeros((len(mode_list),) + vis_rawshp)
        cleaned_mode = np.zeros((max(mode_list),) + vis_rawshp)
        cleaned_data_tmp, cleaned_mode_tmp = \
                clean_mode(vis, s, u, v, mode_list = mode_list)
        #print "\t vis mean after svd", cleaned_data_tmp[0].mean()
        cleaned_data_tmp.shape = (len(mode_list), -1)
        cleaned_data[:, good] = cleaned_data_tmp
        #print "\t vis mean after svd", cleaned_data[0][good].mean()
        cleaned_mode_tmp.shape = (max(mode_list), -1)
        cleaned_mode[:, good] = cleaned_mode_tmp

        self.cleaned_data_list.append(cleaned_data)
        self.cleaned_mode_list.append(cleaned_mode)

def clean_mode(input_data, s, u, v, mode_list = [1, ]):

    i_map = input_data

    c_map = np.zeros((len(mode_list), ) + input_data.shape)
    c_mod = np.zeros((max(mode_list), ) + input_data.shape)

    #mode_list = [0, ] + mode_list

    st_list = [0, ] + mode_list[:-1]
    ed_list = mode_list

    for i in range(len(mode_list)):

        st = st_list[i]
        ed = ed_list[i]

        print "Subtract the %02d-th to %02d-th Modes"%(st, ed)

        for j in range(st, ed):
            print "\t %02d Done."%j

            u_vec = u[:, j, 0]
            v_vec = v[j, :, 0]
            fit = (u_vec[:, None] * v_vec[None, :]) * s[j, 0]
            c_mod[j, :, :, 0] = fit
            i_map[:, :, 0]   -= fit


            u_vec = u[:, j, 1]
            v_vec = v[j, :, 1]
            fit = (u_vec[:, None] * v_vec[None, :]) * s[j, 1]
            c_mod[j, :, :, 1] = fit
            i_map[:, :, 1]   -= fit

        c_map[i, ...] = i_map

    return c_map, c_mod


def clean_t_mode(input_data, u, mode_list = [1, ]):

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

            vec = u[:, j, 0]
            amp = np.dot(vec, i_map[:,:,0])
            fit = amp[None, :] * vec[:, None]
            c_mod[j, :, :, 0] = fit
            i_map[:, :, 0]   -= fit


            vec = u[:, j, 1]
            amp = np.dot(vec, i_map[:, :, 1])
            fit = amp[None, :] * vec[:, None]
            c_mod[j, :, :, 1] = fit
            i_map[:, :, 1]   -= fit



        c_map[i, ...] = i_map

    return c_map, c_mod

