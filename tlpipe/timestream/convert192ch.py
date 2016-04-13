##########################################################
# Raw data processing program for 192 channel correlator #
# made of Institute of Automation.                       #
# This is verison 0.0 last modified on 2015/10/20.       #
# Any problem, contact jxli@bao.ac.cn                    #
##########################################################

import numpy as np
import matplotlib.pyplot as plt
import struct
import time
import sys
import ephem
import h5py

from tlpipe.kiyopy import parse_ini
from tlpipe.utils import mpiutil
from tlpipe.core.base_exe import Base
from tlpipe.utils.pickle_util import get_value
from tlpipe.utils.date_util import get_ephdate
from tlpipe.utils.date_util import get_juldate
from tlpipe.utils.path_util import input_path, output_path

params_init = {

        'nprocs': mpiutil.size, # number of processes to run this module
        'aprocs': range(mpiutil.size), # list of active process rank no.
        
        'input_file' : '',
        'output_file' : '',
        #'antenna_num' : 96,
        'antenna_list' : [],
        'int_time' : 4,
        'obs_starttime': '',
        'tzone' : 'UTC+08',
        'freq_axis' : None,
        'auto_only' : True,
        'file_list' : [0, 1],
        'time_range': [],

        }

prefix = 'cv192_'


class Convert(Base):

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(Convert, self).__init__(parameter_file_or_dict, params_init, prefix, feedback)

    def execute(self):

        if mpiutil.rank0:
            print '-'*50

        input_file_temp = input_path(self.params['input_file'])
        output_file_temp = output_path(self.params['output_file'])

        # open one file, get data shape
        input_file = input_file_temp%0
        data = h5py.File(input_file, 'r')
        data_shape = data['vis'].shape
        data.close()

        tzone = self.params['tzone']
        obs_starttime = get_ephdate(self.params['obs_starttime'], tzone=tzone)

        time_axis  = np.arange(data_shape[0]).astype('float')
        time_axis *= self.params['int_time'] * ephem.second
        time_axis += obs_starttime 

        if self.params['time_range'] != []:
            st_ind  = get_ephdate(self.params['time_range'][0], tzone=tzone)
            ed_ind  = get_ephdate(self.params['time_range'][1], tzone=tzone)

            st_ind -= obs_starttime 
            ed_ind -= obs_starttime

            st_ind /= data_shape[0] * self.params['int_time'] * ephem.second
            ed_ind /= data_shape[0] * self.params['int_time'] * ephem.second

            st_ind  = int(st_ind)
            ed_ind  = int(ed_ind)

            file_list = mpiutil.mpirange(st_ind, ed_ind+1)
            print mpiutil.rank, file_list
        else:
            file_list = mpiutil.partition_list_mpi(self.params['file_list'])

        #for ii in range(len(input_file_list)):
        #for ii in file_list:
        for ii in file_list:

            input_file = input_file_temp%ii
            output_file = output_file_temp%ii

            data = h5py.File(input_file, 'r')

            data_shape = data['vis'].shape
            if mpiutil.rank0:
                print "Data shape: ", data_shape

            #ant_num = self.params['antenna_num']
            #ant_list = np.arange(ant_num)
            ant_list = self.params['antenna_list']
            ant_num  = len(ant_list)
            #bl_num =  np.sum(range(ant_num+1))
            #new_data_shape = (data_shape[0], bl_num, 4, data_shape[1])

            #print "New Data shape: ", new_data_shape

            #new_data = np.zeros(new_data_shape, dtype=np.complex64)

            if self.params['auto_only']:
                bl_num = ant_num
                new_data_shape = (data_shape[0], bl_num, 4, data_shape[1])

                if mpiutil.rank0:
                    print "New Data shape: ", new_data_shape

                new_data = np.zeros(new_data_shape, dtype=np.complex64)

                #for a1 in range(ant_num):
                for a1 in ant_list:
                    a2 = a1

                    #new_index = bl_num - np.sum(range(ant_num-a1+1)) + (a2-a1)
                    new_index = a1

                    index, flag = BLindex([2*a1+1, 2*a2+1])
                    new_data[:, new_index, 0, :] = data['vis'][:,:,index]
                    new_data[:, new_index, 0, :].imag *= flag

                    index, flag = BLindex([2*a1+1, 2*a2+2])
                    new_data[:, new_index, 1, :] = data['vis'][:,:,index]
                    new_data[:, new_index, 1, :].imag *= flag

                    index, flag = BLindex([2*a1+2, 2*a2+1])
                    new_data[:, new_index, 2, :] = data['vis'][:,:,index]
                    new_data[:, new_index, 2, :].imag *= flag

                    index, flag = BLindex([2*a1+2, 2*a2+2])
                    new_data[:, new_index, 3, :] = data['vis'][:,:,index]
                    new_data[:, new_index, 3, :].imag *= flag
            else:
                bl_num =  np.sum(range(ant_num+1))
                new_data_shape = (data_shape[0], bl_num, 4, data_shape[1])

                if mpiutil.rank0:
                    print "New Data shape: ", new_data_shape

                new_data = np.zeros(new_data_shape, dtype=np.complex64)

                for ai in range(ant_num):
                    for aj in range(ai, ant_num):

                        a1 = ant_list[ai]
                        a2 = ant_list[aj]
                        if mpiutil.rank0: print " %02d x %02d"%(a1, a2)

                        new_index = bl_num - np.sum(range(ant_num-ai+1)) + (aj-ai)

                        index, flag = BLindex([2*a1+1, 2*a2+1])
                        new_data[:, new_index, 0, :] = data['vis'][:,:,index]
                        new_data[:, new_index, 0, :].imag *= flag

                        index, flag = BLindex([2*a1+1, 2*a2+2])
                        new_data[:, new_index, 1, :] = data['vis'][:,:,index]
                        new_data[:, new_index, 1, :].imag *= flag

                        index, flag = BLindex([2*a1+2, 2*a2+1])
                        new_data[:, new_index, 2, :] = data['vis'][:,:,index]
                        new_data[:, new_index, 2, :].imag *= flag

                        index, flag = BLindex([2*a1+2, 2*a2+2])
                        new_data[:, new_index, 3, :] = data['vis'][:,:,index]
                        new_data[:, new_index, 3, :].imag *= flag

            freq_axis = self.params['freq_axis']
            if freq_axis == None:
                delta_f = 250.0 / 2048
                freq_axis = (np.arange(0, 1008 * delta_f, delta_f) + 685)[8:-8]

            fout = h5py.File(output_file, 'w')
            fout.create_dataset('time', data=time_axis \
                    + ii * data_shape[0] * self.params['int_time'] * ephem.second)
            hdata = fout.create_dataset('data', data=new_data)
            hdata.attrs['axes'] = ['time', 'bls', 'pol', 'freq']
            hdata.attrs['pol'] = ['XX', 'XY', 'YX', 'YY']
            hdata.attrs['ants'] = ant_list
            hdata.attrs['freq'] = freq_axis
            hdata.attrs['history'] = 'convert from raw 192ch data'
            #hdata.attrs['transit_time'] = transit_time 
            #hdata.attrs['az_alt'] = [pointing,]
            hdata.attrs['int_time'] = self.params['int_time']
            #hdata.attrs['obj_list'] = obj_list
            hdata.attrs['timezone'] = tzone
            hdata.attrs['start_time'] = obs_starttime

            #obs_starttime = time_axis[-1] + self.params['int_time'] * ephem.second

            fout.close()
            data.close()




def BLindex(bl_name = -1):
    '''
    Return baseline bl_name's index from a 2D array, 
    where bl_name can be a list like [1,2] or an array like np.array([1,2]).
    When use default value bl_name == -1, return the baseline 2D array.
    Mostly translated from lichengcheng's matlab script.
    '''
    order = 0
    data_pair = np.empty((18528, 2), int) # 18528 = (192 + 1) * 192 / 2
    for i in xrange(96):
        j = 2*i + 3
        for k in xrange(47):
            j = j - j/193*192
            data_pair[order,0] = 2*i+2
            data_pair[order,1] = j+1
            order=order+1
            data_pair[order,0] = 2*i+1
            data_pair[order,1] = j
            order=order+1
            data_pair[order,0] = 2*i+2
            data_pair[order,1] = j
            order=order+1
            data_pair[order,0] = 2*i+1
            data_pair[order,1] = j+1
            order=order+1
            j=j+2
    for k in xrange(48):
        i = 2*k + 1
        data_pair[order,0] = i+1
        data_pair[order,1] = i+97
        order=order+1;
        data_pair[order,0] = i
        data_pair[order,1] = i+96
        order=order+1
        data_pair[order,0] = i+1
        data_pair[order,1] = i+96
        order=order+1
        data_pair[order,0] = i
        data_pair[order,1] = i+97
        order=order+1
    for k in xrange(96):
        i = 2*k + 1
        data_pair[order,0] = i+1
        data_pair[order,1] = i+1
        order=order+1
        data_pair[order,0] = i
        data_pair[order,1] = i
        order=order+1
        data_pair[order,0] = i+1
        data_pair[order,1] = i
        order=order+1
    try:
        return np.where(np.all(bl_name == data_pair, axis=1))[0][0], 1
    except IndexError:
        try:
            return np.where(np.all(bl_name[::-1] == data_pair, axis=1))[0][0], -1
        except:
            if bl_name == -1:
                return data_pair
            else:
                print 'Error: Cannot find baseline %s.' % str(bl_name)
                return

if __name__=="__main__":

    bl_list = BLindex()

    bl_list = (bl_list - 1) / 2

    print bl_list[bl_list[:,0]==90,:]
    print np.all((bl_list[:,1] - bl_list[:,0]) >= 0)

