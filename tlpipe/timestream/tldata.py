''' 
    module to define the basic tianlei data formate class
    used in the offline data analysis. 

    privde the function that read the raw data formate.
    
'''

import numpy as np
import antenna_array as AA
import h5py
import pickle
import time
import ephem

def get_jul_date(local_time, tzone='CST', time_formate=None):

    #This function can be moved to some other module later.

    if time_formate == None:
        time_formate = "%Y%m%d%H%M%S %Z"
    start_time_gmt = time.strftime("%Y/%m/%d %H:%M:%S", 
            time.gmtime(time.mktime(time.strptime(
                local_time+" "+ tzone, time_formate))))
    start_time_jul = ephem.julian_date(start_time_gmt)
    return start_time_jul

def read_raw(path='', antenna_list=None):

    '''
    read the raw data into TLVis
    '''

    vis_dir = h5py.File(path, 'r')
    vis_data = vis_dir['vis']

    bl_dick = pickle.loads(vis_data.attrs['bl_dict'])

    # get the antenna used according to bl_dick
    antenna_numb = []
    for key in bl_dick.keys():
        ant1, ant2 = key.split('_')
        ant1 = int(ant1)
        if ant1 % 2 == 0 and ant1/2 not in antenna_numb:
            antenna_numb.append(int(ant1)/2)
    antenna_numb = np.sort(antenna_numb)

    # if the antennas and the correlater are conneted 
    # in idfferent way, please specify the antennas number
    # by hand with keyword 'antenna_list'.
    if antenna_list == None:
        antenna_list = list(antenna_numb)

    # setup time axis
    # It would be better to read the time zone information from 
    # the raw data file. For now, set here.
    tzone = 'CST'
    #start_time = data_file.split('_')[0]
    start_time = vis_data.attrs['obs_time']
    start_time_jul = get_jul_date(start_time, tzone, "%Y/%m/%d %H:%M:%S %Z")
    #time_axis = np.arange(vis_data.shape[-1])/86400. + start_time_jul
    time_axis = np.arange(0, vis_data.attrs['duration'], vis_data.attrs['delta_t'])
    time_axis = time_axis/86400. + start_time_jul

    # setup frequency axis
    # It would be great if the raw data can provide frequency information.
    # For not, set here.
    df = 250./1024
    freq_axis = np.arange(0, 512*df, df) + 685.

    vis_list = []
    for i in range(len(antenna_numb)):
        for j in range(i, len(antenna_numb)):
            a1 = antenna_numb[i]
            a2 = antenna_numb[j]

            # remove some bad antennas
            #if antenna_list[i] in [3, 5, 6, 8, 12] : continue
            #if antenna_list[j] in [3, 5, 6, 8, 12] : continue
            if antenna_list[i] in [12,] : continue
            if antenna_list[j] in [12,] : continue

            vis_temp = []
    
            # XX
            bl_key = '%d_%d'%(2*a1-1, 2*a2-1)
            vis_temp.append(vis_data[bl_dick[bl_key], :, :].T)
    
            # YY
            bl_key = '%d_%d'%(2*a1, 2*a2)
            vis_temp.append(vis_data[bl_dick[bl_key], :, :].T)
    
            # XY
            bl_key = '%d_%d'%(2*a1-1, 2*a2)
            vis_temp.append(vis_data[bl_dick[bl_key], :, :].T)
    
            # YX
            if a1 != a2:
                bl_key = '%d_%d'%(2*a1, 2*a2-1)
            else:
                bl_key = '%d_%d'%(2*a1-1, 2*a2)
            vis_temp.append(vis_data[bl_dick[bl_key], :, :].T)

            vis_temp = np.array(vis_temp)

            # switch X Y for the 16th antenna
            if a2 == 16 and a1 != 16:
                vis_temp = vis_temp[(2, 3, 0, 1), ...]
            elif a1 == 16 and a2 != 16:
                vis_temp = vis_temp[(3, 2, 1, 0), ...]
            elif a1 == 16 and a2 == 16:
                vis_temp = vis_temp[(1, 0, 2, 3), ...]

            vis_temp = np.swapaxes(vis_temp, 0, 1)
    
            vis_list.append(vis_temp)

    # remove some bad antennas
    for x in [12,]:
        antenna_list.remove(x)
    print antenna_list
    
    vis_list = np.array(vis_list)
    vis_list = np.swapaxes(vis_list, 0, 1)
    print vis_list.shape


    tlvis = TLVis()
    tlvis.vis = vis_list
    tlvis.time = time_axis
    tlvis.freq = freq_axis

    tlvis.antenna_list = antenna_list
    tlvis.array.tl_freq = freq_axis
    tlvis.array.antenna_array_mask += 1
    tlvis.array.antenna_array_mask[np.array(antenna_list)-1] = False

    tlvis_history  = '--- %s ---\n'%time.strftime('%Y/%m/%d %H:%M:%S [%Z]', time.localtime())
    tlvis.history += 'Read raw data from :\n'
    tlvis.history += '%s\n'%path
    tlvis.history += 'Obserber: \n'
    tlvis.history += 'Start at: %s [%s]\n'%(vis_data.attrs['obs_time'], tzone)
    tlvis.history += 'Target  : %s \n'%vis_data.attrs['source']

    vis_dir.close()

    return tlvis

def load(path='', mode='r'):

    vis_file = h5py.File(path, mode)

    tlvis_data = TLVis()

    tlvis_data.vis  = vis_file['vis'].value
    tlvis_data.time = vis_file['time'].value
    tlvis_data.freq = vis_file['freq'].value

    tlvis_data.antenna_list = vis_file['antenna_list'].value
    tlvis_data.array_tl_freq = vis_file['freq'].value
    tlvis_data.array.antenna_array_mask += 1
    tlvis_data.array.antenna_array_mask[np.array(tlvis_data.antenna_list)-1] = False

    tlvis_data.history = vis_file['vis'].attrs['history']

    vis_file.close()

    return tlvis_data

def write(tlvis_data, path='', history=''):

    vis_file = h5py.File(path, 'w')

    vis_file['vis'] = tlvis_data.vis
    vis_file['time']= tlvis_data.time
    vis_file['freq']= tlvis_data.freq

    vis_file['antenna_list'] = tlvis_data.antenna_list

    vis_file['vis'].attrs['history'] = np.string_(\
            tlvis_data.history + '\n--- %s ---\n'%\
            time.strftime('%Y/%m/%d %H:%M:%S [%Z]', time.localtime()) +\
            history)

    vis_file.close()

class TLVis(object):

    '''
    The basic tianlai data formate class
    '''

    def __init__(self):

        # init the data formate

        # ndarray to store the visibility data 
        # [time, baseline, pol, freq]
        self.vis = None 

        # time axis
        self.time = None

        # freq axis
        self.freq = None

        # array infomation class
        self.array = AA.TLDish_Array()
        self.antenna_list = None

        self.history = ''


if __name__=='__main__':

    # test
    data_root = '/project/ycli/data/tianlai/obs_data/'
    data_file = '20151112004108_CassiopeiaA.hdf5'
    antenna_list = [1, 2, 3, 4, 6, 7, 8, 10, 12, 13, 15, 16]

    time0 = time.time()
    #tlvis = read_raw(path=data_root + data_file, antenna_list=antenna_list)

    time1 = time.time()
    #write(tlvis, '/project/ycli/data/tianlai/CasA_201511_test/' + data_file)

    time2 = time.time()
    tlvis_2 = load('/project/ycli/data/tianlai/CasA_201511_test/' + data_file)

    time3 = time.time()

    print tlvis_2.antenna_list
    print tlvis_2.history
