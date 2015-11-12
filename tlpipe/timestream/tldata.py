''' 
    module to define the basic tianlei data formate class
    used in the offline data analysis. 

    privde the function that read the raw data formate.
    
'''

import antenna_array as AA


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

    def read(path=''):

        pass

    def write(path=''):

        pass

    def read_raw(path=''):

        pass


