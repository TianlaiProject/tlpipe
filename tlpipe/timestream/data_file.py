import numpy as np

import params32ch


F1_flag = np.fromstring('\x01\x01\x01\x01', dtype = np.int32)
F2_flag = np.fromstring('\x02\x02\x02\x02', dtype = np.int32)
F3_flag = np.fromstring('\x03\x03\x03\x03', dtype = np.int32)
F4_flag = np.fromstring('\x04\x04\x04\x04', dtype = np.int32)


def aauto(autoname):
    if autoname == 1:
        return np.array([[ 2, 2], [ 1, 1], [ 4, 4], [ 3, 3], [ 1, 2], [ 1, 3]])
    elif autoname == 2:
        return np.array([[ 1, 4], [ 2, 3], [ 2, 4], [ 3, 4]])
    else:
        raise ValueError('Unknown auto name %s' % autoname)

def across(a, b):
    return np.array([[a, b+1], [a, b+2], [a, b+3], [a, b+4]])

def F1list():
    redlist = np.concatenate((aauto(1), aauto(1)+8, aauto(2), aauto(2)+8))
    orangelist = redlist + 4
    yellowlist = np.array([], dtype = np.int).reshape(0,2)
    for i in range(1, 5):
        yellowlist = np.concatenate((yellowlist, across(i, 4), across(i, 4)+8))
    greenlist  = np.array([], dtype = np.int).reshape(0,2)
    for i in range(1, 5):
        greenlist  = np.concatenate((greenlist , across(i, 8), across(i+4, 8)))
    bluelist   = np.array([], dtype = np.int).reshape(0,2)
    for i in range(1, 5):
        bluelist   = np.concatenate((bluelist  , across(i,12), across(i+4,12)))
    return np.concatenate((redlist, orangelist, yellowlist, greenlist, bluelist))

def F2list():
    redlist = np.array([], dtype = np.int).reshape(0,2)
    for i in range(1, 5):
        redlist   = np.concatenate((redlist  , across(i,16), across(i,20)))
    orangelist = np.array([], dtype = np.int).reshape(0,2)
    for i in range(1, 5):
        orangelist   = np.concatenate((orangelist  , across(i+4,16), across(i+4,20)))
    yellowlist = np.array([], dtype = np.int).reshape(0,2)
    for i in range(1, 5):
        yellowlist   = np.concatenate((yellowlist  , across(i+8,16), across(i+8,20)))
    greenlist  = np.array([], dtype = np.int).reshape(0,2)
    for i in range(1, 5):
        greenlist   = np.concatenate((greenlist  , across(i+12,16), across(i+12,20)))
    return np.concatenate((redlist, orangelist, yellowlist, greenlist))

def F3list():
    lst = F2list().copy()
    lst[:, 1] += 8
    return lst

def F4list():
    lst = F1list().copy()
    lst[:, :2] += 16
    return lst


class DataFlagError(Exception):
    """Error in data flag."""
    pass


class DataCntError(Exception):
    """Error in data counter."""
    pass


class DataFile(object):
    """Class to load a single data file, subtract relevant info from it and convert the data to visibility data set.

    The converted visibility data is saved in self.data, which is a numpy array of shape (ncnt, nfreq, -1).
    """

    def __init__(self, filename):

        self.filename = filename

        # raw_data = np.fromfile(filename, dtype=np.int32)
        raw_data = np.fromfile(filename, dtype='>i4') # note dtype
        dot_bit = params32ch.dot_bit
        block_size = params32ch.block_size
        # int_time = params32ch.int_time
        raw_data = raw_data.reshape(-1, 2, block_size/4)

        self.nfreq = params32ch.nfreq # number of frequencies
        self.ncnt = raw_data.shape[0] # number of count

        # validate and set data flag
        self.flag = 'f' # flag of this data file, 'f12' or 'f43'

        if (raw_data[:, 0, 0] == F1_flag).all():
            self.flag += '1'
        elif (raw_data[:, 0, 0] == F4_flag).all():
            self.flag += '4'
        else:
            raise DataFlagError('Data flat incorrect in file %s' % filename)

        if (raw_data[:, 1, 0] == F2_flag).all():
            self.flag += '2'
        elif (raw_data[:, 1, 0] == F3_flag).all():
            self.flag += '3'
        else:
            raise DataFlagError('Data flat incorrect in file %s' % filename)

        # validate count value
        # cnt1 = raw_data[:, 0, 1].newbyteorder()
        # cnt2 = raw_data[:, 1, 1].newbyteorder()
        cnt1 = raw_data[:, 0, 1]
        cnt2 = raw_data[:, 1, 1]
        if not ((cnt1 == cnt2).all() and (np.diff(cnt1) == 1).all()):
            raise DataFlagError('Data counter incorrect in file %s' % filename)

        self.start_cnt = cnt1[0]
        self.end_cnt = cnt1[-1]

        raw_data = raw_data[:, :, 2:].reshape(self.ncnt, 2, self.nfreq, -1)
        raw_data = np.swapaxes(raw_data, 1, 2).reshape(self.ncnt, self.nfreq, -1)

        if self.flag == 'f12':
            # an [nch_pair, 2] array of channel pairs
            self.ch_pairs = np.vstack((F1list(), F2list()))
        elif self.flag == 'f43':
            self.ch_pairs = np.vstack((F4list(), F3list()))
        else:
            raise ValueError('Incorrect flat %s' % self.flag)
        self.nch_pair = len(self.ch_pairs) # number of channel pairs

        def num(ch1, ch2):
            if ch1 == ch2:
                return 1
            else:
                return 2

        n_nums = np.array([num(ch1, ch2) for (ch1, ch2) in self.ch_pairs])
        assert np.sum(n_nums) == raw_data.shape[-1], 'Incorrect data shape'
        n_sums = np.cumsum(np.insert(n_nums, 0, 0))
        new_data = np.empty((self.ncnt, self.nfreq, self.nch_pair), dtype=np.complex64)
        for i in range(len(n_nums)):
            if n_nums[i] == 1:
                new_data[:, :, i] = raw_data[:, :, n_sums[i]]
            elif n_nums[i] == 2:
                # new_data[:, :, i] = raw_data[:, :, n_sums[i]] + 1.0J * raw_data[:, :, n_sums[i]+1]
                new_data[:, :, i] = raw_data[:, :, n_sums[i]+1] + 1.0J * raw_data[:, :, n_sums[i]]

        del raw_data
        self.data = new_data / 2**dot_bit # now visibilities
        del new_data
        self.axes = ['cnt', 'freq', 'ch_pair']
