import numpy as np

import data_file

class DataSet(object):
    """Visibility data set."""

    def __init__(self, f12_file_list, f43_file_list, check_cnt=True):

        self.data = None
        self.nfreq = None
        self.ch_pairs = None
        self.nch_pair = None
        self.axes = None

        last_end_cnt = None
        for f12, f43 in zip(f12_file_list, f43_file_list):
            f12_data = data_file.DataFile(f12)
            if f12_data.flag != 'f12':
                raise ValueError('Incorrect data file %s' % f12)
            f43_data = data_file.DataFile(f43)
            if f43_data.flag != 'f43':
                raise ValueError('Incorrect data file %s' % f43)
            if not (f12_data.start_cnt == f43_data.start_cnt and f12_data.end_cnt == f43_data.end_cnt):
                raise ValueError('Data file %s and %s do not align'% (f12, f43))

            this_start_cnt = f12_data.start_cnt
            this_end_cnt = f12_data.end_cnt
            if check_cnt:
                if last_end_cnt is not None:
                    if this_start_cnt <=last_end_cnt:
                        raise ValueError('Incorrect count in file %s and %s' % (f12, f43))
                    elif this_start_cnt != last_end_cnt + 1:
                        print 'Not continuous count for file %s and %s' % (f12, f43)

            if self.data is None:
                self.data = np.dstack((f12_data.data, f43_data.data))
            else:
                self.data = np.vstack((self.data, np.dstack((f12_data.data, f43_data.data))))

            if self.nfreq is None:
                self.nfreq = f12_data.nfreq
            if self.ch_pairs is None:
                self.ch_pairs = np.vstack((f12_data.ch_pairs, f43_data.ch_pairs))
            if self.axes is None:
                self.axes  = f12_data.axes

        self.nch_pair = len(self.ch_pairs)
