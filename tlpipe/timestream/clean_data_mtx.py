"""Module to do data clean"""

try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
import matplotlib.pyplot as plt
import h5py
#********************************************************************************
# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
params_init = {
               #'nprocs': mpiutil.size, # number of processes to run this module
               'nprocs': 1,
               'sigma': 2.0 ,
               'time': [500,8000],  #To choose a time to get the index of noise.
               'input_file': '/home/mtx/data/Tianlai/data1_conv.hdf5', 
               'output_file': '/home/mtx/data/Tianlai/output.hdf5',
               'Plot_test': False
              }
prefix = 'cv_'
#********************************************************************************
class CleanData():
    """Class to do data clean."""
    @classmethod
    def execute(self):
        input=params_init['input_file']
        output=params_init['output_file']
        sigma = params_init['sigma']
        time= params_init['time']
        Plot_test= params_init['Plot_test']
        f=h5py.File(input,'r')
        g=h5py.File(output,'w')
        shape=f['data'].shape
        dtype=f['data'].dtype
        data_input=f['data'][...]
        f.close()
        print shape,dtype
        dset = g.create_dataset(name='data',shape=shape,dtype=dtype)

        for bls in np.arange(shape[1]):
            for pol in np.arange(shape[2]):
#               data=f['data'][:,bls,pol,:]
                data=data_input[:,bls,pol,:]
                s = data[time]
                index_all = []

                if Plot_test:
                    if bls<3:
                        plt.imshow(data[8500:9000].real)
                        plt.title('test_clean_%d_%d_origin.eps'%(bls,pol))
                        plt.colorbar()
                        plt.xlim(0, 512)
                        plt.savefig('/home/mtx/data/Tianlai/test_clean_%d_%d_origin.eps'%(bls,pol))
                        plt.clf()

                for k in s:
                    b = []
                    for m in range(1, len(k)):
                        b.append(k[m] - k[m - 1])
                    b = np.array(b)
                    b = np.abs(b)
                    mean = b.mean()
                    std = b.std()
                    index = np.where(b-mean > std * sigma)[0] + 1
                    indexn = []

                    if len(index)==0:
                        continue
                    elif len(index)==1:
                        indexn.append(index[0])
                    else :
                        for m in range(len(index) - 1):
                            if index[m + 1] - index[m] == 1:
                                indexn.append(index[m])
                            elif index[m + 1] - index[m] == 2:
                                indexn.append(index[m])
                                indexn.append(index[m] + 1)
                            elif m!=0:
                                if (index[m]-index[m-1])>2:
                                    indexn.append(index[m])
                    index_all.append(indexn)
                index = np.array([i for i in set([j for i in index_all for j in i])])
                index = index[np.argsort(index)]
                print 'cleaning baseline:%d pol:%d ...'%(bls,pol),index
                
                if Plot_test:
                    if bls<3:
                        data[:,index] = np.nan
                        dset[:,bls,pol,:]=data
                        plt.imshow(data[8500:9000].real)
                        plt.title('test_clean_%d_%d'%(bls,pol))
                        plt.colorbar()
                        plt.xlim(0, 512)
                        plt.savefig('/home/mtx/data/Tianlai/test_clean_%d_%d.eps'%(bls,pol))
                        plt.clf()
        g.close()
CleanData.execute()
