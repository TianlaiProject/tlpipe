"""Module to do data clean"""


from tlpipe.utils import mpiutil
from tlpipe.core.base_exe import Base
import numpy as np
import matplotlib.pyplot as plt
import h5py

#*******************************************************************************
# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
params_init = {
               'nprocs': mpiutil.size, # number of processes to run this module
               'aprocs': range(mpiutil.size),
               'sigma': 2.0 ,
               'time': [500,8000],  #To choose a time list to get the index of noise. Such as [1,2,1000].
               'input_file': 'data1_conv.hdf5', 
               'input_dir': '/home/mtx/data/Tianlai/', 
               'output_file': 'output.hdf5',
               'output_dir': '/home/mtx/data/Tianlai/',
               'Plot_test': False # To test the result.
              }
prefix = 'CD_'
#*******************************************************************************
class CleanData(Base):
    """Class to do data clean."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(CleanData, self).__init__(parameter_file_or_dict, params_init, prefix, feedback)

    @classmethod
    def execute(self):
        if mpiutil.rank==0:
            input=params_init['input_file']
            input_dir=params_init['input_dir']
            output=params_init['output_file']
            output_dir=params_init['output_dir']
            sigma = params_init['sigma']
            time= params_init['time']
            Plot_test= params_init['Plot_test']

#           input=self.params['input_file']
#           input_dir=self.params['input_dir']
#           output=self.params['output_file']
#           output_dir=params_init['output_dir']
#           sigma = params_init['sigma']
#           time= params_init['time']
#           Plot_test= params_init['Plot_test']

            f=h5py.File(input,'r')
            shape=f['data'].shape
            dtype=f['data'].dtype
            data_input=f['data'][...]
            f.close()
            print shape,dtype
#           dset = g.create_dataset(name='data',shape=shape,dtype=dtype)
    
            for bls in np.arange(shape[1]):
                for pol in np.arange(shape[2]):
    #               data=f['data'][:,bls,pol,:]
                    s = data_input[:,bls,pol,:][time]
                    index_all = []
    
                    if Plot_test:
                        if bls<3:
                            plt.imshow(data_input[:,bls,pol,:][8500:9000].real)
                            plt.title('test_clean_%d_%d_origin.eps'%(bls,pol))
                            plt.colorbar()
                            plt.xlim(0, 512)
                            plt.savefig(output_dir+'test_clean_%d_%d_origin.eps'%(bls,pol))
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
                    
                    data_input[:,bls,pol,index] = np.nan

                    if Plot_test:
                        if bls<3:
                            plt.imshow(data_input[:,bls,pol,:][8500:9000].real)
                            plt.title('test_clean_%d_%d'%(bls,pol))
                            plt.colorbar()
                            plt.xlim(0, 512)
                            plt.savefig(output_dir+'test_clean_%d_%d.eps'%(bls,pol))
                            plt.clf()
            g=h5py.File(output,'w')
            g.create_dataset(name='data',data=data_input)
            g.close()
