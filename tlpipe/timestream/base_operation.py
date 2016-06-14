""" Base operation  """

from tlpipe.utils import mpiutil
from tlpipe.core.base_exe import Base

import h5py
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from tlpipe.utils.path_util import input_path, output_path

from tlpipe.core.raw_timestream import RawTimestream
from tlpipe.core.timestream import Timestream


# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
base_params = {
               'nprocs': mpiutil.size, # number of processes to run this module
               'aprocs': range(mpiutil.size), # list of active process rank no.
               'input_file': '',
               'output_file': None,
               # baseline set, used for analysis.
               # "all": for all baseline, 
               # "auto": for auto correlation, 
               # "cross": for cross correlatin, 
               # or specify the feed number in a list, like [1, 2, 3]
               'bl_set' : 'all', 
               'extra_history' : '',
               'raw_tod' : False,
              }

class BaseOperation(Base):
    """ 
    Base class for time stream steps 

    The operatoin is looping baseline by baseline, 
    mpi ranks work on different files.
    
    """

    params_init = {}
    prefix = 'bs_'
    history = ''

    def action(self, vis):

        print "Did nothing!!"

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        self.feedback = feedback
        params_init = dict(base_params)
        params_init.update(self.params_init)
        prefix = self.prefix
        super(BaseOperation, self).__init__(
                parameter_file_or_dict, params_init, prefix, feedback)

    def execute(self):

        input_file  = input_path(self.params['input_file'])
        if self.params['output_file'] != None:
            if mpiutil.rank0:
                output_file = output_path(self.params['output_file'])
            mpiutil.barrier()
            output_file = output_path(self.params['output_file'])
        else:
            output_file = None

        if self.params['raw_tod']:
            tod_vis = RawTimestream(input_file)
        else:
            tod_vis = Timestream(input_file)

        #tod_vis.channel_select(corr=self.params['bl_set'])
        #tod_vis.load_main_data()
        #tod_vis.load_all()
        #tod_vis.main_time_ordered_datasets += ('noisecal', 'noisecal_jul_date')
        #tod_vis.time_ordered_datasets += ('noisecal', 'noisecal_jul_date')
        tod_vis.feed_select(corr=self.params['bl_set'])
        tod_vis.load_common()
        tod_vis.load_time_ordered()

        if self.params['raw_tod']:
            tod_vis = tod_vis.separate_pol_and_bl()

        self.action(tod_vis)

        # add history and write to disk
        if output_file != None:
            tod_vis.add_history(history=self.history + self.params['extra_history'])
            tod_vis.to_files(outfiles=output_file)
        mpiutil.barrier()



