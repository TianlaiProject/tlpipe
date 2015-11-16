"""Module to do data conversion."""

from tlpipe.kiyopy import parse_ini
from tlpipe.utils import mpiutil

from timestream import tldata


# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
params_init = {
               'input_root'  : '',
               'output_root' : '',
               'input_name'  : '', 
               'antenna_list': [1, ],
               'time_range'  : [[], ],
               'nprocs': mpiutil.size, # number of processes to run this module
               'aprocs': range(mpiutil.size), # list of active process rank no.
              }
prefix = 'cv_'

class Conversion(object):
    """Class to do data converion."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        # Read in the parameters.
        self.params = parse_ini.parse(parameter_file_or_dict, params_init,
                                 prefix=prefix, feedback=feedback)
        self.feedback = feedback
        nprocs = min(self.params['nprocs'], mpiutil.size)
        procs = set(range(mpiutil.size))
        aprocs = set(self.params['aprocs']) & procs
        self.aprocs = (list(aprocs) + list(set(range(nprocs)) - aprocs))[:nprocs]
        assert 0 in self.aprocs, 'Process 0 must be active'
        self.comm = mpiutil.active_comm(self.aprocs) # communicator consists of active processes

    def execute(self):

        #print 'Data conversion comes soon...'
        print 'rank %d executing...' % mpiutil.rank

        if mpiutil.rank0:
            print 'Data conversion comes soon...'

            tlvis_raw = tldata.read_raw(
                    root=self.params['input_root'] + self.params['input_name'] + '.hdf5', 
                    antenna_list=self.params['antenna_list'])

            time_axis = tlvis_raw.time

            time_range = self.params['time_range']

            for k in range(len(time_range)):

                time_cut = np.logical_and(
                        time_axis > tldata.get_jul_date(time_range[k][0]), 
                        time_axis < tldata.get_jul_date(time_range[k][1]))

                tlvis = tldata.TLVis()
                tlvis.vis = tlvis_raw.vis[time_cut, ...]
                tlvis.time = tlvis_raw.time[time_cut, ...]
                tlvis.freq = tlvis_raw.freq
                tlvis.antenna_list = tlvis_raw.antenna_list
                tlvis.array = tlvis_raw.array
                tlvis.history  = tlvis_raw.history

                tldata.write(tlvis, root=self.params['output_root']\
                        + self.params['input_name']\
                        + '%s_%s.hdf5'%(time_range[k][0], time_range[k][0]), 
                        history = 'proform time cut\n')



