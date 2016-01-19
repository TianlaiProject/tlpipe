"""Convert data to 4 dimentianal (time, bl, pol, freq)."""

try:
    import cPickle as pickle
except ImportError:
    import pickle

import os
import numpy as np
import h5py
import ephem
import pyfits

from tlpipe.utils import mpiutil
from tlpipe.core.base_exe import Base
from tlpipe.utils.pickle_util import get_value
from tlpipe.utils.date_util import get_ephdate
from tlpipe.utils.path_util import input_path, output_path
from tlpipe.core import tldishes


# Define a dictionary with keys the names of parameters to be read from
# file and values the defaults.
params_init = {
               'nprocs': mpiutil.size, # number of processes to run this module
               'aprocs': range(mpiutil.size), # list of active process rank no.
               'temp_file': 'temp.hdf5',
               'input_file': ['sim_file1', 'sim_file2'],
               'output_file': ['sim_file1.hdf5', 'sim_file2.hdf5'],
               'exclude_ant': [],         # a list of ants to exclude
               'extra_history': '',

               'freq0' : 685.,            # freq of bin 0, in MHz
               'freq_delta': 0.2441,      # freq delta
               'ra_center': 350.8583,     # target ra, will roll to center
               'duration' : 60*60.,       # obervation time
               'source'   : ['23:23:26.0000', '58:48:0.0000']
              }
prefix = 'cvsim_'



class Convert_Sim(Base):
    """Convert data."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(Convert_Sim, self).__init__(parameter_file_or_dict, 
                params_init, prefix, feedback)

    def execute(self):

        temp_file = input_path(self.params['temp_file'])
        input_file = input_path(self.params['input_file'])
        output_file = output_path(self.params['output_file'])

        nfiles = len(input_file)
        assert nfiles > 0, 'No input data file'

        # load the temp file
        print "Load the temp file", temp_file
        tmp = h5py.File(temp_file, 'r')
        self.valid_ants = tmp['vis'].attrs['ants']
        print "Antenna used: ", self.valid_ants

        # load transit time
        transit_time = tmp['vis'].attrs["transit_time"]
        # load pointing direction
        pointing_az, pointing_alt = tmp['vis'].attrs["az_alt"][0]
        # load cite information
        tl = ephem.Observer()
        tl.lon, tl.lat, tl.elev =  tldishes.lon, tldishes.lat, tldishes.elev
        tl.pressure = 0
        tl.date = transit_time[0]
        obj = ephem.readdb("Obj,f|J," + self.params['source'][0] + 
                ',' + self.params['source'][1] + "100,2000")
        obj.compute(tl)
        #print transit_time
        time_center = tl.previous_transit(obj)
        print "Centre time: ", time_center
        #time_center = ephem.julian_date(time_center) 

        for infile, outfile in zip(mpiutil.mpilist(input_file), 
                                   mpiutil.mpilist(output_file)):

            with pyfits.open(infile + '_real.fits', memmap=False) as fin_real,\
                    pyfits.open(infile + '_imag.fits', memmap=False) as fin_imag,\
                    h5py.File(outfile, 'w') as fout:

                sim_data, time_axis = self.read_simfits(fin_real, fin_imag)
                #print sim_data[0, :5, 0, :]
                data = fout.create_dataset('data', data=sim_data)
                data.attrs['axes'] = ['time', 'bls', 'pol', 'freq']
                data.attrs['pol'] = ['I',]
                data.attrs['ants'] = self.valid_ants
                data.attrs['history'] = self.history

                time_axis = [time_center + ti*ephem.second for ti in time_axis]
                time_axis = np.array([ephem.julian_date(te) for te in time_axis])
                fout.create_dataset('time', data=time_axis)

                fout.close()
                fin_real.close()
                fin_imag.close()

    def read_simfits(self, real_list, imag_list):

        freq_binsN = len(real_list)
        base_lineN, time_stepN = real_list[0].data.shape

        print "freqN, base_lineN, time_stepN", freq_binsN, base_lineN, time_stepN

        data_real = np.zeros((time_stepN, base_lineN, 1, freq_binsN))
        data_imag = np.zeros((time_stepN, base_lineN, 1, freq_binsN))

        print "Loading data from fits...",
        for real, imag, i in zip(real_list, imag_list, range(freq_binsN)):

            data_real[:, :, 0, i] = real.data.T
            data_imag[:, :, 0, i] = imag.data.T
        print "Done"

        baseline_index = get_baseline(valid_ants=self.valid_ants)
        data_real = np.take(data_real, baseline_index, axis=1)
        data_imag = np.take(data_imag, baseline_index, axis=1)

        dish_coor = tldishes.dishes_coord
        dish_coor = dish_coor[self.valid_ants - 1]

        u_vect = dish_coor[:,0][None, :] - dish_coor[:,0][:, None]
        u_vect = np.triu(u_vect + 1)
        u_vect = u_vect.flatten()
        u_vect = u_vect[u_vect!=0] - 1
        u_vect[u_vect==0] = 1
        u_vect /= np.abs(u_vect)

        data_imag *= u_vect[None, :, None, None]


        # rotate the target to the center
        pix_center = time_stepN//2
        pix_target = np.digitize([self.params['ra_center'], ], 
                np.linspace(0, 360, time_stepN + 1)) + 1
        data_real = np.roll(data_real, pix_center-pix_target, axis=0)
        data_imag = np.roll(data_imag, pix_center-pix_target, axis=0)

        # time cut
        time_range = int(self.params['duration'] / (86400. / float(time_stepN)))
        data_real = data_real[ pix_center-time_range:pix_center+time_range+1, ...]
        data_imag = data_imag[ pix_center-time_range:pix_center+time_range+1, ...]

        time_axis  = np.arange(data_real.shape[0]) - time_range
        time_axis *= (86400. / float(time_stepN))

        return data_real + 1.J*data_imag, time_axis

def get_baseline(outroot = './data/baseline_index.dat', valid_ants=None):

    ant = [16, 12, 11, 10, 15, 14, 13, 4, 3, 2, 1, 9, 8, 7, 6, 5]

    redundant_list = [
            '10_11', '15_16', '13_14',
            '10_15', '14_16', '12_13',
            '14_15', '13_16', '11_12',
            '11_15',
            '10_14',
            '13_15',
            ]
    baseline_list = ['16_16',]
    for i in range(16):
        for j in range(i+1, 16):
            if ant[i] > ant[j]:
                pair = '%02d_%02d'%(ant[j], ant[i])
            else:
                pair = '%02d_%02d'%(ant[i], ant[j])
            if pair in redundant_list:
                #print pair
                continue
            else:
                baseline_list.append(pair)
    #baseline_list = np.array(baseline_list)[:, None]
    ##print baseline_list.shape
    #np.savetxt(outroot, baseline_list, fmt='%s')
    #baseline_list = baseline_list.flatten().tolist()
    #print baseline_list

    index_list = []
    if valid_ants == None:
        valid_ants = np.arange(16) + 1
    for i in valid_ants:
        for j in valid_ants:

            if i > j: continue

            if i == j:
                index_list.append(0)
                continue

            pair = "%02d_%02d"%(i, j)
            if pair in ['10_11', '15_16', '13_14']:
                pair = '12_16'
            elif pair in ['10_15', '14_16', '12_13']:
                pair = '11_16'
            elif pair in ['14_15', '13_16', '11_12']:
                pair = '10_16'
            elif pair == '11_15':
                pair = '12_14'
            elif pair == '10_14':
                pair = '11_13'
            elif pair == '13_15':
                pair = '10_12'

            index_list.append(baseline_list.index(pair))

    #print len(index_list)
    #print index_list
    return index_list



            #with h5py.File(infile, 'r') as fin, h5py.File(outfile, 'w') as fout:
            #    vis_dataset = fin['vis']
            #    time_zone = get_value(vis_dataset.attrs['timezone'])
            #    start_time = get_value(vis_dataset.attrs['start_time'])
            #    int_time = get_value(vis_dataset.attrs['int_time'])
            #    ants = get_value(vis_dataset.attrs['ants'])
            #    xchans = get_value(vis_dataset.attrs['xchans'])
            #    ychans = get_value(vis_dataset.attrs['ychans'])
            #    bl_dict = get_value(vis_dataset.attrs['bl_dict'])

            #    # convert time to Julian date
            #    stime_ephdate = get_ephdate(start_time, tzone=time_zone)
            #    nt = vis_dataset.shape[0]
            #    time_ephdate = [stime_ephdate + ti*int_time*ephem.second for ti in range(nt)]
            #    time_juldate = np.array([ephem.julian_date(te) for te in time_ephdate])
            #    # select valid antennas (have both x and y)
            #    valid_ants = [ants[i] for i in range(len(ants)) if xchans[i] is not None and ychans[i] is not None]
            #    valid_xchans = [xchans[i] for i in range(len(ants)) if xchans[i] is not None and ychans[i] is not None]
            #    valid_ychans = [ychans[i] for i in range(len(ants)) if xchans[i] is not None and ychans[i] is not None]

            #    # remove excluded ants
            #    for ant in self.params['exclude_ant']:
            #        ant_ind = valid_ants.index(ant)
            #        valid_ants.remove(valid_ants[ant_ind])
            #        valid_xchans.remove(valid_xchans[ant_ind])
            #        valid_ychans.remove(valid_ychans[ant_ind])

            #    nant = len(valid_ants)
            #    xx_pair = [(valid_xchans[i], valid_xchans[j]) for i in range(nant) for j in range(i, nant)]
            #    yy_pair = [(valid_ychans[i], valid_ychans[j]) for i in range(nant) for j in range(i, nant)]
            #    xy_pair = [(valid_xchans[i], valid_ychans[j]) for i in range(nant) for j in range(i, nant)]
            #    yx_pair = [(valid_ychans[i], valid_xchans[j]) for i in range(nant) for j in range(i, nant)]

            #    xx_inds = [bl_dict['%d_%d' % (xi, xj)] for (xi, xj) in xx_pair]
            #    yy_inds = [bl_dict['%d_%d' % (yi, yj)] for (yi, yj) in yy_pair]
            #    xy_inds = [bl_dict['%d_%d' % (xi, yj)] for (xi, yj) in xy_pair]
            #    # yx needs special processing

            #    nbls = nant * (nant + 1) / 2
            #    npol = 4
            #    nfreq = vis_dataset.shape[1]

            #    output_vis = np.zeros((nt, nbls, npol, nfreq), dtype=vis_dataset.dtype)

            #    output_vis[:, :, 0, :] = vis_dataset[:, :, xx_inds].swapaxes(1, 2) # xx
            #    output_vis[:, :, 1, :] = vis_dataset[:, :, yy_inds].swapaxes(1, 2) # yy
            #    output_vis[:, :, 2, :] = vis_dataset[:, :, xy_inds].swapaxes(1, 2) # xy
            #    for bi, (yi, xj) in enumerate(yx_pair):
            #        try:
            #            ind = bl_dict['%d_%d' % (yi, xj)]
            #            output_vis[:, bi, 3, :] = vis_dataset[:, :, ind]
            #        except KeyError:
            #            ind = bl_dict['%d_%d' % (xj, yi)]
            #            output_vis[:, bi, 3, :] = vis_dataset[:, :, ind].conj()

            #    # save data converted
            #    data = fout.create_dataset('data', data=output_vis)
            #    # copy metadata from input file
            #    for attrs_name, attrs_value in vis_dataset.attrs.iteritems():
            #        data.attrs[attrs_name] = attrs_value
            #    # update some attrs
            #    data.attrs['ants'] = valid_ants
            #    data.attrs['xchans'] = valid_xchans
            #    data.attrs['ychans'] = valid_ychans
            #    # data.attrs['time'] = time_juldate # could not save into attributes
            #    fout.create_dataset('time', data=time_juldate)
            #    data.attrs['axes'] = ['time', 'bls', 'pol', 'freq']
            #    data.attrs['pol'] = ['xx', 'yy', 'xy', 'yx']
            #    data.attrs['history'] = self.history
            #    del data.attrs['bl_dict']


def check_plot(file_root):

    import matplotlib.pyplot as plt

    data = h5py.File(file_root, 'r')

    vis = data['data']
    print vis.shape

    plt.pcolormesh(vis[:,3,0,:].real.T)
    plt.show()


if __name__=="__main__":


    #get_baseline(outroot = './data/baseline_index.dat')

    #import pyfits

    #fits_root = '/home/zhangjiao/visib/'
    #fits_file = 'visib_freq_imag.fits'

    #fits_hdulist = pyfits.open(fits_root + fits_file)
    #print len(fits_hdulist)

    #for key in fits_hdulist[0].header.keys():
    #    print key, fits_hdulist[0].header[key]

    file_root = "/project/ycli/data/tianlai/Jsim/sim_CasA_Transit_pm3600s.hdf5"
    check_plot(file_root)
