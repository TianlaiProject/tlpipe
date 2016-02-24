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
import aipy

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
               'ra_center': 351.0497083,  # target ra, will roll to center
               'duration' : 60*60.,       # obervation time
               #'source'   : ['23:23:26.0000', '58:48:0.0000'],
               'source'   : ['23:24:11.93', '58:54:16.7'],
               'phs_ref'  : ['0:0:0.0', '64:54:16.7']
              }
prefix = 'cvsim_'



class Convert_Sim(Base):
    """Convert data."""

    def __init__(self, parameter_file_or_dict=None, feedback=2):

        super(Convert_Sim, self).__init__(parameter_file_or_dict, 
                params_init, prefix, feedback)

    def execute(self):

        print '-'*10

        temp_file = input_path(self.params['temp_file'])
        input_file = input_path(self.params['input_file'])
        output_file = output_path(self.params['output_file'])

        nfiles = len(input_file)
        assert nfiles > 0, 'No input data file'

        # load the temp file
        print "Load the temp file", temp_file
        tmp = h5py.File(temp_file, 'r')
        self.valid_ants = tmp['vis'].attrs['ants'][:3]
        print "Antenna used: ", self.valid_ants

        # load transit time
        transit_time = get_ephdate(tmp['vis'].attrs["transit_time"][0], tzone='UTC+00')

        ## load pointing direction
        #pointing_az, pointing_alt = tmp['vis'].attrs["az_alt"][0]
        # load cite information
        tl = ephem.Observer()
        tl.lon, tl.lat, tl.elev =  tldishes.lon, tldishes.lat, tldishes.elev
        tl.pressure = 0
        tl.date = transit_time
        obj = ephem.readdb("Obj,f|J," + self.params['source'][0] + 
                ',' + self.params['source'][1] + "0,2000")
        #obj_radec = "%s_%s"%tuple(self.params['source'])
        #srclist, cutoff, catalogs = aipy.scripting.parse_srcs(obj_radec, 'misc')
        #obj = aipy.src.get_catalog(srclist, cutoff, catalogs).values()[0]
        obj.compute(tl)
        #print obj.ra, obj.dec
        #print obj.a_ra, obj.a_dec
        #print transit_time
        time_center = tl.previous_transit(obj)
        print "Centre time: ", time_center
        tl.date = time_center
        obj.compute(tl)
        pointing_az, pointing_alt = obj.az, obj.alt
        print "Pointing at: ",pointing_az, pointing_alt
        #time_center = ephem.julian_date(time_center) 

        for infile, outfile in zip(mpiutil.mpilist(input_file), 
                                   mpiutil.mpilist(output_file)):

            with pyfits.open(infile + '_real.fits', memmap=False) as fin_real,\
                    pyfits.open(infile + '_imag.fits', memmap=False) as fin_imag,\
                    h5py.File(outfile, 'w') as fout:

                sim_data, time_axis, freq_axis = self.read_simfits(fin_real,fin_imag)

                sim_data = sim_data.conj()

                #print time_axis[len(time_axis)//2]
                #print time_axis

                # phs to center pix
                #src_phs = np.angle(sim_data[len(time_axis)//2, ...])
                #sim_data *= np.exp(-1.J*src_phs)[None, ...]

                sim_data = sim_data[:, :, None, :] \
                        * np.array([1,1,0,0])[None, None, :, None]
                
                time_axis = [time_center + ti*ephem.second for ti in time_axis]
                time_axis = np.array([ephem.julian_date(te) for te in time_axis])
                fout.create_dataset('time', data=time_axis)

                print "Save data ... ",
                data = fout.create_dataset('data', data=sim_data)
                data.attrs['axes'] = ['time', 'bls', 'pol', 'freq']
                data.attrs['pol'] = ['I',]
                data.attrs['ants'] = self.valid_ants
                data.attrs['freq'] = freq_axis
                data.attrs['history'] = self.history
                data.attrs['transit_time'] = time_center
                data.attrs['az_alt'] = [[pointing_az, pointing_alt],]
                data.attrs['int_time'] = time_axis[1] - time_axis[0]

                #check_rawfits(data.value[:,:,0,:].real, data.value[:,:,0,:].imag, 
                #        output_file=output_file[0].replace('.hdf5', ''))
                #self.phs2zenith(self.params['phs_ref'], fout)
                #self.phs2zenith_flat(self.params['phs_ref'], fout)
                #check_rawfits(data.value[:,:,0,:].real, data.value[:,:,0,:].imag, 
                #        output_file=output_file[0].replace('.hdf5', '') + '_re_ref')

                fout.close()
                print "Done"
                fin_real.close()
                fin_imag.close()

    def phs2zenith_flat(self, phs_ref, data):

        phs_ref = phs_ref[0] + '_' + phs_ref[1]
        srclist, cutoff, catalogs = aipy.scripting.parse_srcs(phs_ref, 'misc')
        phs_ref = aipy.src.get_catalog(srclist, cutoff, catalogs).values()[0]

        # get the tl array configuration
        aa = tldishes.get_aa(data['data'].attrs['freq']*1.e-3)

        aa.set_jultime(data['time'].value[0])
        phs_ref.compute(aa)
        print "The phase originly references to: ", phs_ref.ra, phs_ref.dec
        print "Change reference to Zenith ..."

        for ti, t in enumerate(data['time'].value):
            aa.set_jultime(t)
            phs_ref.compute(aa)
            delta_ra  = phs_ref.ra - aa.sidereal_time()
            #if delta_ra > np.pi: delta_ra -= 2*np.pi
            #if delta_ra < -np.pi: delta_ra += 2*np.pi
            delta_dec = - phs_ref.dec + aa.lat 
            #phs_ref_top = phs_ref.get_crds('top', ncrd=3)
            phs_ref_top = np.array([delta_ra, delta_dec, 0])

            bi = 0
            for i in data['data'].attrs['ants']:
                for j in data['data'].attrs['ants']:

                    if i > j: continue

                    uij = aa.gen_uvw(i-1, j-1, src='z').squeeze()
                    data['data'][ti, bi, :, :] *= \
                            np.exp(-2.0J * np.pi * np.dot(np.sin(phs_ref_top), uij))

                    bi += 1
        return data

    def phs2zenith(self, phs_ref, data):

        phs_ref = phs_ref[0] + '_' + phs_ref[1]
        srclist, cutoff, catalogs = aipy.scripting.parse_srcs(phs_ref, 'misc')
        phs_ref = aipy.src.get_catalog(srclist, cutoff, catalogs).values()[0]

        # get the tl array configuration
        aa = tldishes.get_aa(data['data'].attrs['freq']*1.e-3)

        aa.set_jultime(data['time'].value[0])
        phs_ref.compute(aa)
        print "The phase originly references to: ", phs_ref.ra, phs_ref.dec
        print "Change reference to Zenith ..."

        for ti, t in enumerate(data['time'].value):
            aa.set_jultime(t)
            phs_ref.compute(aa)
            phs_ref_top = phs_ref.get_crds('top', ncrd=3)

            bi = 0
            for i in data['data'].attrs['ants']:
                for j in data['data'].attrs['ants']:

                    if i > j: continue
                    
                    uij = aa.gen_uvw(i-1, j-1, src='z').squeeze()
                    data['data'][ti, bi, :, :] *= \
                            np.exp(-2.0J * np.pi * np.dot(phs_ref_top, uij))

                    bi += 1
        return data

    def read_simfits(self, real_list, imag_list):

        freq_binsN = len(real_list)
        base_lineN, time_stepN = real_list[0].data.shape

        print "freqN, base_lineN, time_stepN", freq_binsN, base_lineN, time_stepN

        data_real = np.zeros((time_stepN, base_lineN, freq_binsN))
        data_imag = np.zeros((time_stepN, base_lineN, freq_binsN))

        print "Loading data from fits ... ",
        for real, imag, i in zip(real_list, imag_list, range(freq_binsN)):
            data_real[:, :, i] = 0.5 * real.data.T
            data_imag[:, :, i] = 0.5 * imag.data.T
        print "Done"

        #output_file = output_path(self.params['output_file'])[0]
        #output_file = output_file.replace('.hdf5', '')
        #check_rawfits(data_real, data_imag, output_file=output_file)

        # add noise
        #print "Add noise ...",
        #data_real += np.random.randn(time_stepN, base_lineN, freq_binsN)
        #data_imag += np.random.randn(time_stepN, base_lineN, freq_binsN)
        #data_imag[:,0,:] = 0
        #print "Done"

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

        #print np.unique(u_vect)
        data_imag *= u_vect[None, :, None]

        # rotate the target to the center
        pix_center = time_stepN//2
        pix_target = np.digitize([self.params['ra_center'], ], 
                np.linspace(0, 360, time_stepN + 1)) - 1
        data_real = np.roll(data_real, pix_center-pix_target, axis=0)
        data_imag = np.roll(data_imag, pix_center-pix_target, axis=0)

        # time cut
        time_range = int(self.params['duration'] / (86400. / float(time_stepN)))
        data_real = data_real[ pix_center-time_range:pix_center+time_range+1, ...]
        data_imag = data_imag[ pix_center-time_range:pix_center+time_range+1, ...]

        time_axis  = np.arange(data_real.shape[0]) - time_range
        time_axis *= (86400. / float(time_stepN))

        freq_axis  = np.arange(freq_binsN) * self.params['freq_delta']
        freq_axis += self.params['freq0']

        return data_real + 1.J*data_imag, time_axis, freq_axis

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

def check_rawfits(data_real, data_imag, output_file='./out'):

    import matplotlib.pyplot as plt

    x = np.arange(data_real.shape[0])

    fig = plt.figure(figsize=(7, 5))
    ax1 = fig.add_axes([0.12, 0.53, 0.78, 0.40])
    ax2 = fig.add_axes([0.12, 0.10, 0.78, 0.40])


    for i in range(data_real.shape[1]):

        ax1.plot(x, np.mean(data_real[:,i,:], axis=-1))
        ax2.plot(x, np.mean(data_imag[:,i,:], axis=-1))

    ax1.set_xticklabels([])
    ax1.minorticks_on()
    ax1.set_xlim(xmin=x.min(), xmax=x.max())
    #ax1.set_xlim(xmin=8300, xmax=8550)
    ax1.tick_params(length=4, width=1., direction='out')
    ax1.tick_params(which='minor', length=2, width=1., direction='out')

    ax2.set_xlabel('Time Index')
    ax2.minorticks_on()
    ax2.set_xlim(xmin=x.min(), xmax=x.max())
    #ax2.set_xlim(xmin=8300, xmax=8550)
    ax2.tick_params(length=4, width=1., direction='out')
    ax2.tick_params(which='minor', length=2, width=1., direction='out')

    plt.savefig(output_file + '_1d.png', format='png')
    #plt.show()


if __name__=="__main__":


    #get_baseline(outroot = './data/baseline_index.dat')

    #import pyfits

    #fits_root = '/home/zhangjiao/visib/'
    #fits_file = 'visib_freq_imag.fits'

    #fits_hdulist = pyfits.open(fits_root + fits_file)
    #print len(fits_hdulist)

    #for key in fits_hdulist[0].header.keys():
    #    print key, fits_hdulist[0].header[key]

    file_root = "/project/ycli/data/tianlai/Jsim/sim_CasA_Transit_pm600s.hdf5"
    #file_root = "/project/ycli/data/tianlai/Jsim/sim_CasA_Transit_pm600s_phs2src.hdf5"
    check_plot(file_root)
