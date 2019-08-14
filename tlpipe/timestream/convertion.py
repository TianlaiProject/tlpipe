import numpy as np
import katdal
import h5py
from os import path
from caput import mpiarray

from tlpipe.pipeline.pipeline import FileIterBase

import time


class MeerKAT2TL(FileIterBase):
    """ class for converting MeerKAT data format to TL

    """
    params_init = {
            'corr' : 'auto',
            'selection' : (), # 'scan',
            }

    prefix = 'm2t_'

    def read_input(self):

        input = []
        for fh in self.input_files:
            input.append(katdal.open(fh, mode='r'))

        return input

    def write_output(self, output):

        #print self.output_files

        # Data Array
        output_data = output[0]
        output_data.select(scans=self.params['selection'][0])
        ants = output_data.ants
        for fi in range(len(self.output_files)):
            output_file = self.output_files[fi]
            output_data.select(scans=self.params['selection'][fi])

            #if path.basename(output_file) != path.basename(output_data.name):
            #    raise 

            #for key in output_data.__dict__.keys():
            #    print key
    

            df = h5py.File(output_file, 'w')
            df.attrs['nickname'] = output_data.experiment_id 
            df.attrs['comment'] = output_data.description
            df.attrs['observer'] = output_data.observer
            history = 'converted from file:\n %s'%output_data.name
            df.attrs['history'] = history
            df.attrs['keywordver'] = '0.0' # Keyword version.

            # Type B Keywords
            df.attrs['sitename'] = 'MeerKAT'
            df.attrs['sitelat'] = -(30. + 42./60. + 47.41/3600.)
            df.attrs['sitelon'] =   21. + 26./60. + 38.00/3600. 
            df.attrs['siteelev'] = 1000.0    # Not precise
            df.attrs['timezone'] = 'UTC+02'  # 
            df.attrs['epoch'] = 2000.0  # year

            df.attrs['telescope'] = 'MeerKAT-Dish-I' # 
            df.attrs['dishdiam'] = output_data.ants[0].diameter
            df.attrs['nants'] = len(output_data.ants)
            df.attrs['npols'] = 2
            df.attrs['cylen'] = -1 # For dish: -1
            df.attrs['cywid'] = -1 # For dish: -1

            df.attrs['recvver'] = '0.0'    # Receiver version.
            df.attrs['lofreq'] = 935.0  # MHz; Local Oscillator Frequency.

            df.attrs['corrver'] = '0.0'    # Correlator version.
            df.attrs['samplingbits'] = 8 # ADC sampling bits.
            df.attrs['corrmode'] = 1 # 2, 3

            obstime = '%s'%output_data.start_time
            obstime = obstime.replace('-', '/')
            inttime = output_data.timestamps[1] - output_data.timestamps[0]
            df.attrs['inttime'] = inttime
            df.attrs['obstime'] = obstime
            df.attrs['sec1970'] = output_data.timestamps[0]
            time_n = output_data.timestamps.shape[0]

            freqs = output_data.freqs * 1.e-6
            freq_n = freqs.shape[0]

            df.attrs['nfreq'] = freq_n # Number of Frequency Points
            df.attrs['freqstart'] = freqs[0] # MHz; Frequency starts.
            df.attrs['freqstep'] = freqs[1] - freqs[0] # MHz; Frequency step.


            feedno = []
            channo = []
            feedpos = []
            for ant in ants:
                antno = int(ant.name[1:]) + 1
                feedno.append(antno)
                channo.append([2 * antno - 1, 2 * antno])
                feedpos.append(ant.position_enu)

            feedno = np.array(feedno)
            channo = np.array(channo)
            feedpos = np.array(feedpos)

            antn   = len(feedno)

            if self.params['corr'] == 'all':
                corr = [' '.join(x) for x in output_data.corr_products]
                hh_indx = [corr.index('m%03dh m%03dh'%(feedno[i]-1, feedno[j]-1))
                    for i in range(antn) for j in range(i, antn)]
                vv_indx = [corr.index('m%03dv m%03dv'%(feedno[i]-1, feedno[j]-1))
                    for i in range(antn) for j in range(i, antn)]
                hv_indx = [corr.index('m%03dh m%03dv'%(feedno[i]-1, feedno[j]-1))
                    for i in range(antn) for j in range(i, antn)]
                vh_indx = [corr.index('m%03dv m%03dh'%(feedno[i]-1, feedno[j]-1))
                    for i in range(antn) for j in range(i, antn)]

                #rvis = np.array(output_data.vis)
                rvis = output_data.vis.dataset
                shp = rvis.shape[:2] + (4, len(hh_indx))
                vis = np.empty(shp, dtype=rvis.dtype)

                t0 = time.time()
                vis[:, :, 0] = rvis[:, :, hh_indx].compute()
                print 'load xx use %8.2f s'%(time.time() - t0)

                t0 = time.time()
                vis[:, :, 1] = rvis[:, :, vv_indx].compute()
                print 'load yy use %8.2f s'%(time.time() - t0)

                t0 = time.time()
                vis[:, :, 2] = rvis[:, :, hv_indx].compute()
                print 'load xy use %8.2f s'%(time.time() - t0)

                t0 = time.time()
                vis[:, :, 3] = rvis[:, :, vh_indx].compute()
                print 'load yx use %8.2f s'%(time.time() - t0)

                df['pol'] = np.array(['hh', 'vv', 'hv', 'vh'])
                df['pol'].attrs['pol_type'] = 'linear'

                blorder = [[feedno[i], feedno[j]] for i in range(antn)\
                        for j in range(i, antn)]
                df['blorder'] = blorder
                df['blorder'].attrs['dimname'] = 'Baselines, BaselineName'

            elif self.params['corr'] == 'auto':

                shp = (time_n, freq_n, 2, antn)
                vis = np.empty(shp, dtype='complex64')
                #ra  = np.empty(shp[:1] + (antn, ), dtype='float')
                #dec = np.empty(shp[:1] + (antn, ), dtype='float')

                #az = np.empty(shp[:1] + (antn, ), dtype='float')
                #el = np.empty(shp[:1] + (antn, ), dtype='float')

                t0 = time.time()
                vis[:] = output_data.vis[:, :, :2*antn].reshape(shp)
                print "\tload vis of scan %3d use %8.2f s"%(
                        self.params['selection'][fi], time.time() - t0)

                df['pol'] = np.array(['hh', 'vv'])
                df['pol'].attrs['pol_type'] = 'linear'

                blorder = [[feedno[i], feedno[i]] for i in range(antn)]
                df['blorder'] = blorder
                df['blorder'].attrs['dimname'] = 'Baselines, BaselineName'

            ra  = output_data.ra[...,:2*antn].reshape((time_n, antn))
            ra  = mpiarray.MPIArray.wrap(ra, axis=0)
            df.create_dataset('ra', data=ra, dtype=ra.dtype, shape=ra.shape)
            df['ra'].attrs['dimname'] = 'Time, Baselines'

            dec  = output_data.dec[...,:2*antn].reshape((time_n, antn))
            dec = mpiarray.MPIArray.wrap(dec, axis=0)
            df.create_dataset('dec', data=dec, dtype=dec.dtype, shape=dec.shape)
            df['dec'].attrs['dimname'] = 'Time, Baselines'

            az  = output_data.az[...,:2*antn].reshape((time_n, antn))
            az  = mpiarray.MPIArray.wrap(az, axis=0)
            df.create_dataset('az', data=az, dtype=az.dtype, shape=az.shape)
            df['az'].attrs['dimname'] = 'Time, Baselines'

            el  = output_data.el[...,:2*antn].reshape((time_n, antn))
            el  = mpiarray.MPIArray.wrap(el, axis=0)
            df.create_dataset('el', data=el, dtype=el.dtype, shape=el.shape)
            df['el'].attrs['dimname'] = 'Time, Baselines'

            flags = output_data.flags[...,:2*antn].reshape(vis.shape)
            flags = mpiarray.MPIArray.wrap(flags, axis=0)
            df.create_dataset('flags', data=flags, dtype=flags.dtype, shape=flags.shape)
            df['flags'].attrs['dimname'] = 'Time, Frequency, Polarization, Baseline'

            #df.create_dataset('vis', chunks = (10, 1024, 1, 4), data=vis,
            df.create_dataset('vis', data=vis, dtype = vis.dtype, shape = vis.shape)
            df['vis'].attrs['dimname'] = 'Time, Frequency, Polarization, Baseline'

            
            df['feedno'] = feedno
            df['channo'] = channo
            df['channo'].attrs['dimname'] = 'Feed No., (HPolarization VPolarization)'
            
            
            df['feedpos'] = feedpos
            df['feedpos'].attrs['dimname'] = 'Feed No., (X,Y,Z) coordinate' ###
            df['feedpos'].attrs['unit'] = 'degree'
            
            #df['antpointing'] = antpointing(16)
            #df['antpointing'].attrs['dimname'] = 'Feed No., (Az,Alt,AzErr,AltErr)'
            #df['antpointing'].attrs['unit'] = 'degree'

            df.close()


def check_data(output_data):
    for key in output_data.__dict__.keys():
        print key
    
    print output_data.freqs
    obstime = "%s"%output_data.start_time
    obstime = obstime.replace('-', '/')
    print obstime
    print output_data.timestamps
    print output_data.timestamps[1] - output_data.timestamps[0]
    print output_data.timestamps[-1] - output_data.timestamps[-2]
    print output_data.corr_products
    
    corr = [' '.join(x) for x in output_data.corr_products]
    
    ants = output_data.ants
    feedno = []
    channo = []
    feedpos = []
    for ant in ants:
        #print ant.__dict__.keys()
        #print ant.ref_position_wgs84
        #print ant.position_ecef
        #print ant.position_enu
        antno = int(ant.name[1:]) + 1
        feedno.append(antno)
        channo.append([2 * antno - 1, 2 * antno])
        feedpos.append(ant.position_enu)
    
    feedno  = np.array(feedno)
    channo  = np.array(channo)
    feedpos = np.array(feedpos)
    antn   = len(feedno)
    
    hh_indx = [corr.index('m%03dh m%03dh'%(feedno[i]-1, feedno[j]-1))
        for i in range(antn) for j in range(i, antn)]
    vv_indx = [corr.index('m%03dv m%03dv'%(feedno[i]-1, feedno[j]-1))
        for i in range(antn) for j in range(i, antn)]
    hv_indx = [corr.index('m%03dh m%03dv'%(feedno[i]-1, feedno[j]-1))
        for i in range(antn) for j in range(i, antn)]
    vh_indx = [corr.index('m%03dv m%03dh'%(feedno[i]-1, feedno[j]-1))
        for i in range(antn) for j in range(i, antn)]
    
    blorder = [[feedno[i], feedno[j]] for i in range(antn) for j in range(i, antn)]
    
    corr = np.array(corr)
    #print corr[hh_indx]
    #print corr[vv_indx]
    #print corr[hv_indx]
    #print corr[vh_indx]
    print feedpos
    
    #print blorder


def creat():
    # Create a hdf5 file object.
    df = h5py.File('example.hdf5', 'w')
    
    # Type A Keywords
    df.attrs['nickname'] = 'Keyword Example Data' # Any nick name for the data file.
    df.attrs['comment'] = 'Here is comment.'
    df.attrs['observer'] = 'Someone'
    df.attrs['history'] = 'No history.'
    df.attrs['keywordver'] = '0.0' # Keyword version.
    # Type B Keywords
    df.attrs['sitename'] = 'Hongliuxia Observatory'
    df.attrs['sitelat'] = 44.17639   # Not precise
    df.attrs['sitelon'] = 91.7413861 # Not precise
    df.attrs['siteelev'] = 1500.0    # Not precise
    df.attrs['timezone'] = 'UTC+08'  # Beijing time
    df.attrs['epoch'] = 2000.0  # year
    # Type C Keywords
    df.attrs['telescope'] = 'Tianlai-Dish-I' # "Tianlai-Cylinder-I", "Tianlai-Cylinder-II" ...
    df.attrs['dishdiam'] = 6.0  # meters; For cylinder: -1.0
    df.attrs['nants'] = 16 # For Cylinder: 3
    df.attrs['npols'] = 2
    df.attrs['cylen'] = 50 # For dish: -1
    df.attrs['cywid'] = 50 # For dish: -1
    # Type D Keywords
    df.attrs['recvver'] = '0.0'    # Receiver version.
    df.attrs['lofreq'] = 935.0  # MHz; Local Oscillator Frequency.
    # Type E Keywords
    df.attrs['corrver'] = '0.0'    # Correlator version.
    df.attrs['samplingbits'] = 8 # ADC sampling bits.
    df.attrs['corrmode'] = 1 # 2, 3
    df.attrs['inttime'] = 1.0
    df.attrs['obstime'] = '2016/02/29 09:30:22' # Year/Month/Day Hour:Minute:Second
    df.attrs['nfreq'] = 512 # Number of Frequency Points
    df.attrs['freqstart'] = 685.0 # MHz; Frequency starts.
    df.attrs['freqstep'] = 0.244140625 # MHz; Frequency step.
    #df.attrs[''] = 
    
    # Data Array
    df['vis'] = vis()
    df['vis'].attrs['dimname'] = 'Time, Frequency, Baseline'
    
    df['feedno'] = np.arange(1, 17, dtype = np.int32) # For Cylinder: 1-192
    
    df['channo'] = np.arange(1, 33, dtype = np.int32).reshape(-1,2) # -1 for invalid channel
    df['channo'].attrs['dimname'] = 'Feed No., (XPolarization YPolarization)'
    
    df['blorder'] = np.array([[2,2],[1,1],[4,4],[3,3], [1,4], [1,3]])
    df['blorder'].attrs['dimname'] = 'Baselines, BaselineName'
    
    df['feedpos'] = np.random.random((16, 3)).astype(np.float32) # Feeds' positions in horizontal coordinate.
    df['feedpos'].attrs['dimname'] = 'Feed No., (X,Y,Z) coordinate' ###
    df['feedpos'].attrs['unit'] = 'degree'
    
    df['antpointing'] = antpointing(16)
    df['antpointing'].attrs['dimname'] = 'Feed No., (Az,Alt,AzErr,AltErr)'
    df['antpointing'].attrs['unit'] = 'degree'
    
    df['polerr'] = np.zeros((16,2), dtype=np.float32) # Clockwise? Anti-Clockwise?
    df['polerr'].attrs['dimname'] = 'Feed No., (XPolErr,YPolErr)'
    df['polerr'].attrs['unit'] = 'degree'
    
    df['noisesource'] = np.array([[60.0, 0.3], [0.0, 0.0], [300.0, 3.5]], np.float32) # Unit: seconds; Cycle < 0 means turned off.
    df['noisesource'].attrs['dimname'] = 'Source No., (Cycle Duration)' #
    df['noisesource'].attrs['unit'] = 'second'
    
    df['transitsource'] = np.array([['2016/2/29 11:03:15', 'Cygnus A'], 
                                    ['2016/2/29 15:32:09', 'Sun'], 
                                    ['2016/2/29 21:54:20', 'Cassiopeia A']])
    df['transitsource'].attrs['dimname'] = 'Source, (DateTime, SourceName)'
    
    df['weather'] = np.array([[0.0,   17.7, -5.6, -9.5,  85.2, 0.0, 29.2, 1.2, 0.0],
                              [300.0, 18.2, -6.9, -11.2, 80.5, 0.0, 31.8, 1.0, 0.0],
                              [600.0, 18.0, -7.5, -13.8, 78.3, 0.0, 30.0, 1.1, 0.0],
                              [900.0, 18.5, -8.0, -14.2, 75.2, 0.0, 29.5, 1.2, 0.0]],
                      dtype = np.float32)
    df['weather'].attrs['dimname'] = 'Weather Data,(TimeOffset, RoomTemperature, SiteTemperature, Dewpoint, Humidity, Precipitation, WindDirection, WindSpeed, Pressure)'
    df['weather'].attrs['unit'] = 'second, Celcius, Celcius, Celcius, %, millimeter, degree, m/s, mmHg'
    
    df.close()
