#!/usr/bin/env python

import aipy
import numpy as np
import ephem
import matplotlib.pyplot as plt
import time
import copy
import gc
import os
import numpy.ma as ma
import scipy.interpolate as interpolate

from aipy.phs import Antenna as AN
from aipy.phs import AntennaArray as ANA


class Antenna(AN):

    def __init__(self, x, y, z, beam, delay=0, offset=0, 
            name='', id=0, antenna_diameter=5,
            poltype=['X','Y'], polangle=[0., 90.]):

        AN.__init__(self, x, y, z, beam, delay=delay, offset=offset)

        self.name = name
        self.id   = id
        self.coor = [x, y, z]
        self.polA  = poltype[0]
        self.polB  = poltype[1]
        self.polA_angle = polangle[0]
        self.polB_angle = polangle[1]

class Antenna_Array(ANA):

    def __init__(self, ants, location, tzone="CST"):

        ANA.__init__(self, location, ants)

        self.location  = location
        self.ants_coor = np.array([ant.coor for ant in ants])
        self.ants_polA = np.array([ant.polA for ant in ants])
        self.ants_polB = np.array([ant.polB for ant in ants])
        self.ants_polA_angle = np.array([ant.polA_angle for ant in ants])
        self.ants_polB_angle = np.array([ant.polB_angle for ant in ants])


# --------------------------------------------------------------------------

tl_location = ("44:10:35.00", "91:44:28.99", 1505)

tl_dish_coor = [ 
        [-34.42, 20.21, 0.00], 
        [-37.53, 31.83, 0.00], 
        [-32.45, 42.57, 0.00],
        [-21.53, 47.83, 0.00], 
        [ -9.91, 44.72, 0.00], 
        [ -3.00, 34.86, 0.00],
        [ -4.05, 22.86, 0.00],
        [-12.56, 14.35, 0.00], 
        [-24.56, 13.30, 0.00],
        [-27.62, 25.90, 0.00],
        [-27.62, 34.70, 0.00], 
        [-20.00, 39.10, 0.00], 
        [-12.38, 34.70, 0.00],
        [-12.38, 25.90, 0.00], 
        [-20.00, 21.50, 0.00], 
        [-20.00, 30.30, 0.00], ]
tl_freq = np.linspace(0.7, 0.8, 256)
tl_dish_diam = 6.

class TLDish_Array(object):
    def __init__(self, antenna_coordinate = None, antenna_channals = None, 
                 antenna_diameter = None, observer_location = None, tzone="CST", 
                 bandpass_data = None, XYZ_coor=False):
        '''
            antenna_coordinate: the coordinate for each antenna in x y z. 
                                in unit of meter
                                x = radial in plane of equator, 
                                y = east, 
                                z = north celestial pole

            antenna_channals: the frequencey channals in unit of GHz
            observer_location: the location (latitude, longitude, elevation)
                               in formate of "dgree:minite:second"
        '''
        self.tzone=tzone
        self.ants = []

        if antenna_coordinate == None:
            antenna_coordinate = tl_dish_coor
        if antenna_channals == None:
            antenna_channals = tl_freq
        if antenna_diameter == None:
            antenna_diameter = tl_dish_diam
        if observer_location == None:
            observer_location = tl_location

        self.get_beam(antenna_channals, antenna_diameter)

        # chenge the coordinat to nanoseconds
        light_speed = ephem.c
        antenna_coordinate = np.array(antenna_coordinate)
        if not XYZ_coor:
            antenna_coordinate = self.convert_to_XYZ(antenna_coordinate, observer_location)
        self.antenna_coordinate = antenna_coordinate/light_speed * 1.e9

        self.antenna_bandpass = []
        for i in range(antenna_coordinate.shape[0]):
            if bandpass_data != None:
                self.antenna_bandpass.append(self.get_bandpass(antenna_channals, 
                                                               bandpass_data[i]))
            ant = Antenna(self.antenna_coordinate[i][0], 
                          self.antenna_coordinate[i][1],
                          self.antenna_coordinate[i][2],
                          self.beam, delay = 0, offset = 0,
                          name = '%02d'%i, id=i, 
                          antenna_diameter = antenna_diameter)
            self.ants.append(ant)
        self.antenna_array = Antenna_Array(ants=self.ants, location=observer_location)
        self.antenna_array_mask = np.zeros(antenna_coordinate.shape[0]).astype('bool')

    def convert_to_XYZ(self, antenna_coor, observer_location):

        lat = ephem.degrees(observer_location[0])
        long= ephem.degrees(observer_location[1])

        A = np.array([[np.cos(lat), 0., np.sin(lat)],
                      [         0., 1.,          0.],
                      [         0., 0.,          0.]])

        antenna_coor = np.dot(antenna_coor, A)
        return antenna_coor

    def get_beam(self, antenna_channals, antenna_diameter, illumination = 0.9):

        self.antenna_channals = antenna_channals
        self.beam = self.creat_beam(self.antenna_channals)
        self.wavelength = 3.e8/(self.antenna_channals*1.e9)
        self.d_ill = np.pi * antenna_diameter * illumination / self.wavelength
        self.beam_pattern = lambda x: (np.sin(self.d_ill*x)/(self.d_ill*x))**2

    def get_source_list(self, source_name_list=[], source_coordinate_list=[]):

        for name in source_name_list:
            self.srcs.append(aipy.phs.RadioSpecial(name))
        for coor in source_coordinate_list:
            self.srcs.append(aipy.phs.RadioFixedBody(coor[0], coor[1], 100, coor[2]))
            source_name_list.append(coor[2])

        return source_name_list


    def get_juliantime_steps(self, start, duration, interval):
        '''
            start time: in form of "year/month/day hour:minute:second"
            duration  : in unit of second
            interval  : in unit of second
        '''
        start_julian = ephem.julian_date(start)
        duration /= (24.*60.*60) # change to unit of day
        interval /= (24.*60.*60) # change to unit of day
    
        time_steps = np.arange(0, duration, interval)
    
        time_steps += start_julian
    
        return time_steps


    def creat_beam(self, freqs_channal):
        #freqs_channal = np.linspace(.700, .800, 1024)
        return aipy.phs.Beam(freqs_channal)

    def get_bandpass(self, freqs_channal, bandpass_file, unit=1.e-3, plot=False):
        '''
            unit: bandpass_data * unit -> GHz
        '''

        bandpass_data = np.load(bandpass_file)
        selected = bandpass_data[1,...] != 0
        bandpass_f = interpolate.interp1d(bandpass_data[0,...][selected]*unit, 
                                          bandpass_data[1,...][selected])

        bandpass = bandpass_f(freqs_channal)
        if plot:
            plt.figure(figsize=(8,8))
            plt.plot(bandpass_data[0]*unit, bandpass_data[1], label='raw data')
            plt.plot(freqs_channal, bandpass, label='interpolated')
            plt.xlim(xmin=freqs_channal.min(), xmax=freqs_channal.max())
            plt.legend()
            plt.show()
            plt.close()

        return bandpass

    def get_gain(self, pointing, source_az, source_alt):
        ''' pointing is antenna direction by azimuth and altitude
        '''
        def convert_dms_2_d(dms):
            dms = dms.split(':')
            return float(dms[0]) + (float(dms[1]) + float(dms[2])/60.)/60.

        pointing_az = convert_dms_2_d(pointing[0])*np.pi/180.
        pointing_alt= convert_dms_2_d(pointing[1])*np.pi/180.

        source_az   = convert_dms_2_d(source_az)*np.pi/180.
        source_alt  = convert_dms_2_d(source_alt)*np.pi/180.

        delta_y = source_alt - pointing_alt
        delta_az= source_az - pointing_az
        delta_x = 2.*np.arcsin(np.cos(source_alt)*np.sin(0.5*delta_az))

        delta = np.sqrt( delta_x**2 + delta_y**2 )

        return self.beam_pattern(delta)

if __name__ == "__main__":
    
    location = ("44:10:35.00","91:44:28.99")

    dish_coor = [ 
            [-34.42, 20.21, 0.00], 
            [-37.53, 31.83, 0.00], 
            [-32.45, 42.75, 0.00],
            [-21.53, 47.83, 0.00], 
            [ -9.91, 44.72, 0.00], 
            [ -3.00, 34.86, 0.00],
            [ -4.05, 22.86, 0.00],
            [-12.56, 14.35, 0.00], 
            [-24.56, 13.30, 0.00],
            [-27.62, 25.90, 0.00],
            [-27.62, 34.70, 0.00], 
            [-20.00, 39.10, 0.00], 
            [-12.38, 34.70, 0.00],
            [-12.38, 25.90, 0.00], 
            [-20.00, 21.50, 0.00], 
            [-20.00, 30.30, 0.00], ]
    freq = np.linspace(0.7, 0.8, 256)
    dish_diam = 6.

    aa = TLDish_Array(dish_coor, freq, dish_diam, location)

