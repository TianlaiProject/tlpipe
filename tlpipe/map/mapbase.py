#! 
import numpy as np
import healpy as hp
import h5py

from caput import mpiutil


class MapBase(object):

    df = None

    def __del__(self):

        if self.df is not None:
            self.df.close()

    def allocate_output(self, fname, mode='w'):

        self.df = h5py.File(fname, mode, driver='mpio', comm=mpiutil._comm)

    def create_dataset(self, name, dset_shp, dset_info={}, dtype='f'):

        d = self.df.create_dataset(name, dset_shp, dtype=dtype)
        for key, value in dset_info.iteritems():
            d.attrs[key] = repr(value)

    def create_dataset_like(self, name, dset_tmp):

        self.create_dataset(name, dset_tmp.shape, dset_tmp.info, dset_tmp.dtype)

class MultiMapBase(object):

    df_out = []
    df_in  = []

    def __del__(self):

        for df in self.df_out:
            df.close()

        for df in self.df_in:
            df.close()

    def open(self, fname, mode='r'):

        self.df_in += [h5py.File(fname, mode, driver='mpio', comm=mpiutil._comm), ]

    def allocate_output(self, fname, mode='w'):

        self.df_out += [h5py.File(fname, mode, driver='mpio', comm=mpiutil._comm), ]

    def create_dataset(self, df_idx, name, dset_shp, dset_info={}, dtype='f'):

        d = self.df_out[df_idx].create_dataset(name, dset_shp, dtype=dtype)
        for key, value in dset_info.iteritems():
            d.attrs[key] = repr(value)

    def create_dataset_like(self, df_idx, name, dset_tmp):

        self.create_dataset(df_idx, name, dset_tmp.shape, dset_tmp.info, dset_tmp.dtype)




