from tlpipe.sim import units
from astropy.io import fits
#from astropy import units as u
import healpy as hp
import numpy as np

class CAT(object):

    _ra_label = 'ra'
    _dec_label = 'dec'
    _z_label = 'zspec'

    def __init__(self, file_path, file_name, suffix='.fits', tbidx=1, feedback=1):

        self.feedback = feedback
        self.name = file_name

        with fits.open(file_path + file_name + suffix) as hdul:

            print hdul.info()

            self.data = hdul[tbidx].data
            self.cols = self.data.columns

        if feedback > 0:
            print self.cols.info

        self._mask = np.zeros(self.data.shape[0]).astype('bool')

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, value):
        self._mask = value

    @property
    def ra_label(self):
        return self._ra_label

    @ra_label.setter
    def ra_label(self, value):
        self._ra_label = value

    @property
    def ra(self):
        return self.data[self.ra_label][~self.mask] # * u.deg

    @property
    def dec_label(self):
        return self._dec_label

    @dec_label.setter
    def dec_label(self, value):
        self._dec_label = value

    @property
    def dec(self):
        return self.data[self.dec_label][~self.mask] # * u.deg

    @property
    def z_label(self):
        return self._z_label

    @z_label.setter
    def z_label(self, value):
        self._z_label = value

    @property
    def z(self):
        return self.data[self.z_label][~self.mask]

    @property
    def freq(self):
        return units.nu21 / (1. + self.z)

    def project_to_healpix_map(self, nside=256):
        cat_bin = np.arange(hp.nside2npix(nside) + 1) - 0.5
        pix_ind = hp.ang2pix(nside=nside, theta=self.ra.value, phi=self.dec.value, lonlat=True)
        cat_map = np.histogram(pix_ind, bins=cat_bin)[0]

        self.hp_map = cat_map
        return cat_map

    def n_vs_z(self):

        pass

    def n_density_vs_z(self):

        pass

class CATs(CAT):

    def __init__(self, file_path, file_name_list, suffix='.fits', tbidx=1, feedback=1):

        self.feedback = feedback
        self.name = file_name_list[0]

        data_list = []
        data_clos = None

        for file_name in file_name_list:

            with fits.open(file_path + file_name + suffix) as hdul:

                print hdul.info()

                #self.data = hdul[tbidx].data
                #self.cols = self.data.columns

                clos = hdul[tbidx].data.columns

                data_list.append(hdul[tbidx].data)
                #print clos
                #print hdul[tbidx].data.shape

        self.data = np.concatenate(data_list, axis=0)
        print self.data.dtype.names

        self._mask = np.zeros(self.data.shape[0]).astype('bool')


def check_fits(file_path, file_name, suffix='.fits'):

    with fits.open(file_path + file_name + suffix) as hdul:

        print hdul
        print hdul.info()
        print '-' * 20

        for hdu in hdul:
            print hdu
            for key in hdu.header.keys():
                print key, hdu.header[key]

        tbdata = hdul[1].data
        tbcols = tbdata.columns
        print tbcols.info
        print tbdata['flags']


if __name__ == '__main__':

    import matplotlib.pyplot as plt


    data_path = '/data/users/ycli/SDSS/'
    #data_name = 'dr14_mattia/sdss_all_small_dr14'
    data_name = 'dr12/galaxy_DR12v5_LOWZ_South'
    c = CATs(data_path, [data_name, data_name], suffix='.fits')
    c.ra_label = 'RA'
    c.dec_label = 'DEC'
    c.z_label = 'Z'
    print c.ra[:10]
    print c.ra[145264:145264+10]
    #mask  = (c.ra  > 20. * u.deg) + (c.ra  < 0. * u.deg)
    #mask += (c.dec > 20. * u.deg) + (c.dec < -20. * u.deg)
    #c.mask = mask
    #hmap = c.project_to_healpix_map()
    #hp.mollview(hmap)
    #plt.show()
