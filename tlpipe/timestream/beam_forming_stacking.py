"""Beam-forming stacking analysis.

Inheritance diagram
-------------------

.. inheritance-diagram:: Stacking
   :parts: 2

"""

from datetime import datetime
import numpy as np
import h5py
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy import units as u
import aipy as a
from . import timestream_task
from tlpipe.container.timestream import Timestream
from tlpipe.core import constants as const

from caput import mpiutil
from caput import mpiarray
from tlpipe.utils.path_util import output_path
from tlpipe.utils import progress
# import matplotlib.pyplot as plt


class Stacking(timestream_task.TimestreamTask):
    """Beam-forming stacking analysis.

    """


    params_init = {
                    'chunk_size': 512,
                    'span': 5, # time points
                    'use_feedpos_in_file': True,
                    'source_file_name': 'eBOSS_ELG_clustering_data-NGC-vDR16.fits',
                    'beamforming_file_name': 'beamforming/beamforming_srcs.hdf5',
                  }

    prefix = 'bfs_'

    def process(self, ts):

        assert isinstance(ts, Timestream), '%s only works for Timestream object' % self.__class__.__name__

        chunk_size = self.params['chunk_size']
        span = self.params['span']
        use_feedpos_in_file = self.params['use_feedpos_in_file']
        tag_output_iter = self.params['tag_output_iter']
        via_memmap = self.params['via_memmap']
        source_file_name = self.params['source_file_name']
        beamforming_file_name = self.params['beamforming_file_name']
        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']

        ts.redistribute('time', via_memmap=via_memmap)

        # Open the FITS file
        with fits.open(source_file_name) as hdul:
            # Access the ELGINFO extension (which is the second HDU as per the header info)
            srcinfo_hdu = hdul[1]

            # Access the data part of the ELGINFO HDU
            src_data = srcinfo_hdu.data

            # Extract RA, DEC, and Z columns
            src_ra = src_data['RA']
            src_dec = src_data['DEC']
            src_z = src_data['Z']


        # starting sec1970
        sec0 = ts['sec1970'].local_data[0] if len(ts['sec1970'].local_data) > 0 else None
        sec0 = mpiutil.bcast(sec0, root=0, comm=ts.comm)
        # 设置观测时间
        obs_time = Time(datetime.utcfromtimestamp(sec0))

        vis_time = mpiutil.gather_array(ts.local_time, root=None, comm=ts.comm)
        vis_ra = mpiutil.gather_array(ts['ra_dec'].local_data[:, 0], root=None, comm=ts.comm)
        nt = len(vis_time)
        freq = ts.freq[:]
        nfreq = len(freq)
        bls = ts.bl[:]
        nbl = len(bls)

        # get the positions of feeds
        if use_feedpos_in_file:
            feedpos = ts['feedpos'][:]
        else:
            # used the fixed feedpos
            feedpos = ts.feedpos
        aa = ts.array # array
        sis = [] # to save src indices
        fis = [] # to save freq indices
        bfm_xx = []
        bfm_yy = []

        nu_21 = 1420.405751768 # MHz

        # split freq axis among different ranks
        chunk_size = min(chunk_size, mpiutil.size)
        n, r = nfreq // chunk_size, nfreq % chunk_size # number of iterations
        if r != 0:
            n = n + 1
        if show_progress and mpiutil.rank0:
            pg = progress.Progress(n, step=progress_step)
        for i in range(n):
            if show_progress and mpiutil.rank0:
                pg.show(i)


            this_vis = ts.local_vis[:, i*chunk_size:(i+1)*chunk_size].copy()
            this_vis = mpiarray.MPIArray.wrap(this_vis, axis=0, comm=ts.comm).redistribute(axis=1) # distribute on freq axis
            this_vis_mask = ts.local_vis_mask[:, i*chunk_size:(i+1)*chunk_size].copy()
            this_vis_mask = mpiarray.MPIArray.wrap(this_vis_mask, axis=0, comm=ts.comm).redistribute(axis=1) # distribute on freq axis

            this_bfm_xx = []
            this_bfm_yy = []
            if this_vis.local_array.shape[-1] != 0:

                fi = i * chunk_size + mpiutil.rank # freq index of this rank

                if fi < nfreq:
                    for si, (ra0, dec0, z0) in enumerate(zip(src_ra, src_dec, src_z)):
                    # for ra0, dec0, z0 in zip(src_ra[:10], src_dec[:10], src_z[:10]):
                    # for si, (ra0, dec0, z0) in enumerate(zip(src_ra[:1000], src_dec[:1000], src_z[:1000])):
                        freq0 = nu_21 / (z0 + 1)
                        if freq0 < freq[0] or freq0 > freq[-1]:
                            # not inside this freq band
                            continue

                        fi0 = np.argmin(np.abs(freq - freq0))

                        # 创建一个SkyCoord对象，表示ICRS坐标
                        # icrs_coords = SkyCoord(ra=ra0*u.degree, dec=dec0*u.degree, frame='icrs')
                        icrs_coords = SkyCoord(ra=ra0*u.degree, dec=dec0*u.degree, frame='icrs', obstime=obs_time)
                        # 转换到CIRS坐标
                        cirs_coords = icrs_coords.transform_to('cirs')
                        # ra, dec in CIRS
                        ra0 = cirs_coords.ra.degree
                        dec0 = cirs_coords.dec.degree

                        ra0, dec0 = np.radians(ra0), np.radians(dec0) # radian
                        ti = np.argmin(np.abs(ra0 - vis_ra))

                        if ti >= span and ti + span < nt:

                            if fi == 0:
                                sis.append(si)
                                fis.append(fi0)

                            bfm_xx1 = np.ma.masked_all((2*span+1, 1), dtype=np.complex128)
                            bfm_yy1 = np.ma.masked_all((2*span+1, 1), dtype=np.complex128)

                            # beam-forming for this src
                            # construct a aipy.FixedRadioBody for this pixel
                            s = a.fit.RadioFixedBody(ra0, dec0)
                            # get topocentric coord of this src
                            aa.set_jultime(vis_time[ti])
                            s.compute(aa)
                            n0 = s.get_crds('top', ncrd=3)
                            uij = (feedpos[bls[:, 0]-1] - feedpos[bls[:, 1]-1]) * (1.0e6*freq[fi]) / const.c # shp = (nbl, 3)
                            eun = np.exp(-2.0J * np.pi * np.dot(uij, n0)) # shp = (nbl,)

                            for ii, ti1 in enumerate(range(ti-span, ti+span+1)):
                                this_vis_xx = np.ma.array(this_vis.local_array[ti1, 0, 0, :], mask=this_vis_mask.local_array[ti1, 0, 0, :])
                                this_vis_yy = np.ma.array(this_vis.local_array[ti1, 0, 1, :], mask=this_vis_mask.local_array[ti1, 0, 1, :])
                                this_vis_xx_fs = this_vis_xx * eun # fringe-stopping
                                this_vis_yy_fs = this_vis_yy * eun # fringe-stopping
                                # bfm_xx1[ii] = this_vis_xx_fs.mean().real
                                # bfm_yy1[ii] = this_vis_yy_fs.mean().real
                                bfm_xx1[ii] = this_vis_xx_fs.mean()
                                bfm_yy1[ii] = this_vis_yy_fs.mean()

                            this_bfm_xx.append(np.ma.abs(bfm_xx1.mean(axis=0)).filled(np.nan))
                            this_bfm_yy.append(np.ma.abs(bfm_yy1.mean(axis=0)).filled(np.nan))

                if len(this_bfm_xx) > 0:
                    this_bfm_xx = [ np.array(this_bfm_xx) ] # shp = (nsrc, nfreq) with nfreq = 1
                    this_bfm_yy = [ np.array(this_bfm_yy) ] # shp = (nsrc, nfreq) with nfreq = 1

            # gather fi, this_bfm__xx, this_bfm_yy to rank0
            this_bfm_xx = mpiutil.gather_list(this_bfm_xx, root=0, comm=ts.comm)
            this_bfm_yy = mpiutil.gather_list(this_bfm_yy, root=0, comm=ts.comm)

            if mpiutil.rank0:
                bfm_xx.extend(this_bfm_xx)
                bfm_yy.extend(this_bfm_yy)

        if mpiutil.rank0:
            sis = np.array(sis)
            fis = np.array(fis)
            if len(bfm_xx) != 0:
                bfm_xx = np.concatenate(bfm_xx, axis=1)
                bfm_yy = np.concatenate(bfm_yy, axis=1)
            else:
                bfm_xx = np.array(bfm_xx)
                bfm_yy = np.array(bfm_yy)

            # save bfm_xx and bfm_yy
            if tag_output_iter:
                beamforming_file_name = output_path(beamforming_file_name, iteration=self.iteration)
            else:
                beamforming_file_name = output_path(beamforming_file_name)

            with h5py.File(beamforming_file_name, 'w') as f:
                f.create_dataset('bfm_xx', data=bfm_xx)
                f.create_dataset('bfm_yy', data=bfm_yy)
                f.create_dataset('sis', data=sis)
                f.create_dataset('fis', data=fis)
                f.create_dataset('freq', data=freq)


        return super(Stacking, self).process(ts)
