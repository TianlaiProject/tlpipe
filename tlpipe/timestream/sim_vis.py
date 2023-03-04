"""Generate simulated visibilities with a sky model.

Inheritance diagram
-------------------

.. inheritance-diagram:: SimVis
   :parts: 2

"""

import numpy as np
import h5py
import healpy as hp
from cora.util import hputil
from caput import mpiutil
from caput import mpiarray
from tlpipe.container.timestream import Timestream
from tlpipe.core import constants as const
# from tlpipe.core import tl_array
from tlpipe.map.drift.core import visibility
from tlpipe.map.drift.telescope import cylbeam
from tlpipe.utils import rotate
from tlpipe.utils import progress
from . import timestream_task


class SimVis(timestream_task.TimestreamTask):
    """Generate simulated visibilities with a sky model.

    With a given sky model :math:`T(\\hat{\\boldsymbol{n}})`, the simulated visibility can be generated as


    .. math:: V_{ij}(\\phi) = \\int A_i(\\hat{\\boldsymbol{n}}; \\phi) A_j^*(\\hat{\\boldsymbol{n}}; \\phi) T(\\hat{\\boldsymbol{n}}) e^{2 \\pi i \\hat{\\boldsymbol{n}} \\cdot \\vec{\\boldsymbol{u}}_{ij}(\\phi)} d^2 \\hat{\\boldsymbol{n}}.

    Note we did not include the normalization factor :math:`1/\\Omega_{ij}` in the above
    expression to make the simulated visibilities consisistent with the result given in ps_cal.py.

    """


    params_init = {
                    'model_maps': [],
                  }

    prefix = 'sv_'

    def process(self, ts):

        assert isinstance(ts, Timestream), '%s only works for Timestream object' % self.__class__.__name__

        via_memmap = self.params['via_memmap']
        model_maps = self.params['model_maps']
        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']

        if model_maps is None or len(model_maps) == 0:
            if mpiutil.rank0:
                print('No sky models provided, will do nothing')
        else:

            ts.redistribute('time', via_memmap=via_memmap)

            freq = ts['freq'][:] # MHz
            nfreq = len(freq)
            bls = ts['blorder'][:]
            nbl = bls.shape[0]
            feedpos = ts['feedpos'][:]
            pol = [ ts.pol_dict[p] for p in ts['pol'][:] ] # as string

            # Load file to find out the map shapes.
            with h5py.File(model_maps[0], 'r') as f:
                mapshape = f['map'].shape
                mapdtype = f['map'].dtype

            assert len(mapshape) == 3 and mapshape[0] == nfreq and mapshape[1] == 1 and hp.isnpixok(mapshape[2]), 'Uncorrect model map shape'

            Tmap = np.zeros((nfreq, 1, mapshape[2]), dtype=mapdtype)

            for mapfile in model_maps:
                with h5py.File(mapfile, 'r') as f:
                    Tmap += f['map'][:]

            ra_dec = ts['ra_dec'].local_data
            # ra = ra_dec[:, 0]

            lat = np.radians(ts.attrs['sitelat'])
            # lon = np.radians(ts.attrs['sitelon'])
            lon = 0.0
            zenith = np.array([0.5*np.pi - lat, lon])

            cywid = ts.attrs['cywid'] # m
            # cylinder width in wavelength
            width = cywid / (const.c / (1.0e6 * freq))

            nside = hp.npix2nside(Tmap.shape[2])
            # nvec = hp.pix2vec(nside, np.arange(Tmap.shape[2]))
            nvec = hputil.ang_positions(nside)
            horizon = visibility.horizon(nvec, zenith)

            sim_vis = np.zeros_like(ts.local_vis) # to save simulated vis

            _fwhm_h = 2.0 * np.pi / 3.0
            h_width = 1.0
            fwhm_h = _fwhm_h * h_width

            # m = tl_array.top2eq_m(lat, lon) # conversion matrix

            if show_progress and mpiutil.rank0:
                pg = progress.Progress(ts.vis.shape[0], step=progress_step)

            pxarea = (4 * np.pi / Tmap.shape[2])
            for ti, ra in enumerate(ra_dec[:, 0]):
                if show_progress and mpiutil.rank0:
                    pg.show(ti)
                for fi in range(nfreq):
                    beam = cylbeam.beam_amp(nvec, zenith, width[fi], fwhm_h, fwhm_h)
                    for bi in range(nbl):
                        fd1, fd2 = bls[bi]
                        uij = (feedpos[fd1-1] - feedpos[fd2-1]) * (1.0e6*freq[fi]) / const.c # bl in unit of wavelength
                        fringe = visibility.fringe(nvec, zenith, uij)

                        # uij_eq = np.dot(m, uij)
                        # efactor = np.exp(2.0J * np.pi * np.dot(nvec, uij_eq))

                        Bij = beam**2 * fringe * horizon
                        # rotate Bij
                        Bij = rotate.rotate_map(Bij, rot=(ra, 0.0, 0.0), deg=False)
                        sim_vis[ti, fi, pol.index('xx'), bi] = (Bij * Tmap[fi, 0]).sum() * pxarea # XX
                        sim_vis[ti, fi, pol.index('yy'), bi] = (Bij * Tmap[fi, 0]).sum() * pxarea # YY

            sim_vis = mpiarray.MPIArray.wrap(sim_vis, axis=ts.main_data_dist_axis)
            axis_order = ts.main_axes_ordered_datasets[ts.main_data_name]
            sim_vis = ts.create_main_axis_ordered_dataset(axis_order, 'sim_vis', sim_vis, axis_order)


        return super(SimVis, self).process(ts)
