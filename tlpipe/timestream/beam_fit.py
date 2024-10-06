"""Beam fit.

Inheritance diagram
-------------------

.. inheritance-diagram:: BeamFit
   :parts: 2

"""

import re
import numpy as np
from scipy import linalg as la
from scipy.optimize import curve_fit
import ephem
import h5py
import aipy as a
from . import timestream_task
from tlpipe.container.timestream import Timestream
from tlpipe.core import constants as const

from caput import mpiarray
from caput import mpiutil
from tlpipe.utils.path_util import output_path
from tlpipe.utils import rpca_decomp
from tlpipe.cal import calibrators
import matplotlib.pyplot as plt


class BeamFit(timestream_task.TimestreamTask):
    """Beam fit.

    """


    params_init = {
                    'srcs': ['cyg', 'cas', 'crab'],
                    'span': 100, # time points
                    'bli': 0, # use which baseline to fit the beam
                    'del_src_vis': True, # delete src_vis after fitting
                    'chunk_size': 512,
                    'plot_figs': False,
                    'fig_name': 'beam_fit/beam_fit',
                  }

    prefix = 'bf_'

    def process(self, ts):

        assert isinstance(ts, Timestream), '%s only works for Timestream object' % self.__class__.__name__
        assert 'src_vis' in ts.keys(), 'No src_vis to do beam fit'

        srcs = self.params['srcs']
        span = self.params['span']
        bli = self.params['bli']
        del_src_vis = self.params['del_src_vis']
        tag_output_iter = self.params['tag_output_iter']
        via_memmap = self.params['via_memmap']
        chunk_size = self.params['chunk_size']
        plot_figs = self.params['plot_figs']
        fig_prefix = self.params['fig_name']

        ts.redistribute('time', via_memmap=via_memmap)

        # gather time to all ranks
        jul_date = mpiutil.gather_array(ts.local_time, axis=0, root=None, comm=ts.comm)
        time = jul_date
        pol = [ ts.pol_dict[p] for p in ts['pol'][:] ] # as string
        gain_pd = {'xx': 0, 'yy': 1,    0: 'xx', 1: 'yy'} # for gain related op
        bls = ts.bl[:]

        n0s = []
        Scs = []
        transit_viss = []
        transit_vis_masks = []
        for calibrator in srcs:
            # get the calibrator
            try:
                s = calibrators.get_src(calibrator)
            except KeyError:
                if mpiutil.rank0:
                    print('Calibrator %s is unavailable, available calibrators are:')
                    for key, d in calibrators.src_data.items():
                        print('%8s  ->  %12s' % (key, d[0]))
                raise RuntimeError('Calibrator %s is unavailable')
            if mpiutil.rank0:
                print('Try to calibrate with %s...' % s.src_name)

            # get transit time of calibrator
            aa = ts.array # array
            t0 = mpiutil.bcast(ts.local_time[0], root=0, comm=ts.comm) # the first obs time point
            aa.set_jultime(t0) # the first obs time point
            next_transit = aa.next_transit(s)
            transit_time = a.phs.ephem2juldate(next_transit) # Julian date
            # get time zone
            pattern = '[-+]?\d+'
            try:
                tz = re.search(pattern, ts.attrs['timezone'].decode('ascii')).group() # ts.attrs['timezone'] is bytes in python3
            except AttributeError:
                tz = re.search(pattern, ts.attrs['timezone']).group() # ts.attrs['timezone'] is str in python3.10
            tz = int(tz)
            local_next_transit = ephem.Date(next_transit + tz * ephem.hour) # plus 8h to get Beijing time
            # if transit_time > jul_date[-1]:
            if transit_time > max(jul_date[-1], jul_date.max()):
                raise RuntimeError('Data does not contain local transit time %s of source %s' % (local_next_transit, calibrator))

            # the first transit index
            transit_inds = [ np.searchsorted(jul_date, transit_time) ]
            # find all other transit indices
            aa.set_jultime(jul_date[0] + 1.0)
            transit_time = a.phs.ephem2juldate(aa.next_transit(s)) # Julian date
            cnt = 2
            while(transit_time <= jul_date[-1]):
                transit_inds.append(np.searchsorted(jul_date, transit_time))
                aa.set_jultime(jul_date[0] + 1.0*cnt)
                transit_time = a.phs.ephem2juldate(aa.next_transit(s)) # Julian date
                cnt += 1

            if mpiutil.rank0:
                print('transit ind of %s: %s, time: %s' % (s.src_name, transit_inds, local_next_transit))

            ### now only use the first transit point to do the cal
            ### may need to improve in the future
            transit_ind = transit_inds[0]
            # int_time = ts.attrs['inttime'] # second
            start_ind = transit_ind - span
            end_ind = transit_ind + span + 1 # plus 1 to make transit_ind at the center

            start_ind = max(0, start_ind)
            end_ind = min(end_ind, ts.vis.shape[0])

            nt = end_ind - start_ind
            freq = ts.freq[:] # MHz
            nf = len(freq)
            nbl = len(bls)

            # compute s_top for this time range
            n0 = np.zeros((nt, 3))
            for ti, jt in enumerate(time[start_ind:end_ind]):
                aa.set_jultime(jt)
                s.compute(aa)
                n0[ti] = s.get_crds('top', ncrd=3)
            n0s.append(n0)

            Sc = s.get_jys(1.0e-3 * freq)
            Scs.append(Sc)
            # lmd = const.c / (1.0e6*freq)
            # Ai = aa.ants[0].beam.response(n0.T)
            # factor = (lmd**2 * 1.0e-26 * Sc) / (2 * const.k_B) * Ai**2 # NOTE: 1Jy = 1.0e-26 W m^-2 Hz^-1
            # fit = factor[0]

            # transit_vis = ts.vis.data.global_slice[start_ind:end_ind, :, :2, :] # only XX and YY pol
            transit_vis = ts['src_vis'].data.global_slice[start_ind:end_ind, :, :2, :] # only XX and YY pol
            transit_vis_mask = ts.vis_mask.data.global_slice[start_ind:end_ind, :, :2, :] # only XX and YY pol

            if transit_vis is None:
                transit_vis = np.zeros((0, nf, 2, nbl), dtype=ts.local_vis.dtype)
                transit_vis_mask = np.zeros((0, nf, 2, nbl), dtype=ts.local_vis_mask.dtype)

            # gather transit_vis and transit_vis_mask to rank 0
            transit_vis = mpiutil.gather_array(transit_vis, root=0, comm=ts.comm)
            transit_vis_mask = mpiutil.gather_array(transit_vis_mask, root=0, comm=ts.comm)

            transit_viss.append(transit_vis)
            transit_vis_masks.append(transit_vis_mask)

        # delete src_vis to save memory
        if del_src_vis:
            ts.delete_a_dataset('src_vis', reserve_hint=False)

        # average over bl
        if mpiutil.rank0:
            # transit_vis = np.ma.abs(np.ma.array(transit_vis, mask=transit_vis_mask)).mean(axis=-1).filled(np.nan)
            # transit_vis = np.ma.abs(np.ma.array(transit_vis[:, :, :, 0], mask=transit_vis_mask[:, :, :, 0])).filled(np.nan)

            for si, (transit_vis, transit_vis_mask) in enumerate(zip(transit_viss, transit_vis_masks)):
                transit_viss[si] = np.ma.abs(np.ma.array(transit_vis[:, :, :, bli], mask=transit_vis_mask[:, :, :, bli])).filled(np.nan)

            beam_params = np.zeros((nf, 2, 3)) # to save the fitted beam params

            fwhm_factor = 2.0 * np.pi / 3.0
            lmd = const.c / (1.0e6*freq)
            for fi in range(nf):
                for pi in range(2):

                    def func(x, width, fwhm_x, fwhm_y):
                        return (lmd[fi]**2 * 1.0e-26 * 1.0) / (2 * const.k_B) * aa.ants[0].beam.response_fit((x.T, fi), width, fwhm_factor*fwhm_x, fwhm_factor*fwhm_y)**2 # set Sc = 1.0 here

                    n0vs = []
                    tvvs = []
                    for n0, Sc, transit_vis in zip(n0s, Scs, transit_viss):
                        tv = transit_vis[:, fi, pi]
                        vind = np.where(np.isfinite(tv))[0]
                        n0vs.append(n0[vind])
                        tvvs.append(tv[vind]/Sc[fi])
                    popt, pcov = curve_fit(func, np.concatenate(n0vs, axis=0), np.concatenate(tvvs), bounds=([10.0, 0.2, 0.2], [20.0, 3.0, 3.0]))
                    print(popt)

                    beam_params[fi, pi] = np.array(popt)

                    if plot_figs:
                        fig_name = f'{fig_prefix}_fi{fi:03d}_{gain_pd[pi]}.png'
                        if tag_output_iter:
                            fig_name = output_path(fig_name, iteration=self.iteration)
                        else:
                            fig_name = output_path(fig_name)

                        xt = ts.attrs['inttime'] * np.arange(nt) / 60.0 # minutes
                        # plot time slice
                        plt.figure()
                        for s, n0, Sc, transit_vis, c in zip(srcs, n0s, Scs, transit_viss, ['r', 'g', 'b']):
                            plt.plot(xt, transit_vis[:, fi, pi], c, label=s)
                            plt.plot(xt, Sc[fi] * func(n0, *popt), c, lw=2)
                            plt.plot(xt, (lmd[fi]**2 * 1.0e-26 * Sc[fi]) / (2 * const.k_B) * aa.ants[0].beam.response(n0.T)[0]**2, c+'--')
                        plt.legend()
                        plt.xlabel('Time [Minutes]', fontsize=14)
                        plt.ylabel('Amplitude of vis [K]', fontsize=14)
                        plt.savefig(fig_name)
                        plt.close()

        # create a frequency ordered data to save beam_params
        if not mpiutil.rank0:
            beam_params = None

        beam_params = mpiutil.bcast(beam_params, root=0, comm=ts.comm)
        ts.create_freq_ordered_dataset('beam_params', beam_params, axis_order=(1, None, None))

        return super(BeamFit, self).process(ts)
