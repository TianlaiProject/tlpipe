"""Check the phase closure relation.

Inheritance diagram
-------------------

.. inheritance-diagram:: Closure
   :parts: 2

"""

import os
import itertools
import numpy as np
from scipy import optimize
import h5py
import aipy as a
import timestream_task
from caput import mpiutil
from tlpipe.utils.path_util import output_path
import tlpipe.plot
import matplotlib.pyplot as plt


# make cache to speed the visibility getting
def mk_cache(cache, bl, bls, vis):
    # if in cache, return
    if bl in cache.iterkeys():
        return cache[bl]

    # else cache it before return
    i, j = bl
    try:
        bi = bls.index((i, j))
        conj = False
    except ValueError:
        bi = bls.index((j, i))
        conj = True

    vij = np.conj(vis[bi]) if conj else vis[bi]

    cache[(i, j)] = (bi, vij)
    cache[(j, i)] = (bi, np.conj(vij))

    return bi, vij


# Equation for Gaussian
def f(x, a, b, c):
    return a * np.exp(-(x - b)**2.0 / (2 * c**2))


class Closure(timestream_task.TimestreamTask):
    """Check the phase closure relation.

    The closure phase of the visibility for feeds :math:`i,j,k` is
    :math:`\\text{Arg}(V_{ij} V_{jk} V_{ki})`. We see for a strong point source,
    the closure phase should be nearly *zero*.

    """

    params_init = {
                    'calibrator': 'cas',
                    'catalog': 'misc', # or helm,nvss
                    'file_name': 'closure/closure',
                    'plot_closure': True,
                    'fig_name': 'closure/closure',
                    'freq_incl': 'all', # or a list of include freq idx
                    'freq_excl': [],
                    'bins': 201,
                    'gauss_fit': False,
                  }

    prefix = 'pcl_'

    def process(self, ts):

        calibrator = self.params['calibrator']
        catalog = self.params['catalog']
        file_prefix = self.params['file_name']
        plot_closure = self.params['plot_closure']
        fig_prefix = self.params['fig_name']
        bins = self.params['bins']
        gauss_fit = self.params['gauss_fit']
        tag_output_iter = self.params['tag_output_iter']
        freq_incl = self.params['freq_incl']
        freq_excl = self.params['freq_excl']

        ts.redistribute('frequency')

        if freq_incl == 'all':
            freq_plt = range(rt.freq.shape[0])
        else:
            freq_plt = [ fi for fi in freq_incl if not fi in freq_excl ]

        nfreq = len(ts.local_freq[:]) # local nfreq
        feedno = ts['feedno'][:].tolist()
        pol = ts['pol'][:].tolist()
        bl = ts.local_bl[:] # local bls
        bls = [ tuple(b) for b in bl ]

        # calibrator
        srclist, cutoff, catalogs = a.scripting.parse_srcs(calibrator, catalog)
        cat = a.src.get_catalog(srclist, cutoff, catalogs)
        assert(len(cat) == 1), 'Allow only one calibrator'
        s = cat.values()[0]
        if mpiutil.rank0:
            print 'Calibrating for source %s with' % calibrator,
            print 'strength', s._jys, 'Jy',
            print 'measured at', s.mfreq, 'GHz',
            print 'with index', s.index

        ra = ts['ra_dec'][:, 0]
        ra = np.unwrap(ra)
        if 'ns_on' in ts.iterkeys():
            ra = ra[np.logical_not(ts['ns_on'][:])] # get only ns_off values
        abs_diff = np.abs(np.diff(s._ra - ra))
        ind1 = np.argmin(abs_diff)
        if mpiutil.rank0:
            print 'ind1:', ind1

        for pi in [ pol.index('xx'), pol.index('yy') ]: # xx and yy
            if nfreq > 0: # skip empty processes
                # find the ind that not be all masked
                for i in xrange(20):
                    if not ts.local_vis_mask[ind1+i, :, pi].all():
                        ind = ind1 + i
                        break
                    if not ts.local_vis_mask[ind1-i, :, pi].all():
                        ind = ind1 - i
                        break
                else:
                    raise RuntimeError('vis is masked during this period for pol %s' % pol[pi])

                if mpiutil.rank0:
                    print 'ind:', ind

                for fi in xrange(nfreq):
                    gfi = fi + ts.freq.local_offset[0] # global freq index
                    vis = ts.local_vis[ind, fi, pi, :] # only I
                    vis_mask = ts.local_vis_mask[ind, fi, pi, :] # only I
                    closure = []
                    cache = dict()
                    for i, j, k in itertools.combinations(feedno, 3):
                        bi, vij = mk_cache(cache, (i, j), bls, vis)
                        if vis_mask[bi]:
                            continue

                        bi, vjk = mk_cache(cache, (j, k), bls, vis)
                        if vis_mask[bi]:
                            continue

                        bi, vki = mk_cache(cache, (k, i), bls, vis)
                        if vis_mask[bi]:
                            continue

                        # closure.append(np.angle(vij, True) + np.angle(vjk, True) + np.angle(vki, True)) # in degree, have 360 deg problem
                        c = lambda x: np.complex128(x) # complex128 to avoid overflow in the product
                        ang = np.angle(c(vij) * c(vjk) * c(vki), True) # in degree
                        closure.append(ang) # in degree

                    # save closure phase to file
                    file_name = '%s_%d_%s.hdf5' % (file_prefix, gfi, pol[pi])
                    if tag_output_iter:
                        file_name = output_path(file_name, iteration=self.iteration)
                    else:
                        file_name = output_path(file_name)
                    with h5py.File(file_name, 'w') as f:
                        f.create_dataset('closure_phase', data=np.array(closure))

                    if plot_closure and gfi in freq_plt:
                        # plot all closure phase
                        plt.figure()
                        plt.plot(closure, 'o')
                        fig_name = '%s_all_%d_%s.png' % (fig_prefix, gfi, pol[pi])
                        if tag_output_iter:
                            fig_name = output_path(fig_name, iteration=self.iteration)
                        else:
                            fig_name = output_path(fig_name)
                        plt.savefig(fig_name)
                        plt.close()

                        # plot histogram of closure phase
                        # histogram
                        plt.figure()
                        data = plt.hist(closure, bins=bins)
                        plt.xlabel('Closure phase / degree')

                        if gauss_fit:
                            # Generate data from bins as a set of points
                            x = [0.5 * (data[1][i] + data[1][i+1]) for i in xrange(len(data[1])-1)]
                            y = data[0]

                            popt, pcov = optimize.curve_fit(f, x, y)
                            A, b, c = popt

                            xmax = max(abs(x[0]), abs(x[-1]))
                            x_fit = np.linspace(-xmax, xmax, bins)
                            y_fit = f(x_fit, *popt)

                            lable = r'$a \, \exp{(- \frac{(x - \mu)^2} {2 \sigma^2})}$' + '\n\n' + r'$a = %f$' % A + '\n' + r'$\mu = %f$' % b + '\n' + r'$\sigma = %f$' % np.abs(c)
                            plt.plot(x_fit, y_fit, lw=2, color="r", label=lable)
                            plt.xlim(-xmax, xmax)
                            plt.legend()

                        fig_name = '%s_hist_%d_%s.png' % (fig_prefix, gfi, pol[pi])
                        if tag_output_iter:
                            fig_name = output_path(fig_name, iteration=self.iteration)
                        else:
                            fig_name = output_path(fig_name)
                        plt.savefig(fig_name)
                        plt.close()


        mpiutil.barrier()

        return super(Closure, self).process(ts)
