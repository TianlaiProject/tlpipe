import os
import sys
import pickle

import h5py
import numpy as np
# import healpy as hp

from caput import mpiutil

from cora.util import hputil

import itertools
from pathos.multiprocessing import ProcessingPool as Pool

from ..core import kltransform
from ..util import util


class Timestream(object):

    #============ Constructor etc. =====================

    def __init__(self, tsdir, tsname, beamtransfer, no_m_zero=True):
        """Create a new Timestream object.

        Parameters
        ----------
        tsdir : string
            Directory to create the Timestream in.
        tsname : string
            Name of the timestream.
        beamtransfer : drfit.core.beamtransfer.BeamTransfer
            BeamTransfer object containing the analysis products.
        """
        self.directory = os.path.abspath(tsdir)
        self.output_directory = '%s/%s' % (self.directory, tsname)
        self.tsname = tsname
        self.beamtransfer = beamtransfer
        self.no_m_zero = no_m_zero

    #====================================================


    #===== Accessing the BeamTransfer and Telescope =====

    @property
    def telescope(self):
        """The telescope object corresponding to this timestream.
        """
        return self.beamtransfer.telescope

    #====================================================


    #======== Fetch and generate the f-stream ===========


    def _fdir(self, fi):
        # Pattern to form the `freq` ordered file.
        pat = self.directory + "/timestream_f/" + util.natpattern(self.telescope.nfreq)
        return pat % fi


    def _ffile(self, fi):
        # Pattern to form the `freq` ordered file.
        return self._fdir(fi) + "/timestream.hdf5"

    @property
    def ntime(self):
        """Get the number of timesamples."""

        with h5py.File(self._ffile(0), 'r') as f:
            ntime = f.attrs['ntime']

        return ntime


    def timestream_f(self, fi):
        """Fetch the timestream for a given frequency.

        Parameters
        ----------
        fi : integer
            Frequency to load.

        Returns
        -------
        timestream : np.ndarray[npairs, ntime]
            The visibility timestream.
        """

        with h5py.File(self._ffile(fi), 'r') as f:
            ts = f['timestream'][:]
        return ts

    #====================================================


    #======== Fetch and generate the m-modes ============

    def _mdir(self, mi):
        # Pattern to form the `m` ordered file.
        pat = self.output_directory + "/mmodes/" + util.natpattern(self.telescope.mmax)
        return pat % abs(mi)


    def _mfile(self, mi):
        # Pattern to form the `m` ordered file.
        return self._mdir(mi) + '/mode.hdf5'


    def mmode(self, mi):
        """Fetch the timestream m-mode for a specified m.

        Parameters
        ----------
        mi : integer
            m-mode to load.

        Returns
        -------
        timestream : np.ndarray[nfreq, pm, npairs]
            The visibility m-modes.
        """

        with h5py.File(self._mfile(mi), 'r') as f:
            return f['mmode'][:]


    def generate_mmodes(self):
        """Calculate the m-modes corresponding to the Timestream.

        Perform an MPI transpose for efficiency.
        """


        if os.path.exists(self.output_directory + "/mmodes/COMPLETED_M"):
            if mpiutil.rank0:
                print("******* m-files already generated ********")
            return

        tel = self.telescope
        mmax = tel.mmax
        nfreq = tel.nfreq

        lfreq, sfreq, efreq = mpiutil.split_local(nfreq)
        lm, sm, em = mpiutil.split_local(mmax + 1)

        # Load in the local frequencies of the time stream
        tstream = np.zeros((lfreq, tel.npairs, self.ntime), dtype=np.complex128)
        for lfi, fi in enumerate(range(sfreq, efreq)):
            tstream[lfi] = self.timestream_f(fi)

        # FFT to calculate the m-modes for the timestream
        row_mmodes = np.fft.fft(tstream, axis=-1) / self.ntime

        ## Combine positive and negative m parts.
        row_mpairs = np.zeros((lfreq, 2, tel.npairs, mmax+1), dtype=np.complex128)

        row_mpairs[:, 0, ..., 0] = row_mmodes[..., 0]
        for mi in range(1, mmax+1):
            row_mpairs[:, 0, ..., mi] = row_mmodes[...,  mi]
            row_mpairs[:, 1, ..., mi] = row_mmodes[..., -mi].conj()

        # Transpose to get the entirety of an m-mode on each process (i.e. all frequencies)
        col_mmodes = mpiutil.transpose_blocks(row_mpairs, (nfreq, 2, tel.npairs, mmax + 1))

        # Transpose the local section to make the m's first
        col_mmodes = np.transpose(col_mmodes, (3, 0, 1, 2))

        for lmi, mi in enumerate(range(sm, em)):

            # Make directory for each m-mode
            if not os.path.exists(self._mdir(mi)):
                os.makedirs(self._mdir(mi))

            # Create the m-file and save the result.
            with h5py.File(self._mfile(mi), 'w') as f:
                f.create_dataset('/mmode', data=col_mmodes[lmi])
                f.attrs['m'] = mi

        if mpiutil.rank0:

            # Make file marker that the m's have been correctly generated:
            open(self.output_directory + "/mmodes/COMPLETED_M", 'a').close()

        mpiutil.barrier()

    #====================================================


    #======== Make and fetch SVD m-modes ================

    def _svdfile(self, mi):
        # Pattern to form the `m` ordered file.
        return self._mdir(mi) + '/svd.hdf5'


    def mmode_svd(self, mi):
        """Fetch the SVD m-mode for a specified m.

        Parameters
        ----------
        mi : integer
            m-mode to load.

        Returns
        -------
        svd_mode : np.ndarray[nfreq, pm, npairs]
            The visibility m-modes.
        """

        with h5py.File(self._svdfile(mi), 'r') as f:
            if f['mmode_svd'].shape[0] == 0:
                return np.zeros((0,), dtype=np.complex128)
            else:
                return f['mmode_svd'][:]


    def generate_mmodes_svd(self):
        """Generate the SVD modes for the Timestream.
        """

        # Iterate over local m's, project mode and save to disk.
        for mi in mpiutil.mpirange(self.telescope.mmax + 1, method='rand'):

            if os.path.exists(self._svdfile(mi)):
                print("File %s exists. Skipping..." % self._svdfile(mi))
                continue

            tm = self.mmode(mi).reshape(self.telescope.nfreq, 2*self.telescope.npairs)
            svdm = self.beamtransfer.project_vector_telescope_to_svd(mi, tm)

            with h5py.File(self._svdfile(mi), 'w') as f:
                f.create_dataset('mmode_svd', data=svdm)
                f.attrs['m'] = mi

        mpiutil.barrier()


    #====================================================


    #======== Make map from uncleaned stream ============

    def mapmake_full(self, nside, mapname, nbin=None, dirty=False, method='svd', normalize=True, threshold=1.0e3, eps=0.01, correct_order=0, prior_map_file=None, save_alm=False, tk_deconv=False, map_to_deconv=None, loop_factor=0.1, n_iter=100):

        nfreq = self.telescope.nfreq
        if nbin is None:
            nbin = nfreq
        else:
            if (nbin < 1 or nbin > nfreq): # invalid nbin
                nbin = nfreq
            else:
                nbin = int(nbin)

        if prior_map_file is not None:
            # read in the prior sky map
            with h5py.File(prior_map_file, 'r') as f:
                prior_map = f['map'][:] # shape (nbin, npol, npix)

            # alm of the prior map
            alm0 = hputil.sphtrans_sky(prior_map, lmax=self.telescope.lmax).reshape(nbin, self.telescope.num_pol_sky, self.telescope.lmax+1, self.telescope.lmax+1) # shape (nbin, npol, lmax+1, lmax+1)
        else:
            alm0 = None

        def _make_alm(mi):

            print("Making %i" % mi)

            mmode = self.mmode(mi)
            if dirty:
                sphmode = self.beamtransfer.project_vector_backward_dirty(mi, mmode, nbin, normalize, threshold)
            else:
                if method == 'svd':
                    sphmode = self.beamtransfer.project_vector_telescope_to_sky(mi, mmode, nbin)
                elif method == 'tk':
                    # sphmode = self.beamtransfer.project_vector_telescope_to_sky_tk(mi, mmode, nbin, eps=eps)
                    mmode0 = alm0[:, :, :, mi] if alm0 is not None else None
                    sphmode = self.beamtransfer.project_vector_telescope_to_sky_tk(mi, mmode, nbin, eps=eps, correct_order=correct_order, mmode0=mmode0)
                else:
                    raise ValueError('Unknown map-making method %s' % method)

            return sphmode

        if not (method == 'tk' and tk_deconv and map_to_deconv is not None):
            alm_list = mpiutil.parallel_map(_make_alm, list(range(self.telescope.mmax + 1)), root=0, method='rand')

        if mpiutil.rank0:

            # get center freq of each bin
            n, s, e = mpiutil.split_m(nfreq, nbin)
            cfreqs = np.array([ self.beamtransfer.telescope.frequencies[(s[i]+e[i])//2] for i in range(nbin) ])

            alm = np.zeros((nbin, self.telescope.num_pol_sky, self.telescope.lmax + 1,
                            self.telescope.lmax + 1), dtype=np.complex128)

            if not (method == 'tk' and tk_deconv and map_to_deconv is not None):
                # mlist = range(1 if self.no_m_zero else 0, self.telescope.mmax + 1)
                mlist = list(range(self.telescope.mmax + 1))

                for mi in mlist:

                    alm[..., mi] = alm_list[mi]

                if save_alm:
                    alm1 = alm.copy()

                skymap = hputil.sphtrans_inv_sky(alm, nside)
            else:

                if map_to_deconv is not None:
                    # read in the skymap to deconv
                    with h5py.File(map_to_deconv, 'r') as f:
                        skymap = f['map'][:]

            if method == 'tk' and tk_deconv:
                residual_map = skymap.copy() # does not change the original sky map
                clean_map = np.zeros_like(skymap)
                max_inds = np.zeros(nfreq, dtype=int)
                max_vals = np.zeros(nfreq, dtype=skymap.dtype)
                for ii in range(n_iter):
                    print('deconv: %d of %d...' % (ii, n_iter))
                    sys.stdout.flush()
                    sys.stderr.flush()
                    alm_psf = np.zeros((nbin, self.telescope.num_pol_sky, self.telescope.lmax + 1, self.telescope.lmax + 1), dtype=np.complex128)
                    for fi in range(nfreq):
                        max_i = np.argmax(residual_map[fi, 0]) # index of the max pixel
                        max_inds[fi] = max_i
                        max_vals[fi] = residual_map[fi, 0, max_i]
                        # build a ppint source map with value 1.0 in this pixel
                        map_unit_ps = np.zeros_like(residual_map[fi, 0])
                        map_unit_ps[max_i] = 1.0 # a unit point source
                        # get a_lm of this point source map
                        alm_unit_ps = hputil.sphtrans_real(map_unit_ps, lmax=self.telescope.lmax) # (lmax, lmax), a_lm of a single pixel with value 1.0
                        # NOTE unit_ps and alm_ps is frequency-independent

                        # compute alm PSF for only pol I
                        for mi in range(self.telescope.mmax + 1):
                            alm_psf[fi, 0, :, mi] = self.beamtransfer.tk_deconv(fi, mi, alm_unit_ps[:, mi], eps=eps)
                        # num_m = self.telescope.mmax + 1
                        # p = Pool(20)
                        # alm_psf[fi, 0, :, :num_m] = np.array(p.map(self.beamtransfer.tk_deconv, itertools.repeat(fi, num_m), range(num_m), alm_unit_ps[:, range(num_m)].T, itertools.repeat(eps, num_m))).T

                    # compute PSF map for alm_psf
                    unit_psf = hputil.sphtrans_inv_sky(alm_psf, nside)
                    ### TODO: check hte compute unit_psf

                    # ################################################################################
                    # # degrade to nside 64
                    # import healpy as hp
                    # res_unit_psf = np.zeros_like(unit_psf)
                    # nf, npol, npix = unit_psf.shape
                    # nside = hp.npix2nside(npix)
                    # for fi in range(nf):
                    #     for pi in range(npol):
                    #         res_unit_psf[fi, pi] = unit_psf[fi, pi] - hp.ud_grade(hp.ud_grade(unit_psf[fi, pi], 64), nside)
                    # unit_psf = res_unit_psf  # replace the original unit_psf
                    # ################################################################################

                    # deconv
                    for fi in range(nfreq):
                        clean_ind = max_inds[fi]
                        clean_component = loop_factor * max_vals[fi]
                        clean_map[fi, 0, clean_ind] +=  clean_component
                        residual_map[fi, 0] -= clean_component * unit_psf[fi, 0] # subtract a fraction of the PSF


            # if method == 'tk' and tk_deconv:
            #     tmp_map = hputil.sphtrans_inv_sky(alm, nside)
            #     clean_map = np.zeros_like(tmp_map)
            #     for ii in range(n_iter):
            #         for fi in range(nfreq):
            #             max_i = np.argmax(tmp_map[fi, 0])
            #             # theta, phi = hp.pix2ang(nside, max_i)
            #             tmp_ps = np.zeros_like(tmp_map[fi, 0])
            #             tmp_ps[max_i] = tmp_map[max_i]
            #             # record the clean component
            #             clean_map[max_i] += loop_factor * tmp_map[max_i]
            #             ps_alm = hputil.sphtrans_real(tmp_ps) # (lmax, lmax)
            #             for mi in range(self.telescope.mmax + 1):
            #                 alm_ps = self.beamtransfer.tk_deconv(fi, mi, ps_alm[:, mi], eps=eps)
            #                 alm[fi, 0, :, mi] -= loop_factor * alm_ps
            #         tmp_map = hputil.sphtrans_inv_sky(alm, nside)
            #     clean_map += tmp_map


        if mpiutil.rank0:

            if not (method == 'tk' and tk_deconv and map_to_deconv is not None):
                if self.no_m_zero:
                    alm[:, :, :, 0] = 0

                    alm[:, :, 100:, 1] = 0

                skymap = hputil.sphtrans_inv_sky(alm, nside)

                with h5py.File(self.output_directory + '/' + mapname, 'w') as f:
                    f.create_dataset('/map', data=skymap)
                    f.attrs['dim'] = 'freq, pol, pix'
                    f.attrs['frequency'] = cfreqs
                    f.attrs['polarization'] = np.string_(['I', 'Q', 'U', 'V'])[:self.beamtransfer.telescope.num_pol_sky] # np.string_ for python 3

                    if save_alm:
                        f.create_dataset('/alm', data=alm1)
                        f.attrs['dim'] = 'freq, pol, l, m'
                        f.attrs['frequency'] = cfreqs
                        f.attrs['polarization'] = np.string_(['I', 'Q', 'U', 'V'])[:self.beamtransfer.telescope.num_pol_sky] # np.string_ for python 3

            if method == 'tk' and tk_deconv:
                # save clean_map and residual_map
                with h5py.File(self.output_directory + '/' + 'deconv_map.hdf5', 'w') as f:
                    f.create_dataset('clean_map', data=clean_map)
                    f.create_dataset('residual_map', data=residual_map)

        mpiutil.barrier()


    def mapmake_svd(self, nside, mapname):

        self.generate_mmodes_svd()

        def _make_alm(mi):

            svdmode = self.mmode_svd(mi)

            sphmode = self.beamtransfer.project_vector_svd_to_sky(mi, svdmode)

            return sphmode

        alm_list = mpiutil.parallel_map(_make_alm, list(range(self.telescope.mmax + 1)), root=0, method='rand')

        if mpiutil.rank0:

            alm = np.zeros((self.telescope.nfreq, self.telescope.num_pol_sky, self.telescope.lmax + 1,
                            self.telescope.lmax + 1), dtype=np.complex128)

            mlist = list(range(1 if self.no_m_zero else 0, self.telescope.mmax + 1))

            for mi in mlist:

                alm[..., mi] = alm_list[mi]

            skymap = hputil.sphtrans_inv_sky(alm, nside)

            with h5py.File(self.output_directory + '/' + mapname, 'w') as f:
                f.create_dataset('/map', data=skymap)
                f.attrs['frequency'] = self.beamtransfer.telescope.frequencies
                f.attrs['polarization'] = np.string_(['I', 'Q', 'U', 'V'])[:self.beamtransfer.telescope.num_pol_sky] # np.string_ for python 3

        mpiutil.barrier()

    #====================================================


    #========== Project into KL-mode basis ==============

    def set_kltransform(self, klname, kl):

        self.klname = klname
        self.kl = kl
        self.klthreshold = kl.threshold

    def _klfile(self, mi):
        # Pattern to form the `m` ordered file.
        return self._mdir(mi) + ('/klmode_%s_%f.hdf5' % (self.klname, self.klthreshold))




    def mmode_kl(self, mi):
        with h5py.File(self._klfile(mi), 'r') as f:
            if f['mmode_kl'].shape[0] == 0:
                return np.zeros((0,), dtype=np.complex128)
            else:
                return f['mmode_kl'][:]


    def generate_mmodes_kl(self):
        """Generate the KL modes for the Timestream.
        """

        kl = self.kl

        # Iterate over local m's, project mode and save to disk.
        for mi in mpiutil.mpirange(self.telescope.mmax + 1, method='rand'):

            if os.path.exists(self._klfile(mi)):
                print("File %s exists. Skipping..." % self._klfile(mi))
                continue

            svdm = self.mmode_svd(mi) #.reshape(self.telescope.nfreq, 2*self.telescope.npairs)
            #svdm = self.beamtransfer.project_vector_telescope_to_svd(mi, tm)

            klm = kl.project_vector_svd_to_kl(mi, svdm, threshold=self.klthreshold)

            with h5py.File(self._klfile(mi), 'w') as f:
                f.create_dataset('mmode_kl', data=klm)
                f.attrs['m'] = mi

        mpiutil.barrier()


    def collect_mmodes_kl(self):

        def evfunc(mi):
            evf = np.zeros(self.beamtransfer.ndofmax, dtype=np.complex128)

            ev = self.mmode_kl(mi)
            if ev.size > 0:
                evf[-ev.size:] = ev

            return evf

        if mpiutil.rank0:
            print("Creating eigenvalues file (process 0 only).")

        mlist = list(range(self.telescope.mmax+1))
        shape = (self.beamtransfer.ndofmax, )
        evarray = kltransform.collect_m_array(mlist, evfunc, shape, np.complex128)

        if mpiutil.rank0:
            fname =  self.output_directory + ("/klmodes_%s_%f.hdf5"% (self.klname, self.klthreshold))
            if os.path.exists(fname):
                print("File: %s exists. Skipping..." % (fname))
                return

            with h5py.File(fname, 'w') as f:
                f.create_dataset('evals', data=evarray)




    def fake_kl_data(self):

        kl = self.manager.kltransforms[self.klname]

        # Iterate over local m's, project mode and save to disk.
        for mi in mpiutil.mpirange(self.telescope.mmax + 1, method='rand'):

            evals = kl.evals_m(mi)

            if evals is None:
                klmode = np.array([], dtype=np.complex128)
            else:
                modeamp = ((evals + 1.0) / 2.0)**0.5
                klmode = modeamp * (np.array([1.0, 1.0J]) * np.random.standard_normal((modeamp.shape[0], 2))).sum(axis=1)


            with h5py.File(self._klfile(mi), 'w') as f:
                f.create_dataset('mmode_kl', data=klmode)
                f.attrs['m'] = mi

        mpiutil.barrier()


    def mapmake_kl(self, nside, mapname, wiener=False):

        mapfile = self.output_directory + '/' + mapname

        if os.path.exists(mapfile):
            if mpiutil.rank0:
                print("File %s exists. Skipping...")
            return

        kl = self.manager.kltransforms[self.klname]

        if not kl.inverse:
            raise Exception("Need the inverse to make a meaningful map.")

        def _make_alm(mi):
            print("Making %i" % mi)

            klmode = self.mmode_kl(mi)

            if wiener:
                evals = kl.evals_m(mi, self.klthreshold)

                if evals is not None:
                    klmode *= (evals / (1.0 + evals))

            isvdmode = kl.project_vector_kl_to_svd(mi, klmode, threshold=self.klthreshold)

            sphmode = self.beamtransfer.project_vector_svd_to_sky(mi, isvdmode)

            return sphmode

        alm_list = mpiutil.parallel_map(_make_alm, list(range(self.telescope.mmax + 1)), root=0, method='rand')

        if mpiutil.rank0:

            alm = np.zeros((self.telescope.nfreq, self.telescope.num_pol_sky, self.telescope.lmax + 1,
                            self.telescope.lmax + 1), dtype=np.complex128)

            # Determine whether to use m=0 or not
            mlist = list(range(1 if self.no_m_zero else 0, self.telescope.mmax + 1))

            for mi in mlist:

                alm[..., mi] = alm_list[mi]

            skymap = hputil.sphtrans_inv_sky(alm, nside)

            with h5py.File(mapfile, 'w') as f:
                f.create_dataset('/map', data=skymap)
                f.attrs['frequency'] = self.beamtransfer.telescope.frequencies
                f.attrs['polarization'] = np.string_(['I', 'Q', 'U', 'V'])[:self.beamtransfer.telescope.num_pol_sky] # np.string_ for python 3

        mpiutil.barrier()

    #====================================================


    #======= Estimate powerspectrum from data ===========


    @property
    def _psfile(self):
        # Pattern to form the `m` ordered file.
        return self.output_directory + ('/ps_%s.hdf5' % self.psname)



    def set_psestimator(self, psname):
        self.psname = psname


    def powerspectrum(self):

        import scipy.linalg as la


        if os.path.exists(self._psfile):
            print("File %s exists. Skipping..." % self._psfile)
            return

        ps = self.manager.psestimators[self.psname]
        ps.genbands()

        def _q_estimate(mi):

            return ps.q_estimator(mi, self.mmode_kl(mi))

        # Determine whether to use m=0 or not
        mlist = list(range(1 if self.no_m_zero else 0, self.telescope.mmax + 1))
        qvals = mpiutil.parallel_map(_q_estimate, mlist)

        qtotal = np.array(qvals).sum(axis=0)

        fisher, bias = ps.fisher_bias()

        powerspectrum =  np.dot(la.inv(fisher), qtotal - bias)


        if mpiutil.rank0:
            with h5py.File(self._psfile, 'w') as f:


                cv = la.inv(fisher)
                err = cv.diagonal()**0.5
                cr = cv / np.outer(err, err)

                f.create_dataset('fisher/', data=fisher)
#                f.create_dataset('bias/', data=self.bias)
                f.create_dataset('covariance/', data=cv)
                f.create_dataset('error/', data=err)
                f.create_dataset('correlation/', data=cr)

                f.create_dataset('bandpower/', data=ps.band_power)
                #f.create_dataset('k_start/', data=ps.k_start)
                #f.create_dataset('k_end/', data=ps.k_end)
                #f.create_dataset('k_center/', data=ps.k_center)
                #f.create_dataset('psvalues/', data=ps.psvalues)

                f.create_dataset('powerspectrum', data=powerspectrum)

        # Delete cache of bands for memory reasons
        del ps.clarray
        ps.clarray = None

        mpiutil.barrier()

        return powerspectrum




    #====================================================


    #======== Load and save the Pickle files ============

    def __getstate__(self):
        ## Remove the attributes we don't want pickled.
        state = self.__dict__.copy()

        for key in self.__dict__:
            #if (key in delkeys) or (key[0] == "_"):
            if (key[0] == "_"):
                del state[key]

        return state


    @property
    def _picklefile(self):
        # The filename for the pickled telescope
        return self.output_directory + "/timestreamobject.pickle"


    def save(self):
        """Save out the Timestream object information."""

        # Save pickled telescope object
        if mpiutil.rank0:
            with open(self._picklefile, 'wb') as f:
                print("=== Saving Timestream object. ===")
                pickle.dump(self, f)


    @classmethod
    def load(cls, tsdir, tsname):
        """Load the Timestream object from disk.

        Parameters
        ----------
        tsdir : string
            Name of the directory containing the Timestream object.
        """

        # Create temporary object to extract picklefile property
        tmp_obj = cls(tsdir, tsname, 'bt')

        with open(tmp_obj._picklefile, 'rb') as f:
            print("=== Loading Timestream object. ===")
            return pickle.load(f)

    #====================================================



def cross_powerspectrum(timestreams, psname, psfile):

    import scipy.linalg as la

    if os.path.exists(psfile):
        print("File %s exists. Skipping..." % psfile)
        return

    products = timestreams[0].manager

    ps = products.psestimators[psname]
    ps.genbands()

    nstream = len(timestreams)

    def _q_estimate(mi):

        qp = np.zeros((nstream, nstream, ps.nbands), dtype=np.float64)

        for ti in range(nstream):
            for tj in range(ti+1, nstream):

                print("Making m=%i (%i, %i)" % (mi, ti, tj))

                si = timestreams[ti]
                sj = timestreams[tj]

                qp[ti, tj] = ps.q_estimator(mi, si.mmode_kl(mi), sj.mmode_kl(mi))
                qp[tj, ti] = qp[ti, tj]

        return qp

    # Determine whether to use m=0 or not
    mlist = list(range(1 if timestreams[0].no_m_zero else 0, products.telescope.mmax + 1))
    qvals = mpiutil.parallel_map(_q_estimate, mlist)

    qtotal = np.array(qvals).sum(axis=0)

    fisher, bias = ps.fisher_bias()

    # Subtract bias and reshape into new array
    qtotal = (qtotal - bias).reshape(nstream**2, ps.nbands).T

    powerspectrum =  np.dot(la.inv(fisher), qtotal)
    powerspectrum = powerspectrum.T.reshape(nstream, nstream, ps.nbands)


    if mpiutil.rank0:
        with h5py.File(psfile, 'w') as f:

            cv = la.inv(fisher)
            err = cv.diagonal()**0.5
            cr = cv / np.outer(err, err)

            f.create_dataset('fisher/', data=fisher)
#                f.create_dataset('bias/', data=self.bias)
            f.create_dataset('covariance/', data=cv)
            f.create_dataset('error/', data=err)
            f.create_dataset('correlation/', data=cr)

            f.create_dataset('bandpower/', data=ps.band_power)
            #f.create_dataset('k_start/', data=ps.k_start)
            #f.create_dataset('k_end/', data=ps.k_end)
            #f.create_dataset('k_center/', data=ps.k_center)
            #f.create_dataset('psvalues/', data=ps.psvalues)

            f.create_dataset('powerspectrum', data=powerspectrum)

    # Delete cache of bands for memory reasons
    del ps.clarray
    ps.clarray = None

    mpiutil.barrier()

    return powerspectrum





# kwargs is to absorb any extra params
def simulate(beamtransfer, outdir, tsname, maps=[], ndays=None, resolution=0, add_noise=True, seed=None, **kwargs):
    """Create a simulated timestream and save it to disk.

    Parameters
    ----------
    m : ProductManager object
        Products of telescope to simulate.
    outdir : directoryname
        Directory that we will save the timestream into.
    maps : list
        List of map filenames. The sum of these form the simulated sky.
    ndays : int, optional
        Number of days of observation. Setting `ndays = None` (default) uses
        the default stored in the telescope object; `ndays = 0`, assumes the
        observation time is infinite so that the noise is zero.
    resolution : scalar, optional
        Approximate time resolution in seconds. Setting `resolution = 0`
        (default) calculates the value from the mmax.

    Returns
    -------
    timestream : Timestream
    """

    # Create timestream object
    tstream = Timestream(outdir, tsname, beamtransfer)

    # Make directory if required
    try:
        os.makedirs(tstream.output_directory)
    except OSError:
         # directory exists
         pass

    if mpiutil.rank0:
        tstream.save()

    ## Read in telescope system
    bt = beamtransfer
    tel = bt.telescope

    lmax = tel.lmax
    mmax = tel.mmax
    nfreq = tel.nfreq
    npol = tel.num_pol_sky

    projmaps = (len(maps) > 0)

    lfreq, sfreq, efreq = mpiutil.split_local(nfreq)
    local_freq = list(range(sfreq, efreq))

    lm, sm, em = mpiutil.split_local(mmax + 1)

    # If ndays is not set use the default value.
    if ndays is None:
        ndays = tel.ndays

    # Calculate the number of timesamples from the resolution
    if resolution == 0:
        # Set the minimum resolution required for the sky.
        ntime = 2*mmax+1
    else:
        # Set the cl
        ntime = int(np.round(24 * 3600.0 / resolution))


    col_vis = np.zeros((tel.npairs, lfreq, ntime), dtype=np.complex128)

    ## If we want to add maps use the m-mode formalism to project a skymap
    ## into visibility space.

    if projmaps:

        # Load file to find out the map shapes.
        with h5py.File(maps[0], 'r') as f:
            mapshape = f['map'].shape

        if lfreq > 0:

            # Allocate array to store the local frequencies
            row_map = np.zeros((lfreq,) + mapshape[1:], dtype=np.float64)

            # Read in and sum up the local frequencies of the supplied maps.
            for mapfile in maps:
                with h5py.File(mapfile, 'r') as f:
                    row_map += f['map'][sfreq:efreq]

            # Calculate the alm's for the local sections
            row_alm = hputil.sphtrans_sky(row_map, lmax=lmax).reshape((lfreq, npol * (lmax+1), lmax+1))

        else:
            row_alm = np.zeros((lfreq, npol * (lmax+1), lmax+1), dtype=np.complex128)

        # Perform the transposition to distribute different m's across processes. Neat
        # tip, putting a shorter value for the number of columns, trims the array at
        # the same time
        col_alm = mpiutil.transpose_blocks(row_alm, (nfreq, npol * (lmax+1), mmax+1))

        # Transpose and reshape to shift m index first.
        col_alm = np.transpose(col_alm, (2, 0, 1)).reshape(lm, nfreq, npol, lmax+1)

        # Create storage for visibility data
        vis_data = np.zeros((lm, nfreq, bt.ntel), dtype=np.complex128)

        # Iterate over m's local to this process and generate the corresponding
        # visibilities
        for mp, mi in enumerate(range(sm, em)):
            vis_data[mp] = bt.project_vector_sky_to_telescope(mi, col_alm[mp])

        # Rearrange axes such that frequency is last (as we want to divide
        # frequencies across processors)
        row_vis = vis_data.transpose((0, 2, 1))#.reshape((lm * bt.ntel, nfreq))

        # Parallel transpose to get all m's back onto the same processor
        col_vis_tmp = mpiutil.transpose_blocks(row_vis, ((mmax+1), bt.ntel, nfreq))
        col_vis_tmp = col_vis_tmp.reshape(mmax + 1, 2, tel.npairs, lfreq)


        # Transpose the local section to make the m's the last axis and unwrap the
        # positive and negative m at the same time.
        col_vis[..., 0] = col_vis_tmp[0, 0]
        for mi in range(1, mmax+1):
            col_vis[...,  mi] = col_vis_tmp[mi, 0]
            col_vis[..., -mi] = col_vis_tmp[mi, 1].conj()  # Conjugate only (not (-1)**m - see paper)


        del col_vis_tmp

    ## If we're simulating noise, create a realisation and add it to col_vis
    if ndays > 0:

        if lfreq > 0:
            # Fetch the noise powerspectrum
            noise_ps = tel.noisepower(np.arange(tel.npairs)[:, np.newaxis], np.array(local_freq)[np.newaxis, :], ndays=ndays).reshape(tel.npairs, lfreq)[:, :, np.newaxis]


            # Seed random number generator to give consistent noise
            if seed is not None:
                # Must include rank such that we don't have massive power deficit from correlated noise
                np.random.seed(seed + mpiutil.rank)

            # Create and weight complex noise coefficients
            noise_vis = (np.array([1.0, 1.0J]) * np.random.standard_normal(col_vis.shape + (2,))).sum(axis=-1)
            noise_vis *= (noise_ps / 2.0)**0.5

            # Reset RNG
            if seed is not None:
                np.random.seed()

            # Add into main noise sims
            col_vis += noise_vis

            del noise_vis

        mpiutil.barrier()


    # Fourier transform m-modes back to get timestream.
    vis_stream = np.fft.ifft(col_vis, axis=-1) * ntime
    vis_stream = vis_stream.reshape(tel.npairs, lfreq, ntime)

    # The time samples the visibility is calculated at
    tphi = np.linspace(0, 2*np.pi, ntime, endpoint=False)

    ## Iterate over the local frequencies and write them to disk.
    for lfi, fi in enumerate(local_freq):

        # Make directory if required
        if not os.path.exists(tstream._fdir(fi)):
            os.makedirs(tstream._fdir(fi))

        # Write file contents
        with h5py.File(tstream._ffile(fi), 'w') as f:

            # Timestream data
            f.create_dataset('/timestream', data=vis_stream[:, lfi])
            f.create_dataset('/phi', data=tphi)

            # Telescope layout data
            f.create_dataset('/feedmap', data=tel.feedmap)
            f.create_dataset('/feedconj', data=tel.feedconj)
            f.create_dataset('/feedmask', data=tel.feedmask)
            f.create_dataset('/uniquepairs', data=tel.uniquepairs)
            f.create_dataset('/baselines', data=tel.baselines)

            # Write metadata
            f.attrs['beamtransfer_path'] = os.path.abspath(bt.directory)
            f.attrs['ntime'] = ntime

    mpiutil.barrier()

    return tstream
