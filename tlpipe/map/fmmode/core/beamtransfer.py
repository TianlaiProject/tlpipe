"""
========================================================
Beam Transfer Matrices (:mod:`~drift.core.beamtransfer`)
========================================================

A class for calculating and managing Beam Transfer matrices

Classes
=======

.. autosummary::
    :toctree: generated/

    BeamTransfer

"""

try:
    import cPickle as pickle
except ImportError:
    import pickle

import os
import time
import warnings

import numpy as np
import scipy.linalg as la
import h5py

from caput import mpiutil
from caput.mpiarray import MPIArray
from tlpipe.map.fmmode.util import util



class BeamTransfer(object):
    """A class for reading and writing Beam Transfer matrices from disk.

    In addition this provides methods for projecting vectors and matrices
    between the sky and the telescope basis.

    Parameters
    ----------
    directory : string
        Path of directory to read and write Beam Transfers from.
    telescope : drift.core.telescope.TransitTelescope, optional
        Telescope object to use for calculation. If `None` (default), try to
        load a cached version from the given directory.

    Attributes
    ----------
    svcut
    polsvcut
    ntel
    nsky
    nfreq
    svd_len
    ndofmax


    Methods
    -------
    ndof
    beam_m
    invbeam_m
    beam_svd
    beam_ut
    invbeam_svd
    beam_singularvalues
    generate
    project_vector_sky_to_telescope
    project_vector_telescope_to_sky
    project_vector_sky_to_svd
    project_vector_svd_to_sky
    project_vector_telescope_to_svd
    project_matrix_sky_to_telescope
    project_matrix_sky_to_svd
    """


    def __init__(self, directory, telescope=None, gen_invbeam=True, noise_weight=True):

        self.directory = directory
        self.telescope = telescope
        self.gen_invbeam = gen_invbeam
        self.noise_weight = noise_weight

        # Create directory if required
        if mpiutil.rank0 and not os.path.exists(directory):
            os.makedirs(directory)

        mpiutil.barrier()

        if self.telescope == None and mpiutil.rank0:
            print "Attempting to read telescope from disk..."

            try:
                f = open(self._picklefile, 'r')
                self.telescope = pickle.load(f)
            except IOError, UnpicklingError:
                raise Exception("Could not load Telescope object from disk.")


    #====== Properties giving internal filenames =======

    @property
    def _picklefile(self):
        # The filename for the pickled telescope
        return self.directory + "/telescopeobject.pickle"

    @property
    def _tel_datafile(self):
        # File to save telescope frequencies and baselines
        return self.directory + '/telescope_data.hdf5'

    @property
    def _mdir(self):
        # Directory to save `m` ordered beam transfer matrix files
        return self.directory + '/beam_m/'

    def _mfile(self, mi):
        # Pattern to form the `m` ordered beam transfer matrix file
        pat = self._mdir + 'beam_%s.hdf5' % util.intpattern(self.telescope.mmax)
        return pat % mi

    @property
    def _inv_mdir(self):
        # Directory to save `m` ordered inverse beam transfer matrix files
        return self.directory + '/inv_beam_m/'

    def _inv_mfile(self, mi):
        # Pattern to form the `m` ordered inverse beam transfer matrix file
        pat = self._inv_mdir + 'inv_beam_%s.hdf5' % util.natpattern(self.telescope.mmax)
        return pat % abs(mi)

    #===================================================


    @property
    def _telescope_pickle(self):
        # The pickled telescope object
        return pickle.dumps(self.telescope)



    #===================================================



    #====== Loading m-order beams ======================

    def _load_beam_m(self, mi, fi=None):
        ## Read in beam from disk
        if mi == 0:
            with h5py.File(self._mfile(0), 'r') as mfile:

                # If fi is None, return all frequency blocks. Otherwise just the one requested.
                if fi is None:
                    beam = mfile['beam_m'][:]
                else:
                    beam = mfile['beam_m'][fi][:]
        elif mi > 0:
            with h5py.File(self._mfile(mi), 'r') as mfile1, h5py.File(self._mfile(-mi), 'r') as mfile2:

                # If fi is None, return all frequency blocks. Otherwise just the one requested.
                if fi is None:
                    ### REMEMBER the conj for negative m
                    beam = np.concatenate([mfile1['beam_m'][:], mfile2['beam_m'][:].conj()], axis=1) # concatenate along the bl axis
                else:
                    beam = np.concatenate([mfile1['beam_m'][fi][:], mfile2['beam_m'][fi][:].conj()])
        else:
            raise ValueError('mi must greater than or equal to 0')

        return beam


    def beam_m(self, mi, fi=None):
        """Fetch the beam transfer matrix for a given m.

        Parameters
        ----------
        mi : integer
            m-mode to fetch.
        fi : integer
            frequency block to fetch. fi=None (default) returns all.

        Returns
        -------
        beam : np.ndarray (nfreq, ntel, npol, ntheta) or (ntel, npol, ntheta).
        """

        beam = self._load_beam_m(mi, fi=fi)
        # multiply sin(theta)
        beam *= np.sin(np.radians(self.telescope.theta_values))
        # multiply constant 2 pi^2 / N
        theta_range = self.telescope.beam_theta_range
        N = 180.0 * self.telescope.theta_size / (theta_range[1] - theta_range[0])
        beam *= 2.0 * np.pi**2 / N

        return beam

    #===================================================



    #====== Pseudo-inverse beams =======================

    def invbeam_m(self, mi, fi=None):
        """Fetch the inverse beam transfer matrix for a given m.

        Parameters
        ----------
        mi : integer
            m-mode to fetch.
        fi : integer
            frequency block to fetch. fi=None (default) returns all.

        Returns
        -------
        beam : np.ndarray (nfreq, nsky, ntel) or (nsky, ntel).
        """

        if self.gen_invbeam:
            with h5py.File(self._inv_mfile(mi), 'r') as ifile:

                # If fi is None, return all frequency blocks. Otherwise just the one requested.
                if fi is None:
                    ibeam = ifile['ibeam_m'][:]
                else:
                    ibeam = ifile['ibeam_m'][fi][:]
        else:
            ibeam = self.compute_invbeam_m(mi)
            if fi is not None:
                ibeam = ibeam[fi]

        return ibeam

    #===================================================




    #====== Generation of all the cache files ==========

    def generate(self, regen=False):
        """Save out all beam transfer matrices to disk.

        Parameters
        ----------
        regen : boolean, optional
            Force regeneration even if cache files exist (default: False).
        """

        self._generate_dirs()
        self._generate_teldatafile(regen)

        st = time.time()
        self._generate_mfiles(regen)
        if self.gen_invbeam:
            self._generate_invbeam(regen)
        et = time.time()
        if mpiutil.rank0:
            print "***** Beam transfer matrices generation time: %f" % (et - st)

        # Save pickled telescope object
        if mpiutil.rank0:
            print
            print '=' * 80
            print "=== Saving Telescope object. ==="
            with open(self._picklefile, 'w') as f:
                pickle.dump(self.telescope, f)

        # If we're part of an MPI run, synchronise here.
        mpiutil.barrier()


    def _generate_dirs(self):
        ## Create all the directories required to store the beam transfers.

        if mpiutil.rank0:

            # Create main directory for beamtransfer
            if not os.path.exists(self.directory):
                os.makedirs(self.directory)

            # Create directories for m beams
            if not os.path.exists(self._mdir):
                os.makedirs(self._mdir)

            if self.gen_invbeam and not os.path.exists(self._inv_mdir):
                os.makedirs(self._inv_mdir)

        mpiutil.barrier()


    def _generate_teldatafile(self, regen=False):

        if mpiutil.rank0:
            if os.path.exists(self._tel_datafile) and not regen:
                print
                print '=' * 80
                print 'File %s exists. Skipping...' % self._tel_datafile
            else:
                print
                print '=' * 80
                print 'Crreate telescope data file %s...' % self._tel_datafile
                with h5py.File(self._tel_datafile, 'w') as f:
                    f.create_dataset('baselines', data=self.telescope.baselines)
                    f.create_dataset('frequencies', data=self.telescope.frequencies)
                    f.create_dataset('uniquepairs', data=self.telescope.uniquepairs)
                    f.create_dataset('allpairs', data=self.telescope.allpairs)
                    f.create_dataset('redundancy', data=self.telescope.redundancy)

        mpiutil.barrier()


    def _generate_mfiles(self, regen=False):

        completed_file = self._mdir + 'COMPLETED_BEAM'
        if os.path.exists(completed_file) and not regen:
            if mpiutil.rank0:
                print
                print '=' * 80
                print "******* Beam transfer m-files already generated ********"
            mpiutil.barrier()
            return

        if mpiutil.rank0:
            print
            print '=' * 80
            print 'Create beam transfer m-files...'

        st = time.time()

        # Calculate the Beam Transfer Matrices
        nfreq = self.telescope.nfreq
        nbl = self.telescope.nbase
        npol = self.telescope.num_pol_sky
        ntheta = self.telescope.theta_size
        nphi = self.telescope.phi_size

        # create file to save beam transfer matrices
        dsize = (nfreq, nbl, npol, ntheta)
        csize = (nfreq, 1, npol, ntheta)
        mmax = self.telescope.mmax
        ms = np.concatenate([np.arange(0, mmax+1), np.arange(-mmax, 0)])
        # get local section of m'th
        for ind, mi in enumerate(mpiutil.mpilist(ms, method='con')):
            with h5py.File(self._mfile(mi), 'w') as f:
                f.create_dataset('beam_m', dsize, chunks=csize, compression='lzf', dtype=np.complex128)
                f.attrs['m'] = mi

        # calculate the total memory needed for the transfer matrix
        total_memory = nfreq * nbl * npol * ntheta * nphi * 16.0 # Bytes, 16 for complex128
        limit = 1.0 # GB, memory limit for each process
        # make each process have maximum `limit` GB
        sigle_memory = limit * 2**30 # Bytes
        # how many chunks
        num_chunks = np.int(np.ceil(total_memory / (mpiutil.size * sigle_memory)))

        # split bls to num_chunks sections
        if nbl < num_chunks:
            warnings.warn('Could not split to %d chunks for %d baselines' % (num_chunks, nbl))
        num_chunks = min(num_chunks, nbl)
        num, start, end = mpiutil.split_m(nbl, num_chunks)
        for ci in range(num_chunks):
            if mpiutil.rank0:
                print "Starting chunk %i of %i" % (ci+1, num_chunks)

            tarray = self.telescope.transfer_matrices(np.arange(start[ci], end[ci]), np.arange(nfreq))
            tarray = MPIArray.wrap(tarray, axis=0)
            # redistribute along different m
            tarray = tarray.redistribute(axis=3)

            # save beam transfer matrices to file
            for ind, mi in enumerate(mpiutil.mpilist(ms, method='con')):
                with h5py.File(self._mfile(mi), 'r+') as f:
                    f['beam_m'][:, start[ci]:end[ci]] = tarray[..., ind].view(np.ndarray).reshape(nfreq, num[ci], npol, ntheta)

        mpiutil.barrier()

        et = time.time()

        if mpiutil.rank0:

            # Make file marker that the m's have been correctly generated:
            open(completed_file, 'a').close()

            # Print out timing
            print "=== Create beam transfer m-files took %f s ===" % (et - st)


    # noise_weight = True

    def compute_invbeam_m(self, mi):
        """Compute the inverse beam transfer matrix for the m-mode `mi`."""

        nfreq = self.telescope.nfreq

        if mi == 0:
            beam_shape = (nfreq, self.ntel/2, self.nsky)
            ibeam_shape = (nfreq, self.nsky, self.ntel/2)
            if self.noise_weight:
                noisew = self.telescope.noisepower(np.arange(self.telescope.npairs), 0).flatten()**(-0.5)
        else:
            beam_shape = (nfreq, self.ntel, self.nsky)
            ibeam_shape = (nfreq, self.nsky, self.ntel)
            if self.noise_weight:
                noisew = self.telescope.noisepower(np.arange(self.telescope.npairs), 0).flatten()**(-0.5)
                noisew = np.concatenate([noisew, noisew.conj()])

        beam = self.beam_m(mi)
        beam = beam.reshape(beam_shape)
        inv_beam = np.empty(ibeam_shape, dtype=beam.dtype)
        if self.noise_weight:
            beam *= noisew[:, np.newaxis]
        for fi in range(nfreq):
            # inv_beam[fi] = la.pinv2(beam[fi], rcond=1.0e-6)
            inv_beam[fi] = la.pinv2(beam[fi], rcond=1.0e-4)
        if self.noise_weight:
            inv_beam *= noisew

        return inv_beam


    def _generate_invbeam(self, regen=False):

        completed_file = self._inv_mdir + 'COMPLETED_IBEAM'
        if os.path.exists(completed_file) and not regen:
            if mpiutil.rank0:
                print
                print '=' * 80
                print "******* Inverse beam transfer m-files already generated ********"
            mpiutil.barrier()
            return

        if mpiutil.rank0:
            print
            print '=' * 80
            print 'Create inverse beam transfer m-files...'

        st = time.time()

        mmax = self.telescope.mmax
        for mi in mpiutil.mpilist(range(mmax+1)):
            inv_beam = self.compute_invbeam_m(mi)
            # save to file
            with h5py.File(self._inv_mfile(mi), 'w') as f:
                f.create_dataset('ibeam_m', data=inv_beam)
                f.attrs['m'] = mi

        mpiutil.barrier()

        et = time.time()

        if mpiutil.rank0:

            # Make file marker that the m's have been correctly generated:
            open(completed_file, 'a').close()

            # Print out timing
            print "=== Create inverse beam transfer m-files took %f s ===" % (et - st)


    #===================================================



    #====== Projection between spaces ==================

    def project_vector_sky_to_telescope(self, mi, vec):
        """Project a vector from the sky into the visibility basis.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        vec : np.ndarray
            Sky data vector packed as [nfreq, npol, ntheta]

        Returns
        -------
        tvec : np.ndarray
            Telescope vector to return packed as [nfreq, nbl].
        """

        nbl = self.telescope.nbase
        # vecf = np.zeros((nbl, self.nfreq), dtype=np.complex128)
        vecf = np.zeros((self.nfreq, nbl), dtype=np.complex128)

        beam = self.beam_m(abs(mi))

        if mi >= 0:
            beam = beam[:, :nbl]
        else:
            beam = beam[:, -nbl:].conj() # REMEMBER the conj here

        beam = beam.reshape((self.nfreq, nbl, self.nsky))

        for fi in range(self.nfreq):
            vecf[fi, :] = np.dot(beam[fi], vec[fi].reshape((self.nsky,)))

        return vecf

    project_vector_forward = project_vector_sky_to_telescope


    def project_vector_telescope_to_sky(self, mi, mmode):
        """Invert a vector from the telescope space onto the sky. This is the
        map-making process.

        Parameters
        ----------
        mi : integer
            Mode index to fetch for.
        mmode : np.ndarray
            Sky data vector.

        Returns
        -------
        Tm : np.ndarray
            Sky vector to return.
        """

        tel = self.telescope
        npol = tel.num_pol_sky
        ntheta = tel.theta_size

        inv_beam = self.invbeam_m(mi)

        Tm = np.zeros((self.nfreq, npol, ntheta), dtype=mmode.dtype)

        for fi in range(self.nfreq):
            Tm[fi] = np.dot(inv_beam[fi], mmode[fi, :]).reshape((npol, ntheta))

        return Tm


    project_vector_backward = project_vector_telescope_to_sky


    #===================================================

    #====== Dimensionality of the various spaces =======

    @property
    def ntel(self):
        """Degrees of freedom measured by the telescope (per frequency)"""
        return 2 * self.telescope.npairs

    @property
    def nsky(self):
        """Degrees of freedom on the sky at each frequency and `m`."""
        return self.telescope.num_pol_sky * self.telescope.theta_size

    @property
    def nfreq(self):
        """Number of frequencies measured."""
        return self.telescope.nfreq

    #===================================================
