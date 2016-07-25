try:
    import cPickle as pickle
except ImportError:
    import pickle

import os
import itertools

import h5py
import numpy as np

from cora.util import hputil

from caput import mpiutil
from caput.mpiarray import MPIArray
from caput import memh5

# from fmmode.core import manager
from tlpipe.map.fmmode.util import util
from tlpipe.map.fmmode.util import hpproj


class Timestream(object):

    # directory = None
    # output_directory = None
    # beamtransfer_dir = None

    # no_m_zero = True



    #============ Constructor etc. =====================

    def __init__(self, tsdir, tsname, beamtransfer):
        """Create a new Timestream object.

        Parameters
        ----------
        tsdir : string
            Directory to create the Timestream in.
        prodmanager : drift.core.manager.ProductManager
            ProductManager object containing the analysis products.
        """
        self.directory = os.path.abspath(tsdir)
        self.output_directory = '%s/%s' % (self.directory, tsname)
        self.tsname = tsname
        self.beamtransfer = beamtransfer

    #====================================================


    #===== Accessing the BeamTransfer and Telescope =====

    # @property
    # def beamtransfer(self):
    #     """The BeamTransfer object corresponding to this timestream.
    #     """

    #     return self.manager.beamtransfer

    @property
    def telescope(self):
        """The telescope object corresponding to this timestream.
        """
        return self.beamtransfer.telescope

    #====================================================


    @property
    def _tsdir(self):
        return self.output_directory + '/timestream'

    @property
    def _tsfile(self):
        return self._tsdir + '/timestream.hdf5'


    @property
    def ntime(self):
        """Get the number of timesamples."""

        with h5py.File(self._tsfile, 'r') as f:
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
        timestream : np.ndarray[ntime, npairs]
            The visibility timestream.
        """

        with h5py.File(self._tsfile, 'r') as f:
            ts = f['timestream'][..., fi, :]
        return ts

    #====================================================


    #======== Fetch and generate the m-modes ============

    @property
    def _mdir(self):
        return self.output_directory + '/mmodes/'


    def _mfile(self, mi):
        # Pattern to form the `m` ordered file.
        pat = self._mdir + 'mode_%s.hdf5' % util.intpattern(self.telescope.mmax)
        return pat % mi


    def mmode(self, mi, fi=None):
        """Fetch the timestream m-mode for a specified m.

        Parameters
        ----------
        mi : integer
            m-mode to load.

        Returns
        -------
        timestream : np.ndarray[nfreq, ntel] or [ntel]
            The visibility m-modes.
        """

        if mi == 0:
            with h5py.File(self._mfile(0), 'r') as f:
                if fi is None:
                    return f['mmode'][:]
                else:
                    return f['mmode'][fi, :]
        elif mi > 0:
            with h5py.File(self._mfile(mi), 'r') as f1, h5py.File(self._mfile(-mi), 'r') as f2:
                if fi is None:
                    ### REMEMBER the conj for negative m
                    return np.concatenate([f1['mmode'][:], f2['mmode'][:].conj()], axis=1)
                else:
                    return np.concatenate([f1['mmode'][fi, :], f2['mmode'][fi, :].conj()])
        else:
            raise ValueError('mi must greater than or equal to 0')



    def generate_mmodes(self, ts_data=None):
        """Calculate the m-modes corresponding to the Timestream.

        Perform an MPI transpose for efficiency.
        """

        completed_file = self._mdir + 'COMPLETED_M'
        if os.path.exists(completed_file):
            if mpiutil.rank0:
                print "******* m-files already generated ********"
            mpiutil.barrier()
            return

        # Make directory if required
        # if mpiutil.rank0 and not os.path.exists(self._mdir):
        #     os.makedirs(self._mdir)

        try:
            os.makedirs(self._mdir)
        except OSError:
            # directory exists
            pass

        tel = self.telescope
        mmax = tel.mmax
        ntime = ts_data.shape[0] if ts_data is not None else self.ntime
        nbl = tel.nbase
        nfreq = tel.nfreq

        indices = list(itertools.product(np.arange(nfreq), np.arange(nbl)))
        lind, sind, eind = mpiutil.split_local(nfreq * nbl)

        # load the local section of the time stream
        tstream = np.zeros((ntime, lind), dtype=np.complex128)
        for ind, (f_ind, bl_ind) in enumerate(indices[sind:eind]):
            if ts_data is not None:
                tstream[:, ind] = ts_data[:, f_ind, bl_ind]
            else:
                with h5py.File(self._tsfile, 'r') as f:
                    tstream[:, ind] = f['/timestream'][:, f_ind, bl_ind]

        # FFT to get m-mode
        mmodes = np.fft.fft(tstream, axis=0) / ntime # m = 0 is at left
        mmodes = MPIArray.wrap(mmodes, axis=1)
        # redistribute along different m
        mmodes = mmodes.redistribute(axis=0)

        # save m-modes to file
        ms = np.concatenate([np.arange(0, mmax+1), np.arange(-mmax, 0)])
        for ind, mi in enumerate(mpiutil.mpilist(ms, method='con')):
            with h5py.File(self._mfile(mi), 'w') as f:
              f.create_dataset('/mmode', data=mmodes[ind].view(np.ndarray).reshape(nfreq, nbl))
              f.attrs['m'] = mi

        mpiutil.barrier()

        if mpiutil.rank0:

            # Make file marker that the m's have been correctly generated:
            open(completed_file, 'a').close()

        # mpiutil.barrier()

    #====================================================


    @property
    def _Tmsdir(self):
        return self.output_directory + '/Tms/'

    @property
    def _mapsdir(self):
        return self.output_directory + '/maps/'

    #======== Make map from uncleaned stream ============

    def mapmake_full(self, nside, maptype):

        mapfile = self._mapsdir + 'map_%s.hdf5' % maptype
        Tmfile = self._Tmsdir + 'Tm_%s.hdf5' % maptype

        if os.path.exists(mapfile):
            if mpiutil.rank0:
                print "File %s exists. Skipping..." % mapfile
            mpiutil.barrier()
            return
        elif os.path.exists(Tmfile):
            if mpiutil.rank0:
                print "File %s exists. Read from it..." % Tmfile

            Tm = MPIArray.from_hdf5(Tmfile, 'Tm')
        else:

            def _make_Tm(mi):

                print "Making %i" % mi

                mmode = self.mmode(mi)

                return self.beamtransfer.project_vector_telescope_to_sky(mi, mmode)


            # if mpiutil.rank0 and not os.path.exists(self._Tmsdir):
            #     # Make directory for Tms file
            #     os.makedirs(self._Tmsdir)

            # Make directory for Tms file
            try:
                os.makedirs(self._Tmsdir)
            except OSError:
                # directory exists
                pass

            tel = self.telescope
            mmax = tel.mmax
            lm, sm, em = mpiutil.split_local(mmax+1)

            nfreq = tel.nfreq
            npol = tel.num_pol_sky
            ntheta = tel.theta_size
            # the local Tm array
            Tm = np.zeros((nfreq, npol, ntheta, lm), dtype=np.complex128)
            for ind, mi in enumerate(range(sm, em)):
                Tm[..., ind] = _make_Tm(mi)
            Tm = MPIArray.wrap(Tm, axis=3)
            Tm = Tm.redistribute(axis=0) # redistribute along freq

            # Save Tm
            Tm.to_hdf5(Tmfile, 'Tm', create=True)


        # if mpiutil.rank0 and not os.path.exists(self._mapsdir):
        #     # Make directory for maps file
        #     os.makedirs(self._mapsdir)

        # Make directory for maps file
        try:
            os.makedirs(self._mapsdir)
        except OSError:
            # directory exists
            pass

        tel = self.telescope
        npol = tel.num_pol_sky
        ntime = self.ntime

        # irfft to get map
        # cart_map = np.fft.irfft(Tm, axis=3, n=ntime) * ntime # NOTE the normalization constant ntime here to be consistant with the simulation fft
        cart_map = np.fft.hfft(Tm, axis=3, n=ntime)
        lfreq = cart_map.shape[0]
        hp_map = np.zeros((lfreq, npol, 12*nside**2), dtype=cart_map.dtype)
        for fi in range(lfreq):
            for pi in range(npol):
                hp_map[fi, pi] = tel.cart_projector.inv_projmap(cart_map[fi, pi], nside)

        mpiutil.barrier()
        hp_map = MPIArray.wrap(hp_map, axis=0)

        # save map
        hp_map.to_hdf5(mapfile, 'map', create=True)


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
        print
        print '=' * 80
        print "Saving Timestream object %s..." % self.tsname
        with open(self._picklefile, 'w') as f:
            pickle.dump(self, f)


    @classmethod
    def load(cls, tsdir):
        """Load the Timestream object from disk.

        Parameters
        ----------
        tsdir : string
            Name of the directory containing the Timestream object.
        """

        # Create temporary object to extract picklefile property
        tmp_obj = cls(tsdir, tsdir)

        with open(tmp_obj._picklefile, 'r') as f:
            print "=== Loading Timestream object. ==="
            return pickle.load(f)

    #====================================================



# kwargs is to absorb any extra params
def simulate(m, outdir, tsname, maps=[], ndays=None, resolution=0, add_noise=True, seed=None, **kwargs):
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
    add_noise : bool, optional
        Weather to add random noise to the simulated visibilities. Default True.

    Returns
    -------
    timestream : Timestream
    """

    # Create timestream object
    tstream = Timestream(outdir, tsname, m)

    completed_file = tstream._tsdir + '/COMPLETED_TIMESTREAM'
    if os.path.exists(completed_file):
        if mpiutil.rank0:
            print "******* timestream-files already generated ********"
        mpiutil.barrier()
        return tstream

    # Make directory if required
    try:
        os.makedirs(tstream._tsdir)
    except OSError:
         # directory exists
         pass

    if mpiutil.rank0:
        # if not os.path.exists(tstream._tsdir):
        #     os.makedirs(tstream._tsdir)

        tstream.save()

    ## Read in telescope system
    bt = m.beamtransfer
    tel = bt.telescope

    lmax = tel.lmax
    mmax = tel.mmax
    nfreq = tel.nfreq
    nbl = tel.nbase
    npol = tel.num_pol_sky

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

    indices = list(itertools.product(np.arange(nfreq), np.arange(npol)))
    lind, sind, eind = mpiutil.split_local(nfreq * npol)

    # local section of the Tm array
    theta_size = tel.theta_size
    phi_size = tel.phi_size
    Tm = np.zeros((lind, theta_size, phi_size), dtype=np.complex128)

    for ind, (f_ind, p_ind) in enumerate(indices[sind:eind]):
        hp_map = None
        for idx, mapfile in enumerate(maps):
            with h5py.File(mapfile, 'r') as f:
                if idx == 0:
                    hp_map = f['map'][f_ind, p_ind, :]
                else:
                    hp_map += f['map'][f_ind, p_ind, :]
        if hp_map is not None:
            cart_map = hpproj.cartesian_proj(hp_map, tel.cart_projector)
            # Calculate the Tm's for the local sections
            Tm[ind] = np.fft.ifft(cart_map, axis=1) # / phi_size # m = 0 is at left

    Tm = MPIArray.wrap(Tm, axis=0)
    # redistribute along different m
    Tm = Tm.redistribute(axis=2)
    Tm = Tm.reshape((nfreq, npol, theta_size, None))
    Tm = Tm.reshape((nfreq, npol*theta_size, None))

    ms = np.concatenate([np.arange(0, mmax+1), np.arange(-mmax, 0)])
    lm, sm, em = mpiutil.split_local(phi_size)
    # local section of mmode
    # mmode = np.zeros((lm, nbl, nfreq), dtype=np.complex128)
    mmode = np.zeros((lm, nfreq, nbl), dtype=np.complex128)

    for ind, mi in enumerate(ms[sm:em]):
        mmode[ind] = bt.project_vector_sky_to_telescope(mi, Tm[:, :, ind].view(np.ndarray))

    mmode = MPIArray.wrap(mmode, axis=0)
    mmode = mmode.redistribute(axis=2) # distribute along bl

    # add noise if required
    if add_noise:
        lbl, sbl, ebl = mpiutil.split_local(nbl)
        # Fetch the noise powerspectrum
        noise_ps = tel.noisepower(np.arange(sbl, ebl)[:, np.newaxis], np.arange(nfreq)[np.newaxis, :], ndays=ndays).reshape(lbl, nfreq).T[np.newaxis, :, :]

        # Seed random number generator to give consistent noise
        if seed is not None:
        # Must include rank such that we don't have massive power deficit from correlated noise
            np.random.seed(seed + mpiutil.rank)

        # Create and weight complex noise coefficients
        noise_mode = (np.array([1.0, 1.0J]) * np.random.standard_normal(mmode.shape + (2,))).sum(axis=-1)
        noise_mode *= (noise_ps / 2.0)**0.5

        mmode += noise_mode

        del noise_mode

        # Reset RNG
        if seed is not None:
            np.random.seed()

    # The time samples the visibility is calculated at
    tphi = np.linspace(0, 2*np.pi, ntime, endpoint=False)

    # inverse FFT to get timestream
    vis_stream = np.fft.ifft(mmode, axis=0) * ntime
    vis_stream = MPIArray.wrap(vis_stream, axis=2)

    # save vis_stream to file
    vis_h5 = memh5.MemGroup(distributed=True)
    vis_h5.create_dataset('/timestream', data=vis_stream)
    vis_h5.create_dataset('/phi', data=tphi)

    # Telescope layout data
    vis_h5.create_dataset('/feedmap', data=tel.feedmap)
    vis_h5.create_dataset('/feedconj', data=tel.feedconj)
    vis_h5.create_dataset('/feedmask', data=tel.feedmask)
    vis_h5.create_dataset('/uniquepairs', data=tel.uniquepairs)
    vis_h5.create_dataset('/baselines', data=tel.baselines)

    # Telescope frequencies
    vis_h5.create_dataset('/frequencies', data=tel.frequencies)

    # Write metadata
    vis_h5.attrs['beamtransfer_path'] = os.path.abspath(bt.directory)
    vis_h5.attrs['ntime'] = ntime

    # save to file
    vis_h5.to_hdf5(tstream._tsfile)

    if mpiutil.rank0:
        # Make file marker that the m's have been correctly generated:
        open(completed_file, 'a').close()

    mpiutil.barrier()

    return tstream
