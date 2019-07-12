"""Calibrate the visibility by divide the gain.

Inheritance diagram
-------------------

.. inheritance-diagram:: Apply
   :parts: 2

"""

import numpy as np
import h5py
import timestream_task
from tlpipe.container.timestream import Timestream
from tlpipe.utils.path_util import input_path

import logging

logger = logging.getLogger(__name__)


class Apply(timestream_task.TimestreamTask):
    """Calibrate the visibility by divide the gain.

    .. math:: V_{ij}^{\\text{cal}} = V_{ij} / (g_i g_j^*).

    """


    params_init = {
                    'gain_file': 'gain.hdf5',
                  }

    prefix = 'ag_'

    def process(self, ts):

        assert isinstance(ts, Timestream), '%s only works for Timestream object' % self.__class__.__name__
        
        #Tolerance for frequency matching between gain table and visibilities
        FTOL = 0.001
        #Mask bit for invalid calibration
        MASKNOCAL = 4
 
        ts.redistribute('baseline')
       
        # read gain from file
        gain_file = self.params['gain_file']
        tag_input_iter = self.params['tag_input_iter']
        if tag_input_iter:
            gain_file = input_path(gain_file, self.iteration)
        with h5py.File(gain_file, 'r') as f:
            #cal_algorith = f.attrs['cal_algorith']
            #calibrator
            #cal_time
            #cal_unit
            #cal_date
            gain = f['gain'][:]
            gain_src = f['gain'].attrs['calibrator']
            gain_freq = f['gain'].attrs['freq']
            gain_pol = f['gain'].attrs['pol']
            gain_feed = f['gain'].attrs['feed']
        #Number of feeds in gain table
        ngfeed = gain_feed.shape[0]
        #Number of frequencies in gain table
        ngfreq = gain_freq.shape[0]
        #Number of polarizations in gain table 
        ngpol = gain_pol.shape[0]
        #The gain table should generally have two polarizations: x and y
        #The following code finds an index to x (pindex[0]) and y (pindex[1])
        pindex = np.full(2,-1,dtype=np.int)
        for n in range(ngpol):
            elt = gain_pol[n]
            if elt[0]=='x':
                pindex[0] = n
            if elt[0]=='y':
                pindex[1] = n

        #pol is the index of polarizations in the visibility table
        pol = ts.pol[:]
        #dictionary gives polarization name as a function of index
        pol_dict = [ ts.pol_dict[p] for p in ts['pol'][:] ] # as string
        freq = ts.freq[:]
    
        nfreq = len(freq)
        npol = len(pol)

        #Build a index for visibility frequencies in gain table
        findex = np.full(nfreq,-1,dtype=np.int)
        for nf in range(nfreq):
            for ngf in range(ngfreq):
                if np.abs(freq[nf]-gain_freq[ngf])<FTOL:
                    findex[nf] = ngf


        #Loop over baselines.  fd1 and fd2 are the two feed numbers
        for bi, (fd1, fd2) in enumerate(ts['blorder'].local_data):
            #gfd1 and gfd2 are the indices of fd1 and fd2 in the gain table.
            #Initialize gfd1 and gfd2 to not found
            gfd1 = -1
            gfd2 = -1
            for ng in range(ngfeed):
                if fd1==gain_feed[ng]:
                    gfd1 = ng
                if fd2==gain_feed[ng]:
                    gfd2 = ng

            #Loop over polarizations in the visibility array
            for pi in range(npol):
                #p1 and p2 are the indices of the polarization in the gain file.
                #The code will work for polarizations that are
                #labeled xx, yy, xy, and yx only.  Other possibilities 
                #(I,Q,U,V, e.g.) are not programmed.

                #Initialize to p1 and p2 not found
                p1 = -1
                p2 = -1
                ptype = pol[pi]
                if pol_dict[ptype]=='xx':
                    p1 = pindex[0]
                    p2 = pindex[0]
                if pol_dict[ptype]=='yy':
                    p1 = pindex[1]
                    p2 = pindex[1]
                if pol_dict[ptype]=='xy':
                    p1 = pindex[0]
                    p2 = pindex[1]
                if pol_dict[ptype]=='yx':
                    p1 = pindex[1]
                    p2 = pindex[0]
                #See if there is a valid gain for this baseline/polarization
                if gfd1>=0 and gfd2>=0 and p1>=0 and p2>=0:
                    for nf in range(nfreq):
                        nfg = findex[nf]
                        g1 = gain[nfg, p1, gfd1]
                        g2 = gain[nfg, p2, gfd2]
                        if np.isfinite(g1) and np.isfinite(g2) and nfg>=0:
                            ts.local_vis[:, nf, pi, bi] /= (g1 * np.conj(g2))
                        else:
                            # mask the un-calibrated vis
                            logger.info("Masking freq=%i baseline=%i polarization=%i" % (fi,bi,pi))
                            ts.local_vis[:,fi,pi,bi] = np.nan
                            ts.local_vis_mask[:, fi, pi, bi] |= MASKNOCAL
                else:
                    #mask all times & frequencies for this baseline/polarization
                    logger.info("Masking baseline=%i polarization=%i" % (bi,pi))
                    ts.local_vis[:,:,pi,bi] = np.nan
                    ts.local_vis_mask[:,:, pi, bi] |= MASKNOCAL
#ADD INFO
#CALIBRATION K or J
#FMIN, FMAX
# self.attrs['xxx'] = data
#CALIBRATION DATE? FILENAME?

        return super(Apply, self).process(ts)
