#! /usr/bin/python
"""This module flags rfi and other forms of bad data.
"""
import copy

import numpy as np
import numpy.ma as ma
import scipy as sp
import scipy.signal as signal

#import hanning
from tlpipe.container.timestream import Timestream
import timestream_task

# XXX
#import matplotlib.pyplot as plt

class FlagData(timestream_task.TimestreamTask) :
    '''Pipeline module that flags RFI and other forms of bad data.

    '''

    params_init = {
                   # In multiples of the standard deviation of the whole block
                   # once normalized to the time median.
                   #'perform_hanning' : False,
                   #'cal_scale' : False,
                   #'cal_phase' : False,
                   # Rotate to XX,XY,YX,YY is True.
                   #'rotate' : False,
                   # Any frequency with variance > sigma_thres sigmas will be 
                   # flagged (recursively).
                   'sigma_thres' : 6.,
                   # A Data that has more than badness_thres frequencies flagged
                   # (as a fraction) will be considered bad.
                   'badness_thres' : 0.1,
                   # How many times to hide around a bad time.
                   'time_cut' : 5,
                   'time_block' : 108,
                   'bad_freq_list'  : None,
                   'bad_time_list'  : None,
                   }
    prefix = 'fd_'
    
    def process(self, ts):
        '''Prepares Data and flags RFI.

        '''
        bad_time_list = self.params['bad_time_list']
        bad_freq_list = self.params['bad_freq_list']

        if bad_time_list is not None:
            for bad_time in bad_time_list:
                print bad_time
                ts.vis_mask[slice(*bad_time), ...] = True


        if bad_freq_list is not None:
            print "Mask bad freq"
            for bad_freq in bad_freq_list:
                print bad_freq
                ts.vis_mask[:, slice(*bad_freq), ...] = True

        ts.redistribute('baseline')

        show_progress = self.params['show_progress']
        progress_step = self.params['progress_step']

        already_flagged = np.sum(ts.vis_mask)

        func = ts.bl_data_operate

        func(self.apply_cuts, full_data=False, show_progress=show_progress,
                progress_step=progress_step, keep_dist_axis=False)

        new_flags = np.sum(ts.vis_mask) - already_flagged
        percent = float(new_flags) / np.prod(ts.vis.shape) * 100
        #print '%d (%f%%), ' % (new_flags, percent)
        print '%d (%f%%), ' % (new_flags, percent)

        return super(FlagData, self).process(ts)

    def apply_cuts(self, vis, vis_mask, li, gi, tf, ts, **kwargs):
        #sigma_thres=6, badness_thres=0.1, time_cut=40):
        '''Flags bad data from RFI and far outliers.

        See `flag_data()` for parameter explanations and more info.
        '''
        sigma_thres   = self.params['sigma_thres']
        badness_thres = self.params['badness_thres']
        time_cut      = self.params['time_cut']
        time_block    = self.params['time_block']

        if 'ns_on' in ts.iterkeys():
            ns_on = ts['ns_on'][:, gi]
            #_vis = vis[~ns_on, ...]
            #_vis_mask = vis_mask[~ns_on, ...].copy()

        already_flagged = np.sum(vis_mask[~ns_on])
        if time_block  is None:
            time_block = vis.shape[0]
        for i in range(0, vis.shape[0], time_block):

            time_slice = slice(i, i+time_block)
            _ns_off = ~ns_on[time_slice]
            for j in range(10):
                _already_flagged = np.sum(vis_mask[time_slice][_ns_off])

                Data      = ma.array(vis[time_slice][_ns_off])
                Data.mask = vis_mask[time_slice][_ns_off]
                badness   = flag_data(Data, sigma_thres, badness_thres, time_cut)
                vis_mask[time_slice][_ns_off] += Data.mask

                _new_flags = np.sum(vis_mask[time_slice][_ns_off]) - _already_flagged
                percent = float(_new_flags) / np.prod(vis[time_slice][_ns_off].shape)
                if percent < 0.01 : break

        new_flags = np.sum(vis_mask[~ns_on]) - already_flagged
        percent = float(new_flags) / np.prod(vis[~ns_on].shape) * 100
        print "global index [bl] %2d: %f%%"%(gi, percent)

        # Can print or return badness here if you would like
        # to see if the Data had a problem in time or not.

def flag_data(Data, sigma_thres, badness_thres, time_cut):
    '''Flag bad data from RFI and far outliers.

    Parameters
    ----------
    Data : DataBlock
        Contains information in a usable format direct from GBT. Bad
        frequencies will be flagged in all polarizations and cal states.
    sigma_thres : int or float
        Any frequency with variance > `sigma_thres` sigmas will be 
        flagged (recursively).
    badness_thres : float
        A `Data` that has more than `badness_thres` frequencies flagged
        (as a fraction) will be considered 'bad'. `0` means that everything
        will be considered bad while `1` means nothing will be. 
    time_cut : int
        How many time bins (as an absolute number) to flag if `Data` has been
        considered 'bad'. See `destroy_time_with_mean_arrays` for more 
        infomation on this.

    Returns
    -------
    badness : bool
        Returns `True` iff a `Data` has been considered 'bad'.
    
    Notes
    -----
    'badness' is when more than a certain fraction of freqs has been flagged
    from `Data`. This certain fraction comes from `badness_thres`. `Data` that
    is 'bad' has a lot of frequencies flagged and this can because a lot of 
    frequencies are actually bad or because there was a blip in time (maybe
    the machine choked for a second).
    If a `Data` gets considered 'bad' then the algorithm tries to find
    something wrong in time (and masks those bad times) and redoes the RFI
    flagging. If there is a significant decrease (5%) in the number of 
    frequencies flagged, then the problem was in time and it uses the mask
    from this second run with bad times flagged. If not, then the `Data` is
    bad either way and it uses the mask from the first run. Increasing the
    `time_cut` in this situation is not recommended since you lose a lot more
    data (there are 10 times as many freq. bins as time bins in `Data`). 
    '''
    # Flag data on a [deep]copy of Data. If too much destroyed,
    # check if localized in time. If that sucks too, then just hide freq.

    Data1 = copy.deepcopy(Data)
    itr = 0            # For recursion
    max_itr = 20       # For recursion
    bad_freqs = []
    amount_masked = -1 # For recursion

    destroy_time_with_mean_arrays(Data1, 2.5, flag_size=8)

    while not (amount_masked == 0) and itr < max_itr:
        amount_masked = destroy_with_variance(Data1, sigma_thres, bad_freqs) 
        itr += 1
    bad_freqs.sort()
    # Remember the flagged data.
    mask = Data1.mask
    # Check for badness.
    percent_masked1 = (float(len(bad_freqs)) / Data1.shape[1])
    badness = (percent_masked1 > badness_thres)
    # If too many frequencies flagged, it may be that the problem
    # happens in time, not in frequency.
    if badness:
        Data2 = copy.deepcopy(Data)
        # Mask the bad times.
        destroy_time_with_mean_arrays(Data2, flag_size=time_cut)
        # Then try to flag again with bad times masked.
        # Bad style for repeating as above, sorry.
        itr = 0
        bad_freqs = []
        amount_masked = -1
        while not (amount_masked == 0) and itr < max_itr:
            amount_masked = destroy_with_variance(Data2, bad_freq_list=bad_freqs) 
            itr += 1
        bad_freqs.sort()
        percent_masked2 = (float(len(bad_freqs)) / Data2.shape[1])
        # If the data is 5% or more cleaner this way <=> it is not bad.
        badness = (percent_masked1 - percent_masked2) < 0.05
        # If this data does not have badness, that means there was
        # a problem in time and it was solved, so use this mask.
        # If the data is still bad, then the mask from Data1 will be used.
        if not badness:
            itr = 0
            while not (amount_masked == 0) and itr < max_itr:
                amount_masked = destroy_with_variance(Data2, sigma_thres,
                                                      bad_freqs) 
                itr += 1
            Data1 = Data2
    # We've flagged the RFI down to the foreground limit.  Filter out the
    # foregrounds and flag again to get below the foreground limit.
    # TODO, hard coded time_bins_smooth acctually depends on the scan speed and
    # the time sampling.
    filter_foregrounds(Data1, n_bands=40, time_bins_smooth=10)
    itr = 0 
    while not (amount_masked == 0) and itr < max_itr:
        amount_masked = destroy_with_variance(Data1, sigma_thres, bad_freqs) 
        itr += 1
    mask = Data1.mask
    # Finally copy the mask to origional data block.
    Data.mask = mask
    return badness

#def destroy_badtime(data, sigma_thres=6):

def destroy_with_variance(data, sigma_thres=6, bad_freq_list=[]):
    '''Mask frequencies with high variance.

    Since the signal we are looking for is much weaker than what is in `data`,
    any frequency that is 'too spiky' is not signal and is RFI instead. Using
    variance as a test really makes this 'spikyness' stand out.

    Parameters
    ----------
    data : DataBlock
        Contains information in a usable format direct from GBT. Bad
        frequencies will be flagged in all polarizations and cal states.
    sigma_thres : int or float
        Any frequency with variance > `sigma_thres` sigmas will be 
        flagged (recursively).
    bad_freq_list : list of int
        A list of bad frequencies. Since this method is called over and over,
        this list keeps track of what has been flagged. Bad frequencies that
        are found will be appended to this list.

    Returns
    -------
    amount_masked : int
        The amount of frequencies masked.

    Notes
    -----
    Polarizations must be in XX,XY,YX,YY format.

    '''
    XX_YY_0 = ma.mean(data[:, :,  0], 0) * ma.mean(data[:, :, 1], 0)
    # Get the normalized variance array for each polarization.
    a = ma.var(data[:, :, 0,], 0) / (ma.mean(data[:, :, 0], 0)**2) # XX
    b = ma.var(data[:, :, 1,], 0) / (ma.mean(data[:, :, 1], 0)**2) # YY
    # Get the mean and standard deviation [sigma].
    means = sp.array([ma.mean(a), ma.mean(b)]) 
    sig   = sp.array([ma.std(a), ma.std(b)])
    # Get the max accepted value [sigma_thres*sigma, sigma_thres=6 works really well].
    max_sig = sigma_thres*sig
    max_accepted = means + max_sig
    min_accepted = means - max_sig
    amount_masked = 0
    for freq in range(0, len(a)):
        if ((a[freq] > max_accepted[0]) or 
            (b[freq] > max_accepted[1]) or
            (a[freq] < min_accepted[0]) or 
            (b[freq] < min_accepted[1])):
            amount_masked += 1
            bad_freq_list.append(freq)
            data[:,freq,:].mask = True
    return amount_masked

def destroy_time_with_mean_arrays(data, sigma_thres = 3., flag_size=5):
    '''Mask times with high means.
    
    If there is a problem in time, the mean over all frequencies
    will stand out greatly [>10 sigma has been seen]. Flag these bad
    times and +- `flag_size` times around it. Will only be called if `Data`
    has 'badness'.

    Parameters
    ----------
    Data : DataBlock
        Contains information in a usable format direct from GBT. Bad
        times will be flagged in all polarizations and cal states.
    time_cut : int
        How many frequency bins (as an absolute number) to flag in time.
    '''
    # Get the means over all frequencies. (for all pols. and cals.)
    a = ma.mean(data[:, :, 0], -1)
    b = ma.mean(data[:, :, 1], -1)
    # Get means and std for all arrays.
    means = sp.array([ma.mean(a), ma.mean(b)])
    sig   = sp.array([ma.std(a),  ma.std(b)])
    # Get max accepted values.
    max_accepted = means + sigma_thres * sig
    min_accepted = means - sigma_thres * sig
    # Find bad times.
    bad_times = []
    for time in range(0,len(a)):
        if ((a[time] > max_accepted[0]) or
            (b[time] > max_accepted[1]) or
            (a[time] < min_accepted[0]) or
            (b[time] < min_accepted[1])):
            bad_times.append(time)
    # Mask bad times and those +- flag_size around.
    for time in bad_times:
        data[(time-flag_size):(time+flag_size),...].mask = True
    return

def filter_foregrounds(data, n_bands=20, time_bins_smooth=10.):
    """Gets an estimate of the foregrounds and subtracts it out of the data.
    
    The Foreground removal is very rough, just used to push the foreground down
    a bunch so the RFI can be more easily found.
    Two things are done to estimate the foregrounds: averaging over a fairly
    wide band, and smoothing to just below the beam crossing time scale.

    Parameters
    ----------
    Data : DataBolock object
        Data from which to remove the foregrounds.
    n_bands : int
        Number of bands to split the data into.  Forgrounds are assumed to
        be the same throughout this band.
    time_bins : float
        Number of time bins to smooth over to find the foregrounds (full width
        half max of the filter kernal). Should be
        shorter than the beam crossing time (by about a factor of 2).
    """
    
    # Some basic numbers.
    n_chan = data.shape[1]
    sub_band_width = float(n_chan)/n_bands
    # First up, initialize the smoothing kernal.
    width = time_bins_smooth/2.355
    # Two sigma edge cut off.
    nk = int(round(4*width)) + 1
    smoothing_kernal = signal.gaussian(nk, width)
    smoothing_kernal /= sp.sum(smoothing_kernal)
    smoothing_kernal.shape = (nk, 1)
    # Now loop through the sub-bands. Foregrounds are assumed to be identical
    # within a sub-band.
    for subband_ii in range(n_bands):
        # Figure out what data is in this subband.
        band_start = int(round(subband_ii * sub_band_width))
        band_end = int(round((subband_ii + 1) * sub_band_width))
        _data = data[:,band_start:band_end,:]
        # Estimate the forgrounds.
        # Take the band mean.
        foregrounds = ma.mean(_data, 1)
        # Now low pass filter.
        fore_weights = (sp.ones(foregrounds.shape, dtype=float)
                        - ma.getmaskarray(foregrounds))
        foregrounds -= ma.mean(foregrounds, 0)
        foregrounds = foregrounds.filled(0)
        foregrounds = signal.convolve(foregrounds, smoothing_kernal, mode='same')
        fore_weights = signal.convolve(fore_weights, smoothing_kernal, mode='same')
        foregrounds /= fore_weights
        # Subtract out the foregrounds.
        data[:,band_start:band_end,:] -= foregrounds[:,None,:]
        del _data


# If this file is run from the command line, execute the main function.
if __name__ == "__main__":
    import sys
    FlagData(str(sys.argv[1])).execute()

