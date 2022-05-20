from numpy import (asarray, savetxt, std, max, min, genfromtxt, load, nan as npnan, arange as nparange, array as nparray, isfinite as npisfinite,
    argmax as npargmax, digitize as npdigitize, median as npmedian, asarray,
    std as npstd, argsort as npargsort, unique as npunique, sum as npsum, isfinite as npisfinite, median as npmedian, mean as npmean,abs as npabs, std as npstddev, pi as pi_value, cos as npcos, sin as npsin, vdot as npvdot, max as npmax, savetxt, std, max, min, genfromtxt, load, nonzero as npnonzero,arctan as nparctan)
import numpy as np
import sys
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import logging

from astrosource.utils import photometry_files_to_array, AstrosourceException
from astropy.timeseries import LombScargle

logger = logging.getLogger('astrosource')
NCPUS = 1

# Note that the functions that calculate the ANOVA periodograms have been adapted from the astrobase codeset
# These are aov_theta, resort_by_time, get_frequency_grid, sigclip_magseries, phase_magseries, aov_periodfind, phase_magseries_with_errs, aovhm_theta, aovhm_periodfind
# The astrobase code is available here: https://github.com/waqasbhatti/astrobase
#
# The astrobase code was adapted rather than imported to remove dependencies unnecessary for astrosource.
#
# The astrobase codeset is distributed under an MIT license. For those functions, the following notice is applicable
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

def aov_theta(times, mags, errs, frequency,
              binsize=0.05, minbin=9):
    '''Calculates the Schwarzenberg-Czerny AoV statistic at a test frequency.

    Parameters
    ----------

    times,mags,errs : np.array
        The input time-series and associated errors.

    frequency : float
        The test frequency to calculate the theta statistic at.

    binsize : float
        The phase bin size to use.

    minbin : int
        The minimum number of items in a phase bin to consider in the
        calculation of the statistic.

    Returns
    -------

    theta_aov : float
        The value of the AoV statistic at the specified `frequency`.

    '''

    period = 1.0/frequency
    fold_time = times[0]

    phased = phase_magseries(times,
                             mags,
                             period,
                             fold_time,
                             wrap=False,
                             sort=True)

    phases = phased['phase']
    pmags = phased['mags']
    bins = nparange(0.0, 1.0, binsize)
    ndets = phases.size

    binnedphaseinds = npdigitize(phases, bins)

    bin_s1_tops = []
    bin_s2_tops = []
    binndets = []
    goodbins = 0

    all_xbar = npmedian(pmags)

    for x in npunique(binnedphaseinds):

        thisbin_inds = binnedphaseinds == x
        thisbin_mags = pmags[thisbin_inds]

        if thisbin_mags.size > minbin:

            thisbin_ndet = thisbin_mags.size
            thisbin_xbar = npmedian(thisbin_mags)

            # get s1
            thisbin_s1_top = (
                thisbin_ndet *
                (thisbin_xbar - all_xbar) *
                (thisbin_xbar - all_xbar)
            )

            # get s2
            thisbin_s2_top = npsum((thisbin_mags - all_xbar) *
                                   (thisbin_mags - all_xbar))

            bin_s1_tops.append(thisbin_s1_top)
            bin_s2_tops.append(thisbin_s2_top)
            binndets.append(thisbin_ndet)
            goodbins = goodbins + 1

    # turn the quantities into arrays
    bin_s1_tops = nparray(bin_s1_tops)
    bin_s2_tops = nparray(bin_s2_tops)
    binndets = nparray(binndets)

    # calculate s1 first
    s1 = npsum(bin_s1_tops)/(goodbins - 1.0)

    # then calculate s2
    s2 = npsum(bin_s2_tops)/(ndets - goodbins)

    theta_aov = s1/s2

    return theta_aov

def resort_by_time(times, mags, errs):
    '''
    Resorts the input arrays so they're in time order.
    NOTE: the input arrays must not have nans in them.
    Parameters
    ----------
    times,mags,errs : np.arrays
        The times, mags, and errs arrays to resort by time. The times array is
        assumed to be the first one in the input args.
    Returns
    -------
    times,mags,errs : np.arrays
        The resorted times, mags, errs arrays.
    '''

    sort_order = np.argsort(times)
    times, mags, errs = times[sort_order], mags[sort_order], errs[sort_order]

    return times, mags, errs

def get_frequency_grid(times,
                       samplesperpeak=5,
                       nyquistfactor=5,
                       minfreq=None,
                       maxfreq=None,
                       returnf0dfnf=False):
    '''This calculates a frequency grid for the period finding functions in this
    module.
    Based on the autofrequency function in astropy.stats.lombscargle.
    http://docs.astropy.org/en/stable/_modules/astropy/stats/lombscargle/core.html#LombScargle.autofrequency
    Parameters
    ----------
    times : np.array
        The times to use to generate the frequency grid over.
    samplesperpeak : int
        The minimum sample coverage each frequency point in the grid will get.
    nyquistfactor : int
        The multiplier over the Nyquist rate to use.
    minfreq,maxfreq : float or None
        If not None, these will be the limits of the frequency grid generated.
    returnf0dfnf : bool
        If this is True, will return the values of `f0`, `df`, and `Nf`
        generated for this grid.
    Returns
    -------
    np.array
        A grid of frequencies.
    '''

    baseline = times.max() - times.min()
    nsamples = times.size

    df = 1. / baseline / samplesperpeak

    if minfreq is not None:
        f0 = minfreq
    else:
        f0 = 0.5 * df

    if maxfreq is not None:
        Nf = int(np.ceil((maxfreq - f0) / df))
    else:
        Nf = int(0.5 * samplesperpeak * nyquistfactor * nsamples)

    if returnf0dfnf:
        return f0, df, Nf, f0 + df * np.arange(Nf)
    else:
        return f0 + df * np.arange(Nf)

def sigclip_magseries(times, mags, errs,
                      sigclip=None,
                      iterative=False,
                      niterations=None,
                      meanormedian='median',
                      magsarefluxes=False):
    '''Sigma-clips a magnitude or flux time-series.
    Selects the finite times, magnitudes (or fluxes), and errors from the passed
    values, and apply symmetric or asymmetric sigma clipping to them.
    Parameters
    ----------
    times,mags,errs : np.array
        The magnitude or flux time-series arrays to sigma-clip. This doesn't
        assume all values are finite or if they're positive/negative. All of
        these arrays will have their non-finite elements removed, and then will
        be sigma-clipped based on the arguments to this function.
        `errs` is optional. Set it to None if you don't have values for these. A
        'faked' `errs` array will be generated if necessary, which can be
        ignored in the output as well.
    sigclip : float or int or sequence of two floats/ints or None
        If a single float or int, a symmetric sigma-clip will be performed using
        the number provided as the sigma-multiplier to cut out from the input
        time-series.
        If a list of two ints/floats is provided, the function will perform an
        'asymmetric' sigma-clip. The first element in this list is the sigma
        value to use for fainter flux/mag values; the second element in this
        list is the sigma value to use for brighter flux/mag values. For
        example, `sigclip=[10., 3.]`, will sigclip out greater than 10-sigma
        dimmings and greater than 3-sigma brightenings. Here the meaning of
        "dimming" and "brightening" is set by *physics* (not the magnitude
        system), which is why the `magsarefluxes` kwarg must be correctly set.
        If `sigclip` is None, no sigma-clipping will be performed, and the
        time-series (with non-finite elems removed) will be passed through to
        the output.
    iterative : bool
        If this is set to True, will perform iterative sigma-clipping. If
        `niterations` is not set and this is True, sigma-clipping is iterated
        until no more points are removed.
    niterations : int
        The maximum number of iterations to perform for sigma-clipping. If None,
        the `iterative` arg takes precedence, and `iterative=True` will
        sigma-clip until no more points are removed.  If `niterations` is not
        None and `iterative` is False, `niterations` takes precedence and
        iteration will occur for the specified number of iterations.
    meanormedian : {'mean', 'median'}
        Use 'mean' for sigma-clipping based on the mean value, or 'median' for
        sigma-clipping based on the median value.  Default is 'median'.
    magsareflux : bool
        True if your "mags" are in fact fluxes, i.e. if "fainter" corresponds to
        `mags` getting smaller.
    Returns
    -------
    (stimes, smags, serrs) : tuple
        The sigma-clipped and nan-stripped time-series.
    '''

    returnerrs = True

    # fake the errors if they don't exist
    # this is inconsequential to sigma-clipping
    # we don't return these dummy values if the input errs are None
    #print (errs)
    if errs is None:
        # assume 0.1% errors if not given
        # this should work for mags and fluxes
        errs = 0.001*mags
        returnerrs = False

    # filter the input times, mags, errs; do sigclipping and normalization
    find = npisfinite(times) & npisfinite(mags) & npisfinite(errs)
    ftimes, fmags, ferrs = times[find], mags[find], errs[find]

    # get the center value and stdev
    if meanormedian == 'median':  # stddev = 1.483 x MAD

        center_mag = npmedian(fmags)
        stddev_mag = (npmedian(npabs(fmags - center_mag))) * 1.483

    elif meanormedian == 'mean':

        center_mag = npmean(fmags)
        stddev_mag = npstddev(fmags)

    else:
        logger.info("unrecognized meanormedian value given to "
                   "sigclip_magseries: %s, defaulting to 'median'" %
                   meanormedian)
        meanormedian = 'median'
        center_mag = npmedian(fmags)
        stddev_mag = (npmedian(npabs(fmags - center_mag))) * 1.483

    # sigclip next for a single sigclip value
    if sigclip and isinstance(sigclip, (float, int)):

        if not iterative and niterations is None:

            sigind = (npabs(fmags - center_mag)) < (sigclip * stddev_mag)

            stimes = ftimes[sigind]
            smags = fmags[sigind]
            serrs = ferrs[sigind]

        else:

            #
            # iterative version adapted from scipy.stats.sigmaclip
            #

            # First, if niterations is not set, iterate until covergence
            if niterations is None:

                delta = 1

                this_times = ftimes
                this_mags = fmags
                this_errs = ferrs

                while delta:

                    if meanormedian == 'mean':
                        this_center = npmean(this_mags)
                        this_stdev = npstddev(this_mags)
                    elif meanormedian == 'median':
                        this_center = npmedian(this_mags)
                        this_stdev = (
                            npmedian(npabs(this_mags - this_center))
                        ) * 1.483
                    this_size = this_mags.size

                    # apply the sigclip
                    tsi = (
                        (npabs(this_mags - this_center)) <
                        (sigclip * this_stdev)
                    )

                    # update the arrays
                    this_times = this_times[tsi]
                    this_mags = this_mags[tsi]
                    this_errs = this_errs[tsi]

                    # update delta and go to the top of the loop
                    delta = this_size - this_mags.size

            else:  # If iterating only a certain number of times

                this_times = ftimes
                this_mags = fmags
                this_errs = ferrs

                iter_num = 0
                delta = 1
                while iter_num < niterations and delta:

                    if meanormedian == 'mean':

                        this_center = npmean(this_mags)
                        this_stdev = npstddev(this_mags)

                    elif meanormedian == 'median':

                        this_center = npmedian(this_mags)
                        this_stdev = (npmedian(npabs(this_mags -
                                                     this_center))) * 1.483
                    this_size = this_mags.size

                    # apply the sigclip
                    tsi = (
                        (npabs(this_mags - this_center)) <
                        (sigclip * this_stdev)
                    )

                    # update the arrays
                    this_times = this_times[tsi]
                    this_mags = this_mags[tsi]
                    this_errs = this_errs[tsi]

                    # update the number of iterations and delta and
                    # go to the top of the loop
                    delta = this_size - this_mags.size
                    iter_num += 1

            # final sigclipped versions
            stimes, smags, serrs = this_times, this_mags, this_errs

    # this handles sigclipping for asymmetric +ve and -ve clip values
    elif sigclip and isinstance(sigclip, (list,tuple)) and len(sigclip) == 2:

        # sigclip is passed as [dimmingclip, brighteningclip]
        dimmingclip = sigclip[0]
        brighteningclip = sigclip[1]

        if not iterative and niterations is None:

            if magsarefluxes:
                nottoodimind = (
                    (fmags - center_mag) > (-dimmingclip*stddev_mag)
                )
                nottoobrightind = (
                    (fmags - center_mag) < (brighteningclip*stddev_mag)
                )
            else:
                nottoodimind = (
                    (fmags - center_mag) < (dimmingclip*stddev_mag)
                )
                nottoobrightind = (
                    (fmags - center_mag) > (-brighteningclip*stddev_mag)
                )

            sigind = nottoodimind & nottoobrightind

            stimes = ftimes[sigind]
            smags = fmags[sigind]
            serrs = ferrs[sigind]

        else:

            #
            # iterative version adapted from scipy.stats.sigmaclip
            #
            if niterations is None:

                delta = 1

                this_times = ftimes
                this_mags = fmags
                this_errs = ferrs

                while delta:

                    if meanormedian == 'mean':

                        this_center = npmean(this_mags)
                        this_stdev = npstddev(this_mags)

                    elif meanormedian == 'median':
                        this_center = npmedian(this_mags)
                        this_stdev = (npmedian(npabs(this_mags -
                                                     this_center))) * 1.483
                    this_size = this_mags.size

                    if magsarefluxes:
                        nottoodimind = (
                            (this_mags - this_center) >
                            (-dimmingclip*this_stdev)
                        )
                        nottoobrightind = (
                            (this_mags - this_center) <
                            (brighteningclip*this_stdev)
                        )
                    else:
                        nottoodimind = (
                            (this_mags - this_center) <
                            (dimmingclip*this_stdev)
                        )
                        nottoobrightind = (
                            (this_mags - this_center) >
                            (-brighteningclip*this_stdev)
                        )

                    # apply the sigclip
                    tsi = nottoodimind & nottoobrightind

                    # update the arrays
                    this_times = this_times[tsi]
                    this_mags = this_mags[tsi]
                    this_errs = this_errs[tsi]

                    # update delta and go to top of the loop
                    delta = this_size - this_mags.size

            else:  # If iterating only a certain number of times
                this_times = ftimes
                this_mags = fmags
                this_errs = ferrs

                iter_num = 0
                delta = 1

                while iter_num < niterations and delta:

                    if meanormedian == 'mean':
                        this_center = npmean(this_mags)
                        this_stdev = npstddev(this_mags)
                    elif meanormedian == 'median':
                        this_center = npmedian(this_mags)
                        this_stdev = (npmedian(npabs(this_mags -
                                                     this_center))) * 1.483
                    this_size = this_mags.size

                    if magsarefluxes:
                        nottoodimind = (
                            (this_mags - this_center) >
                            (-dimmingclip*this_stdev)
                        )
                        nottoobrightind = (
                            (this_mags - this_center) <
                            (brighteningclip*this_stdev)
                        )
                    else:
                        nottoodimind = (
                            (this_mags - this_center) < (dimmingclip*this_stdev)
                        )
                        nottoobrightind = (
                            (this_mags - this_center) >
                            (-brighteningclip*this_stdev)
                        )

                    # apply the sigclip
                    tsi = nottoodimind & nottoobrightind

                    # update the arrays
                    this_times = this_times[tsi]
                    this_mags = this_mags[tsi]
                    this_errs = this_errs[tsi]

                    # update the number of iterations and delta
                    # and go to top of the loop
                    delta = this_size - this_mags.size
                    iter_num += 1

            # final sigclipped versions
            stimes, smags, serrs = this_times, this_mags, this_errs

    else:

        stimes = ftimes
        smags = fmags
        serrs = ferrs

    if returnerrs:
        return stimes, smags, serrs
    else:
        return stimes, smags, None

def phase_magseries(times, mags, period, epoch, wrap=True, sort=True):
    '''Phases a magnitude/flux time-series using a given period and epoch.
    The equation used is::
        phase = (times - epoch)/period - floor((times - epoch)/period)
    This phases the given magnitude timeseries using the given period and
    epoch. If wrap is True, wraps the result around 0.0 (and returns an array
    that has twice the number of the original elements). If sort is True,
    returns the magnitude timeseries in phase sorted order.
    Parameters
    ----------
    times,mags : np.array
        The magnitude/flux time-series values to phase using the provided
        `period` and `epoch`. Non-fiinite values will be removed.
    period : float
        The period to use to phase the time-series.
    epoch : float
        The epoch to phase the time-series. This is usually the time-of-minimum
        or time-of-maximum of some periodic light curve
        phenomenon. Alternatively, one can use the minimum time value in
        `times`.
    wrap : bool
        If this is True, the returned phased time-series will be wrapped around
        phase 0.0, which is useful for plotting purposes. The arrays returned
        will have twice the number of input elements because of this wrapping.
    sort : bool
        If this is True, the returned phased time-series will be sorted in
        increasing phase order.
    Returns
    -------
    dict
        A dict of the following form is returned::
            {'phase': the phase values,
             'mags': the mags/flux values at each phase,
             'period': the input `period` used to phase the time-series,
             'epoch': the input `epoch` used to phase the time-series}
    '''

    # find all the finite values of the magnitudes and times
    finiteind = np.isfinite(mags) & np.isfinite(times)

    finite_times = times[finiteind]
    finite_mags = mags[finiteind]

    magseries_phase = (
        (finite_times - epoch)/period -
        np.floor(((finite_times - epoch)/period))
    )

    outdict = {'phase':magseries_phase,
               'mags':finite_mags,
               'period':period,
               'epoch':epoch}

    if sort:
        sortorder = np.argsort(outdict['phase'])
        outdict['phase'] = outdict['phase'][sortorder]
        outdict['mags'] = outdict['mags'][sortorder]

    if wrap:
        outdict['phase'] = np.concatenate((outdict['phase']-1.0,
                                           outdict['phase']))
        outdict['mags'] = np.concatenate((outdict['mags'],
                                          outdict['mags']))

    return outdict


def aov_periodfind(times,
                   mags,
                   errs,
                   magsarefluxes=False,
                   startp=None,
                   endp=None,
                   stepsize=1.0e-4,
                   autofreq=True,
                   normalize=True,
                   phasebinsize=0.05,
                   mindetperbin=9,
                   nbestpeaks=5,
                   periodepsilon=0.1,
                   sigclip=10.0,
                   nworkers=None,
                   verbose=True,
                   periodPath=None, variableName="NoName"):
    '''This runs a parallelized Analysis-of-Variance (AoV) period search.

    NOTE: `normalize = True` here as recommended by Schwarzenberg-Czerny 1996,
    i.e. mags will be normalized to zero and rescaled so their variance = 1.0.

    Parameters
    ----------

    times,mags,errs : np.array
        The mag/flux time-series with associated measurement errors to run the
        period-finding on.

    magsarefluxes : bool
        If the input measurement values in `mags` and `errs` are in fluxes, set
        this to True.

    startp,endp : float or None
        The minimum and maximum periods to consider for the transit search.

    stepsize : float
        The step-size in frequency to use when constructing a frequency grid for
        the period search.

    autofreq : bool
        If this is True, the value of `stepsize` will be ignored and the
        :py:func:`astrobase.periodbase.get_frequency_grid` function will be used
        to generate a frequency grid based on `startp`, and `endp`. If these are
        None as well, `startp` will be set to 0.1 and `endp` will be set to
        `times.max() - times.min()`.

    normalize : bool
        This sets if the input time-series is normalized to 0.0 and rescaled
        such that its variance = 1.0. This is the recommended procedure by
        Schwarzenberg-Czerny 1996.

    phasebinsize : float
        The bin size in phase to use when calculating the AoV theta statistic at
        a test frequency.

    mindetperbin : int
        The minimum number of elements in a phase bin to consider it valid when
        calculating the AoV theta statistic at a test frequency.

    nbestpeaks : int
        The number of 'best' peaks to return from the periodogram results,
        starting from the global maximum of the periodogram peak values.

    periodepsilon : float
        The fractional difference between successive values of 'best' periods
        when sorting by periodogram power to consider them as separate periods
        (as opposed to part of the same periodogram peak). This is used to avoid
        broad peaks in the periodogram and make sure the 'best' periods returned
        are all actually independent.

    sigclip : float or int or sequence of two floats/ints or None
        If a single float or int, a symmetric sigma-clip will be performed using
        the number provided as the sigma-multiplier to cut out from the input
        time-series.

        If a list of two ints/floats is provided, the function will perform an
        'asymmetric' sigma-clip. The first element in this list is the sigma
        value to use for fainter flux/mag values; the second element in this
        list is the sigma value to use for brighter flux/mag values. For
        example, `sigclip=[10., 3.]`, will sigclip out greater than 10-sigma
        dimmings and greater than 3-sigma brightenings. Here the meaning of
        "dimming" and "brightening" is set by *physics* (not the magnitude
        system), which is why the `magsarefluxes` kwarg must be correctly set.

        If `sigclip` is None, no sigma-clipping will be performed, and the
        time-series (with non-finite elems removed) will be passed through to
        the output.

    nworkers : int
        The number of parallel workers to use when calculating the periodogram.

    verbose : bool
        If this is True, will indicate progress and details about the frequency
        grid used for the period search.

    Returns
    -------

    dict
        This function returns a dict, referred to as an `lspinfo` dict in other
        astrobase functions that operate on periodogram results. This is a
        standardized format across all astrobase period-finders, and is of the
        form below::

            {'bestperiod': the best period value in the periodogram,
             'bestlspval': the periodogram peak associated with the best period,
             'nbestpeaks': the input value of nbestpeaks,
             'nbestlspvals': nbestpeaks-size list of best period peak values,
             'nbestperiods': nbestpeaks-size list of best periods,
             'lspvals': the full array of periodogram powers,
             'periods': the full array of periods considered,
             'method':'aov' -> the name of the period-finder method,
             'kwargs':{ dict of all of the input kwargs for record-keeping}}

    '''

    # get rid of nans first and sigclip
    stimes, smags, serrs = sigclip_magseries(times,
                                             mags,
                                             errs,
                                             magsarefluxes=magsarefluxes,
                                             sigclip=sigclip)
    stimes, smags, serrs = resort_by_time(stimes, smags, serrs)

    # make sure there are enough points to calculate a spectrum
    if len(stimes) > 9 and len(smags) > 9 and len(serrs) > 9:

        # get the frequencies to use
        if startp:
            endf = 1.0/startp
        else:
            # default start period is 0.1 day
            endf = 1.0/0.1

        if endp:
            startf = 1.0/endp
        else:
            # default end period is length of time series
            startf = 1.0/(stimes.max() - stimes.min())

        # if we're not using autofreq, then use the provided frequencies
        if not autofreq:
            frequencies = nparange(startf, endf, stepsize)

        else:
            # this gets an automatic grid of frequencies to use
            frequencies = get_frequency_grid(stimes,
                                             minfreq=startf,
                                             maxfreq=endf)

        # renormalize the working mags to zero and scale them so that the
        # variance = 1 for use with our LSP functions
        if normalize:
            nmags = (smags - npmedian(smags))/npstd(smags)
        else:
            nmags = smags


        lsp=[]
        for x in frequencies:
            theta = aov_theta(stimes, nmags, serrs, x,
                          binsize=phasebinsize, minbin=mindetperbin)
            lsp.append(theta)

        lsp = nparray(lsp)
        periods = 1.0/frequencies

        plt.plot(periods, lsp)
        #plt.gca().invert_yaxis()
        #plt.title("Range {0} d  Steps: {1}".format(trialRange, periodsteps))
        plt.xlabel(r"Trial Period")
        plt.ylabel(r"Likelihood of Period")
        plt.savefig(periodPath / f"{variableName}_ANOVALikelihoodPlot.png")
        plt.clf()

        # find the nbestpeaks for the periodogram: 1. sort the lsp array by
        # highest value first 2. go down the values until we find five
        # values that are separated by at least periodepsilon in period

        # make sure to filter out non-finite values
        finitepeakind = npisfinite(lsp)
        finlsp = lsp[finitepeakind]
        finperiods = periods[finitepeakind]

        # make sure that finlsp has finite values before we work on it
        try:

            bestperiodind = npargmax(finlsp)

        except ValueError:

            logger.info('no finite periodogram values '
                     'for this mag series, skipping...')
            return {'bestperiod':npnan,
                    'bestlspval':npnan,
                    'nbestpeaks':nbestpeaks,
                    'nbestlspvals':None,
                    'nbestperiods':None,
                    'lspvals':None,
                    'periods':None,
                    'method':'aov',
                    'kwargs':{'startp':startp,
                              'endp':endp,
                              'stepsize':stepsize,
                              'normalize':normalize,
                              'phasebinsize':phasebinsize,
                              'mindetperbin':mindetperbin,
                              'autofreq':autofreq,
                              'periodepsilon':periodepsilon,
                              'nbestpeaks':nbestpeaks,
                              'sigclip':sigclip}}

        sortedlspind = npargsort(finlsp)[::-1]
        sortedlspperiods = finperiods[sortedlspind]
        sortedlspvals = finlsp[sortedlspind]

        # now get the nbestpeaks
        nbestperiods, nbestlspvals, peakcount = (
            [finperiods[bestperiodind]],
            [finlsp[bestperiodind]],
            1
        )
        prevperiod = sortedlspperiods[0]

        # find the best nbestpeaks in the lsp and their periods
        for period, lspval in zip(sortedlspperiods, sortedlspvals):

            if peakcount == nbestpeaks:
                break
            perioddiff = abs(period - prevperiod)
            bestperiodsdiff = [abs(period - x) for x in nbestperiods]

            # print('prevperiod = %s, thisperiod = %s, '
            #       'perioddiff = %s, peakcount = %s' %
            #       (prevperiod, period, perioddiff, peakcount))

            # this ensures that this period is different from the last
            # period and from all the other existing best periods by
            # periodepsilon to make sure we jump to an entire different peak
            # in the periodogram
            if (perioddiff > (periodepsilon*prevperiod) and
                all(x > (periodepsilon*period) for x in bestperiodsdiff)):
                nbestperiods.append(period)
                nbestlspvals.append(lspval)
                peakcount = peakcount + 1

            prevperiod = period

        return {'bestperiod':finperiods[bestperiodind],
                'bestlspval':finlsp[bestperiodind],
                'nbestpeaks':nbestpeaks,
                'nbestlspvals':nbestlspvals,
                'nbestperiods':nbestperiods,
                'lspvals':lsp,
                'periods':periods,
                'method':'aov',
                'kwargs':{'startp':startp,
                          'endp':endp,
                          'stepsize':stepsize,
                          'normalize':normalize,
                          'phasebinsize':phasebinsize,
                          'mindetperbin':mindetperbin,
                          'autofreq':autofreq,
                          'periodepsilon':periodepsilon,
                          'nbestpeaks':nbestpeaks,
                          'sigclip':sigclip}}

    else:

        logger.info('no good detections for these times and mags, skipping...')
        return {'bestperiod':npnan,
                'bestlspval':npnan,
                'nbestpeaks':nbestpeaks,
                'nbestlspvals':None,
                'nbestperiods':None,
                'lspvals':None,
                'periods':None,
                'method':'aov',
                'kwargs':{'startp':startp,
                          'endp':endp,
                          'stepsize':stepsize,
                          'normalize':normalize,
                          'phasebinsize':phasebinsize,
                          'mindetperbin':mindetperbin,
                          'autofreq':autofreq,
                          'periodepsilon':periodepsilon,
                          'nbestpeaks':nbestpeaks,
                          'sigclip':sigclip}}


################################################### MULTIHARMONIC BARRIER

def phase_magseries_with_errs(times, mags, errs, period, epoch,
                              wrap=True, sort=True):
    '''Phases a magnitude/flux time-series using a given period and epoch.
    The equation used is::
        phase = (times - epoch)/period - floor((times - epoch)/period)
    This phases the given magnitude timeseries using the given period and
    epoch. If wrap is True, wraps the result around 0.0 (and returns an array
    that has twice the number of the original elements). If sort is True,
    returns the magnitude timeseries in phase sorted order.
    Parameters
    ----------
    times,mags,errs : np.array
        The magnitude/flux time-series values and associated measurement errors
        to phase using the provided `period` and `epoch`. Non-fiinite values
        will be removed.
    period : float
        The period to use to phase the time-series.
    epoch : float
        The epoch to phase the time-series. This is usually the time-of-minimum
        or time-of-maximum of some periodic light curve
        phenomenon. Alternatively, one can use the minimum time value in
        `times`.
    wrap : bool
        If this is True, the returned phased time-series will be wrapped around
        phase 0.0, which is useful for plotting purposes. The arrays returned
        will have twice the number of input elements because of this wrapping.
    sort : bool
        If this is True, the returned phased time-series will be sorted in
        increasing phase order.
    Returns
    -------
    dict
        A dict of the following form is returned::
            {'phase': the phase values,
             'mags': the mags/flux values at each phase,
             'errs': the err values at each phase,
             'period': the input `period` used to phase the time-series,
             'epoch': the input `epoch` used to phase the time-series}
    '''

    # find all the finite values of the magnitudes and times
    finiteind = np.isfinite(mags)
    finite_times = times[finiteind]
    finite_mags = mags[finiteind]
    finite_errs = errs[finiteind]

    magseries_phase = (
        (finite_times - epoch)/period -
        np.floor(((finite_times - epoch)/period))
    )

    outdict = {'phase':magseries_phase,
               'mags':finite_mags,
               'errs':finite_errs,
               'period':period,
               'epoch':epoch}

    if sort:
        sortorder = np.argsort(outdict['phase'])
        outdict['phase'] = outdict['phase'][sortorder]
        outdict['mags'] = outdict['mags'][sortorder]
        outdict['errs'] = outdict['errs'][sortorder]

    if wrap:
        outdict['phase'] = np.concatenate((outdict['phase']-1.0,
                                           outdict['phase']))
        outdict['mags'] = np.concatenate((outdict['mags'],
                                          outdict['mags']))
        outdict['errs'] = np.concatenate((outdict['errs'],
                                          outdict['errs']))

    return outdict

def aovhm_theta(times, mags, errs, frequency,
                nharmonics, magvariance):
    '''This calculates the harmonic AoV theta statistic for a frequency.

    This is a mostly faithful translation of the inner loop in `aovper.f90`. See
    the following for details:

    - http://users.camk.edu.pl/alex/
    - Schwarzenberg-Czerny (`1996
      <http://iopscience.iop.org/article/10.1086/309985/meta>`_)

    Schwarzenberg-Czerny (1996) equation 11::

        theta_prefactor = (K - 2N - 1)/(2N)
        theta_top = sum(c_n*c_n) (from n=0 to n=2N)
        theta_bot = variance(timeseries) - sum(c_n*c_n) (from n=0 to n=2N)

        theta = theta_prefactor * (theta_top/theta_bot)

        N = number of harmonics (nharmonics)
        K = length of time series (times.size)

    Parameters
    ----------

    times,mags,errs : np.array
        The input time-series to calculate the test statistic for. These should
        all be of nans/infs and be normalized to zero.

    frequency : float
        The test frequency to calculate the statistic for.

    nharmonics : int
        The number of harmonics to calculate up to.The recommended range is 4 to
        8.

    magvariance : float
        This is the (weighted by errors) variance of the magnitude time
        series. We provide it as a pre-calculated value here so we don't have to
        re-calculate it for every worker.

    Returns
    -------

    aov_harmonic_theta : float
        THe value of the harmonic AoV theta for the specified test `frequency`.

    '''

    period = 1.0/frequency

    ndet = times.size
    two_nharmonics = nharmonics + nharmonics

    # phase with test period
    phasedseries = phase_magseries_with_errs(
        times, mags, errs, period, times[0],
        sort=True, wrap=False
    )

    # get the phased quantities
    phase = phasedseries['phase']
    pmags = phasedseries['mags']
    perrs = phasedseries['errs']

    # this is sqrt(1.0/errs^2) -> the weights
    pweights = 1.0/perrs

    # multiply by 2.0*PI (for omega*time)
    phase = phase * 2.0 * pi_value

    # this is the z complex vector
    z = npcos(phase) + 1.0j*npsin(phase)

    # multiply phase with N
    phase = nharmonics * phase

    # this is the psi complex vector
    psi = pmags * pweights * (npcos(phase) + 1j*npsin(phase))

    # this is the initial value of z^n
    zn = 1.0 + 0.0j

    # this is the initial value of phi
    phi = pweights + 0.0j

    # initialize theta to zero
    theta_aov = 0.0

    # go through all the harmonics now up to 2N
    for _ in range(two_nharmonics):

        # this is <phi, phi>
        phi_dot_phi = npsum(phi * phi.conjugate())

        # this is the alpha_n numerator
        alpha = npsum(pweights * z * phi)

        # this is <phi, psi>. make sure to use npvdot and NOT npdot to get
        # complex conjugate of first vector as expected for complex vectors
        phi_dot_psi = npvdot(phi, psi)

        # make sure phi_dot_phi is not zero
        phi_dot_phi = npmax([phi_dot_phi, 10.0e-9])

        # this is the expression for alpha_n
        alpha = alpha / phi_dot_phi

        # update theta_aov for this harmonic
        theta_aov = (theta_aov +
                     npabs(phi_dot_psi) * npabs(phi_dot_psi) / phi_dot_phi)

        # use the recurrence relation to find the next phi
        phi = phi * z - alpha * zn * phi.conjugate()

        # update z^n
        zn = zn * z

    # done with all harmonics, calculate the theta_aov for this freq
    # the max below makes sure that magvariance - theta_aov > zero
    theta_aov = ( (ndet - two_nharmonics - 1.0) * theta_aov /
                  (two_nharmonics * npmax([magvariance - theta_aov,
                                           1.0e-9])) )

    return theta_aov

def aovhm_periodfind(times,
                     mags,
                     errs,
                     magsarefluxes=False,
                     startp=None,
                     endp=None,
                     stepsize=1.0e-4,
                     autofreq=True,
                     normalize=True,
                     nharmonics=6,
                     nbestpeaks=5,
                     periodepsilon=0.1,
                     sigclip=10.0,
                     nworkers=None,
                     verbose=True,
                     periodPath=None, variableName="NoName"):
    '''This runs a parallelized harmonic Analysis-of-Variance (AoV) period
    search.

    NOTE: normalize = True here as recommended by Schwarzenberg-Czerny 1996,
    i.e. mags will be normalized to zero and rescaled so their variance = 1.0.

    Parameters
    ----------

    times,mags,errs : np.array
        The mag/flux time-series with associated measurement errors to run the
        period-finding on.

    magsarefluxes : bool
        If the input measurement values in `mags` and `errs` are in fluxes, set
        this to True.

    startp,endp : float or None
        The minimum and maximum periods to consider for the transit search.

    stepsize : float
        The step-size in frequency to use when constructing a frequency grid for
        the period search.

    autofreq : bool
        If this is True, the value of `stepsize` will be ignored and the
        :py:func:`astrobase.periodbase.get_frequency_grid` function will be used
        to generate a frequency grid based on `startp`, and `endp`. If these are
        None as well, `startp` will be set to 0.1 and `endp` will be set to
        `times.max() - times.min()`.

    normalize : bool
        This sets if the input time-series is normalized to 0.0 and rescaled
        such that its variance = 1.0. This is the recommended procedure by
        Schwarzenberg-Czerny 1996.

    nharmonics : int
        The number of harmonics to use when calculating the AoV theta value at a
        test frequency. This should be between 4 and 8 in most cases.

    nbestpeaks : int
        The number of 'best' peaks to return from the periodogram results,
        starting from the global maximum of the periodogram peak values.

    periodepsilon : float
        The fractional difference between successive values of 'best' periods
        when sorting by periodogram power to consider them as separate periods
        (as opposed to part of the same periodogram peak). This is used to avoid
        broad peaks in the periodogram and make sure the 'best' periods returned
        are all actually independent.

    sigclip : float or int or sequence of two floats/ints or None
        If a single float or int, a symmetric sigma-clip will be performed using
        the number provided as the sigma-multiplier to cut out from the input
        time-series.

        If a list of two ints/floats is provided, the function will perform an
        'asymmetric' sigma-clip. The first element in this list is the sigma
        value to use for fainter flux/mag values; the second element in this
        list is the sigma value to use for brighter flux/mag values. For
        example, `sigclip=[10., 3.]`, will sigclip out greater than 10-sigma
        dimmings and greater than 3-sigma brightenings. Here the meaning of
        "dimming" and "brightening" is set by *physics* (not the magnitude
        system), which is why the `magsarefluxes` kwarg must be correctly set.

        If `sigclip` is None, no sigma-clipping will be performed, and the
        time-series (with non-finite elems removed) will be passed through to
        the output.

    nworkers : int
        The number of parallel workers to use when calculating the periodogram.

    verbose : bool
        If this is True, will indicate progress and details about the frequency
        grid used for the period search.

    Returns
    -------

    dict
        This function returns a dict, referred to as an `lspinfo` dict in other
        astrobase functions that operate on periodogram results. This is a
        standardized format across all astrobase period-finders, and is of the
        form below::

            {'bestperiod': the best period value in the periodogram,
             'bestlspval': the periodogram peak associated with the best period,
             'nbestpeaks': the input value of nbestpeaks,
             'nbestlspvals': nbestpeaks-size list of best period peak values,
             'nbestperiods': nbestpeaks-size list of best periods,
             'lspvals': the full array of periodogram powers,
             'periods': the full array of periods considered,
             'method':'mav' -> the name of the period-finder method,
             'kwargs':{ dict of all of the input kwargs for record-keeping}}

    '''

    # get rid of nans first and sigclip
    stimes, smags, serrs = sigclip_magseries(times,
                                             mags,
                                             errs,
                                             magsarefluxes=magsarefluxes,
                                             sigclip=sigclip)
    stimes, smags, serrs = resort_by_time(stimes, smags, serrs)

    # make sure there are enough points to calculate a spectrum
    if len(stimes) > 9 and len(smags) > 9 and len(serrs) > 9:

        # get the frequencies to use
        if startp:
            endf = 1.0/startp
        else:
            # default start period is 0.1 day
            endf = 1.0/0.1

        if endp:
            startf = 1.0/endp
        else:
            # default end period is length of time series
            startf = 1.0/(stimes.max() - stimes.min())

        # if we're not using autofreq, then use the provided frequencies
        if not autofreq:
            frequencies = nparange(startf, endf, stepsize)

        else:
            # this gets an automatic grid of frequencies to use
            frequencies = get_frequency_grid(stimes,
                                             minfreq=startf,
                                             maxfreq=endf)

        # map to parallel workers
        if (not nworkers) or (nworkers > NCPUS):
            nworkers = NCPUS

        # renormalize the working mags to zero and scale them so that the
        # variance = 1 for use with our LSP functions
        if normalize:
            nmags = (smags - npmedian(smags))/npstd(smags)
        else:
            nmags = smags

        # figure out the weighted variance
        # www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weighvar.pdf
        magvariance_top = npsum(nmags/(serrs*serrs))
        magvariance_bot = (nmags.size - 1)*npsum(1.0/(serrs*serrs)) / nmags.size
        magvariance = magvariance_top/magvariance_bot

        #tasks = [(stimes, nmags, serrs, x, nharmonics, magvariance)
        #         for x in frequencies]

        lsp=[]
        for x in frequencies:
            lsp.append(aovhm_theta(times, mags, errs, x, nharmonics, magvariance))

        lsp = nparray(lsp)
        periods = 1.0/frequencies

        plt.plot(periods, lsp)
        #plt.gca().invert_yaxis()
        #plt.title("Range {0} d  Steps: {1}".format(trialRange, periodsteps))
        plt.xlabel(r"Trial Period")
        plt.ylabel(r"Likelihood of Period")
        plt.savefig(periodPath / f"{variableName}_ANOVAharmonic_LikelihoodPlot.png")
        plt.clf()


        # find the nbestpeaks for the periodogram: 1. sort the lsp array by
        # highest value first 2. go down the values until we find five
        # values that are separated by at least periodepsilon in period

        # make sure to filter out non-finite values
        finitepeakind = npisfinite(lsp)
        finlsp = lsp[finitepeakind]
        finperiods = periods[finitepeakind]

        # make sure that finlsp has finite values before we work on it
        try:

            bestperiodind = npargmax(finlsp)

        except ValueError:

            logger.info('no finite periodogram values '
                     'for this mag series, skipping...')
            return {'bestperiod':npnan,
                    'bestlspval':npnan,
                    'nbestpeaks':nbestpeaks,
                    'nbestlspvals':None,
                    'nbestperiods':None,
                    'lspvals':None,
                    'periods':None,
                    'method':'mav',
                    'kwargs':{'startp':startp,
                              'endp':endp,
                              'stepsize':stepsize,
                              'normalize':normalize,
                              'nharmonics':nharmonics,
                              'autofreq':autofreq,
                              'periodepsilon':periodepsilon,
                              'nbestpeaks':nbestpeaks,
                              'sigclip':sigclip}}

        sortedlspind = npargsort(finlsp)[::-1]
        sortedlspperiods = finperiods[sortedlspind]
        sortedlspvals = finlsp[sortedlspind]

        # now get the nbestpeaks
        nbestperiods, nbestlspvals, peakcount = (
            [finperiods[bestperiodind]],
            [finlsp[bestperiodind]],
            1
        )
        prevperiod = sortedlspperiods[0]

        # find the best nbestpeaks in the lsp and their periods
        for period, lspval in zip(sortedlspperiods, sortedlspvals):

            if peakcount == nbestpeaks:
                break
            perioddiff = abs(period - prevperiod)
            bestperiodsdiff = [abs(period - x) for x in nbestperiods]

            # print('prevperiod = %s, thisperiod = %s, '
            #       'perioddiff = %s, peakcount = %s' %
            #       (prevperiod, period, perioddiff, peakcount))

            # this ensures that this period is different from the last
            # period and from all the other existing best periods by
            # periodepsilon to make sure we jump to an entire different peak
            # in the periodogram
            if (perioddiff > (periodepsilon*prevperiod) and
                all(x > (periodepsilon*period) for x in bestperiodsdiff)):
                nbestperiods.append(period)
                nbestlspvals.append(lspval)
                peakcount = peakcount + 1

            prevperiod = period

        return {'bestperiod':finperiods[bestperiodind],
                'bestlspval':finlsp[bestperiodind],
                'nbestpeaks':nbestpeaks,
                'nbestlspvals':nbestlspvals,
                'nbestperiods':nbestperiods,
                'lspvals':lsp,
                'periods':periods,
                'method':'mav',
                'kwargs':{'startp':startp,
                          'endp':endp,
                          'stepsize':stepsize,
                          'normalize':normalize,
                          'nharmonics':nharmonics,
                          'autofreq':autofreq,
                          'periodepsilon':periodepsilon,
                          'nbestpeaks':nbestpeaks,
                          'sigclip':sigclip}}

    else:

        logger.info('no good detections for these times and mags, skipping...')
        return {'bestperiod':npnan,
                'bestlspval':npnan,
                'nbestpeaks':nbestpeaks,
                'nbestlspvals':None,
                'nbestperiods':None,
                'lspvals':None,
                'periods':None,
                'method':'mav',
                'kwargs':{'startp':startp,
                          'endp':endp,
                          'stepsize':stepsize,
                          'normalize':normalize,
                          'nharmonics':nharmonics,
                          'autofreq':autofreq,
                          'periodepsilon':periodepsilon,
                          'nbestpeaks':nbestpeaks,
                          'sigclip':sigclip}}

#########################################

def sortByPhase (phases, fluxes):
    phaseIndices = asarray(phases).argsort()
    sortedPhases = []
    sortedFluxes = []
    for i in range(0,len(phases)):
        sortedPhases.append(phases[phaseIndices[i]])
        sortedFluxes.append(fluxes[phaseIndices[i]])

    return (sortedPhases, sortedFluxes)

#########################################

def normalize(fluxes):

    normalizedFluxes = []

    for flux in fluxes:
        normalizedFlux = (flux - min(fluxes)) / (max(fluxes) - min(fluxes))
        normalizedFluxes.append(normalizedFlux)

    return(normalizedFluxes)

#########################################

def getPhases(julian_dates, fluxes, period):

    phases = []

    for jd in julian_dates:
        phases.append((jd / period) % 1)

    (sortedPhases, sortedFluxes) = sortByPhase(phases, fluxes)

    return(sortedPhases, sortedFluxes)

#########################################

# Find the value in array2 corresponding to the minimum value in array1.

def find_minimum(array1, array2):
    value_to_return = 0.0

    minimum = min(array1)

    for i in range(0, len(array1)):
        if (array1[i] == minimum):
          value_to_return = array2[i]

    return (value_to_return, minimum)

#########################################

def sum_distances (sortedPhases, sortedNormalizedFluxes):

    distanceSum = 0.0

    for i in range(0, (len(sortedPhases) - 1)):
        fluxdiff = sortedNormalizedFluxes[i + 1] - sortedNormalizedFluxes[i]
        phasediff = sortedPhases[i + 1] - sortedPhases[i]

        distanceSum = distanceSum + (((fluxdiff ** 2) + (phasediff ** 2)) ** 0.5)

    return(distanceSum)

#########################################

def sum_stdevs (sortedPhases, sortedNormalizedFluxes, numBins):

    stdevSum = 0.0

    for i in range (0, numBins):
        fluxes_inrange = []
        minIndex = (float(i) / float(numBins)) * float(len(sortedPhases))
        maxIndex = (float(i + 1) / float(numBins)) * float(len(sortedPhases))

        for j in range (0, len(sortedPhases)):
            if (j >= minIndex and j < maxIndex):
                fluxes_inrange.append(sortedNormalizedFluxes[j])

        stdev_of_bin_i = std(fluxes_inrange)
        stdevSum = stdevSum + stdev_of_bin_i

    return(stdevSum)

#########################################

def phase_dispersion_minimization(varData, periodsteps, minperiod, maxperiod, numBins, periodPath, variableName):

    periodguess_array = []
    distance_results = []
    stdev_results = []

    (julian_dates, fluxes) = (varData[:,0],varData[:,1])
    normalizedFluxes = normalize(fluxes)

    for r in range(periodsteps):
        periodguess = minperiod + (r * ((maxperiod-minperiod)/periodsteps))
        (sortedPhases, sortedNormalizedFluxes) = getPhases(julian_dates, normalizedFluxes, periodguess)

        distance_sum = sum_distances(sortedPhases, sortedNormalizedFluxes)
        stdev_sum = sum_stdevs(sortedPhases, sortedNormalizedFluxes, numBins)

        periodguess_array.append(periodguess)
        distance_results.append(distance_sum)
        stdev_results.append(stdev_sum)

    periodTrialMatrix=[]
    for r in range(periodsteps):
        periodTrialMatrix.append([periodguess_array[r],distance_results[r],stdev_results[r]])
    periodTrialMatrix=np.asarray(periodTrialMatrix)
    np.savetxt(periodPath / f"{variableName}_Trials.csv", periodTrialMatrix, delimiter=",", fmt='%0.8f')

    (distance_minperiod, distance_min) = find_minimum(distance_results, periodguess_array)
    (stdev_minperiod, stdev_min) = find_minimum(stdev_results, periodguess_array)

    pdm = {}
    pdm["periodguess_array"] = periodguess_array
    pdm["distance_results"] = distance_results
    pdm["distance_minperiod"] = distance_minperiod
    pdm["stdev_results"] = stdev_results
    pdm["stdev_minperiod"] = stdev_minperiod

    # Estimating the error
    # stdev method
    # Get deviation to the left
    totalRange=np.max(stdev_results) - np.min(stdev_results)


    if np.isnan(pdm["stdev_results"][0]) or pdm["stdev_results"][0] == 0.0:
        logger.info("Not enough datapoint coverage to undertake a Phase-Dispersion Minimization routine.")

    else:
        for q in range(len(periodguess_array)):
            if periodguess_array[q]==pdm["stdev_minperiod"]:
                beginIndex=q
                beginValue=stdev_results[q]

        currentperiod=stdev_minperiod
        stepper=0
        thresholdvalue=beginValue+(0.5*totalRange)

        while True:
            if stdev_results[beginIndex-stepper] > thresholdvalue:
                lefthandP=periodguess_array[beginIndex-stepper]
                break
            stepper=stepper+1

        stepper=0
        thresholdvalue=beginValue+(0.5*totalRange)

        while True:
            if beginIndex+stepper+1 == periodsteps:
                righthandP=periodguess_array[beginIndex+stepper]
                logger.debug("Warning: Peak period for stdev method too close to top of range")
                break
            if stdev_results[beginIndex+stepper] > thresholdvalue:
                righthandP=periodguess_array[beginIndex+stepper]
                break
            stepper=stepper+1


        #print ("Stdev method error: " + str((righthandP - lefthandP)/2))
        pdm["stdev_error"] = (righthandP - lefthandP)/2


    # Estimating the error
    # stdev method
    # Get deviation to the left
    totalRange=np.max(distance_results) - np.min(distance_results)
    for q in range(len(periodguess_array)):
        if periodguess_array[q]==pdm["distance_minperiod"]:
            beginIndex=q
            beginValue=distance_results[q]
    currentperiod=distance_minperiod
    stepper=0
    thresholdvalue=beginValue+(0.5*totalRange)
    while True:
        if distance_results[beginIndex-stepper] > thresholdvalue:
            lefthandP=periodguess_array[beginIndex-stepper]
            break
        stepper=stepper+1

    stepper=0
    thresholdvalue=beginValue+(0.5*totalRange)
    while True:
        if beginIndex+stepper+1 == periodsteps:
            righthandP=periodguess_array[beginIndex+stepper]
            logger.debug("Warning: Peak period for distance method too close to top of range")
            break

        if distance_results[beginIndex+stepper] > thresholdvalue:
            righthandP=periodguess_array[beginIndex+stepper]
            break
        stepper=stepper+1

    pdm["distance_error"] = (righthandP - lefthandP)/2

    return (pdm)

#########################################

def LombScargleMultiterm(infile, t, m, d, periodlower=0.2, periodupper=2.5, nterms=1, multisearch=False, samples=5,
                         disablelightcurve=False, periodPath=False, variableName="NoName"):
    #print(
    #    'using ' + str(samples) + ' samples per peak, start P = ' + str(periodlower) + ', end P = ' + str(periodupper))
    # Calculate the Lomb-Scargle periodogram values
    ls = LombScargle(t, m, d, nterms=nterms)
    freq, power = ls.autopower(samples_per_peak=samples, minimum_frequency=1 / periodupper,
                               maximum_frequency=1 / periodlower)

    # Create the likelihood plot
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.set(xlabel='Period',
            ylabel='Power',
            title=' Lomb-Scargle N=' + str(nterms) + ' Likelihood\n Period Range [' + str(
                periodlower) + ', ' + str(periodupper) + ']')
    ax2.plot(1 / freq, power, '-k', rasterized=True)
    tempfile=str(f"{variableName}_LombScargle_N" + str(nterms) + "_LikelihoodPlot.png")
    plt.savefig(periodPath / tempfile)

    # Find peak of the likelihood plot (most likely frequency)
    best_freq = freq[np.argmax(power)]

    if not disablelightcurve:
        # Create phased lightcurve for fourier fit
        phase_fit = np.linspace(0, 1, 100)
        fig, ax = plt.subplots(figsize=(8, 6))

        # Phase observed data points
        ax.errorbar((t * best_freq) % 1, m, d,
                    fmt='.', color='gray', ecolor='lightgray', capsize=0, zorder=1)
        # Plot Fourier fit over to of data
        ax.plot(phase_fit, ls.model(phase_fit / best_freq, best_freq), '-k', lw=2, zorder=5)
        ax.text(0.98, 0.03, "P = {0:.5f} days".format(1 / best_freq),
                ha='right', va='bottom', transform=ax.transAxes)
        ax.set_title(infile.split('/')[-1] + ' ' + str(nterms) + "-term Lomb-Scargle Fourier fit")
        ax.invert_yaxis()
        tempfile=str(f"{variableName}_LombScargle_N" + str(nterms) + "_Lightcurve.png")
        plt.savefig(periodPath / tempfile)

    best_period = 1 / best_freq

    plt.close('all')
    return (best_period)


#########################################


def plot_with_period(paths, filterCode, numBins = 10, minperiod=0.2, maxperiod=1.2, periodsteps=10000):

    if minperiod==-99.9:
            minperiod=0.05

    if maxperiod==-99.9 or maxperiod==None:
        # Load in list of used files
        fileList = []
        with open(paths['parent'] / "usedImages.txt", "r") as f:
            for line in f:
                fileList.append(line.strip())

        dateList=[]
        for file in fileList:
            dateList.append(file.split('_')[2].replace('d','.'))
        dateList=np.asarray(dateList,dtype=float)
        maxperiod=(np.max(dateList)-np.min(dateList))/3 # At least three periods should fit into the dataset.

    if periodsteps == -99:
        periodsteps=int(maxperiod/0.001)
        if periodsteps < 10000:
            periodsteps = 10000

        if periodsteps > 1000000:
            periodsteps = 1000000

    logger.info("Minimum Period Tested  : " +str(minperiod))
    logger.info("Maximum Period Tested  : " +str(maxperiod))
    logger.info("Number of Period Trials: " +str(periodsteps))

    trialRange=[minperiod, maxperiod]

    # Get list of phot files
    periodPath = paths['periods']
    if not periodPath.exists():
        os.makedirs(periodPath)

    logger.debug("Filter Set: " + filterCode)

    fileList = paths['outcatPath'].glob('*_diffExcel.csv')
    with open(paths['parent'] / "periodEstimates.txt", "w") as f:
        f.write("Period Estimates \n\n")

    # Load in the files
    for file in fileList:
        logger.debug(file)
        variableName=file.stem.split('_')[0]
        #logger.debug(str(outcatPath).replace('//',''))
        logger.debug("Variable Name: {}".format(variableName))
        varData = genfromtxt(file, dtype=float, delimiter=',')
        calibFile = file.parent / "{}{}".format(file.stem.replace('diff','calib'), file.suffix)
        logger.debug(calibFile)
        if calibFile.exists():
            calibData=genfromtxt(calibFile, dtype=float, delimiter=',')

            #print (calibData.size)
        if calibFile.exists():
            if (calibData.size > 3):
                pdm=phase_dispersion_minimization(calibData, periodsteps, minperiod, maxperiod, numBins, periodPath, variableName)
        else:
            pdm=phase_dispersion_minimization(varData, periodsteps, minperiod, maxperiod, numBins, periodPath, variableName)

        plt.figure(figsize=(15, 5))

        logger.debug("Distance Method Estimate (days): " + str(pdm["distance_minperiod"]))
        logger.debug("Distance method error: " + str(pdm["distance_error"]))
        
        with open(paths['parent'] / "periodEstimates.txt", "a+") as f:
            f.write("Variable : "+str(variableName) +"\n")
            f.write("Distance Method Estimate (days): " + str(pdm["distance_minperiod"])+"\n")
            f.write("Distance method error: " + str(pdm["distance_error"])+"\n")
        
        plt.plot(pdm["periodguess_array"], pdm["distance_results"])
        plt.gca().invert_yaxis()
        plt.title("Range {0} d  Steps: {1}".format(trialRange, periodsteps))
        plt.xlabel(r"Trial Period")
        plt.ylabel(r"Likelihood of Period")
        plt.savefig(periodPath / f"{variableName}_StringLikelihoodPlot.png")
        plt.clf()
        
        if (varData.size > 3):
            phaseTest=(varData[:,0] / (pdm["distance_minperiod"])) % 1        
    
            plt.plot(phaseTest, varData[:,1], 'bo', linestyle='None')
            plt.plot(phaseTest+1, varData[:,1], 'ro', linestyle='None')
            plt.errorbar(phaseTest, varData[:,1], yerr=varData[:,2], linestyle='None')
            plt.errorbar(phaseTest+1, varData[:,1], yerr=varData[:,2], linestyle='None')
            plt.gca().invert_yaxis()
            plt.title("Period: {0} d  Steps: {1}".format(pdm["distance_minperiod"], periodsteps))
            plt.xlabel(r"Phase ($\phi$)")
            plt.ylabel(f"Differential {filterCode} Magnitude")
            plt.savefig(periodPath / f"{variableName}_StringTestPeriodPlot.png")
            plt.clf()

        if calibFile.exists():
            if (calibData.size > 3):
                phaseTestCalib=(calibData[:,0] / (pdm["distance_minperiod"])) % 1
                plt.plot(phaseTestCalib, calibData[:,1], 'bo', linestyle='None')
                plt.plot(phaseTestCalib+1, calibData[:,1], 'ro', linestyle='None')
                plt.errorbar(phaseTestCalib, calibData[:,1], yerr=calibData[:,2], linestyle='None')
                plt.errorbar(phaseTestCalib+1, calibData[:,1], yerr=calibData[:,2], linestyle='None')
                plt.gca().invert_yaxis()
                plt.title("Period: {0} d  Steps: {1}".format(pdm["distance_minperiod"], periodsteps))
                plt.xlabel(r"Phase ($\phi$)")
                plt.ylabel(f"Calibrated {filterCode} Magnitude")
                plt.savefig(periodPath / f"{variableName}_StringTestPeriodPlot_Calibrated.png")
                plt.clf()
    
                tempPeriodCatOut=[]
                for g in range(len(calibData[:,0])):
                    tempPeriodCatOut.append([(calibData[g,0]/(pdm["distance_minperiod"]) % 1), calibData[g,1], calibData[g,2]])
                tempPeriodCatOut=asarray(tempPeriodCatOut)
                savetxt(periodPath / f"{variableName}_String_PhasedCalibMags.csv", tempPeriodCatOut, delimiter=",", fmt='%0.8f')

        if (varData.size > 3):

            tempPeriodCatOut=[]
            for g in range(len(phaseTest)):
                tempPeriodCatOut.append([phaseTest[g],varData[g,1]])
            tempPeriodCatOut=asarray(tempPeriodCatOut)
            savetxt(periodPath / f"{variableName}_StringTrial.csv", tempPeriodCatOut, delimiter=",", fmt='%0.8f')
    
            tempPeriodCatOut=[]
            for g in range(len(varData[:,0])):
                tempPeriodCatOut.append([(varData[g,0]/(pdm["distance_minperiod"]) % 1), varData[g,1], varData[g,2]])
            tempPeriodCatOut=asarray(tempPeriodCatOut)
            savetxt(periodPath / f"{variableName}_String_PhasedDiffMags.csv", tempPeriodCatOut, delimiter=",", fmt='%0.8f')

        if np.isnan(pdm["stdev_results"][0]) or pdm["stdev_results"][0] == 0.0 or (varData.size < 4):
            logger.info("No PDM results due to lack of datapoint coverage")
        else:
            logger.debug("PDM Method Estimate (days): "+ str(pdm["stdev_minperiod"]))
            phaseTest=(varData[:,0] / (pdm["stdev_minperiod"])) % 1
            logger.debug("PDM method error: " + str(pdm["stdev_error"]))

            with open(paths['parent'] / "periodEstimates.txt", "a+") as f:
                f.write("PDM Method Estimate (days): "+ str(pdm["stdev_minperiod"])+"\n")
                f.write("PDM method error: " + str(pdm["stdev_error"])+"\n\n")


        plt.plot(pdm["periodguess_array"], pdm["stdev_results"])
        plt.gca().invert_yaxis()
        plt.title("Range {0} d  Steps: {1}".format(trialRange, periodsteps))
        plt.xlabel(r"Trial Period")
        plt.ylabel(r"Likelihood of Period")
        plt.savefig(periodPath / f"{variableName}_PDMLikelihoodPlot.png")

        plt.clf()

        if (varData.size > 3):
            plt.plot(phaseTest, varData[:,1], 'bo', linestyle='None')
            plt.plot(phaseTest+1, varData[:,1], 'ro', linestyle='None')
            plt.errorbar(phaseTest, varData[:,1], yerr=varData[:,2], linestyle='None')
            plt.errorbar(phaseTest+1, varData[:,1], yerr=varData[:,2], linestyle='None')
            plt.gca().invert_yaxis()
            plt.title("Period: {0} d  Steps: {1}".format(pdm["stdev_minperiod"], periodsteps))
            plt.xlabel(r"Phase ($\phi$)")
            plt.ylabel(r"Differential " + str(filterCode) + " Magnitude")
            plt.savefig(periodPath / f"{variableName}_PDMTestPeriodPlot.png")
            plt.clf()

        if calibFile.exists():
            if (calibData.size > 3):
                phaseTestCalib=(calibData[:,0] / (pdm["stdev_minperiod"])) % 1
                plt.plot(phaseTestCalib, calibData[:,1], 'bo', linestyle='None')
                plt.plot(phaseTestCalib+1, calibData[:,1], 'ro', linestyle='None')
                plt.errorbar(phaseTestCalib, calibData[:,1], yerr=calibData[:,2], linestyle='None')
                plt.errorbar(phaseTestCalib+1, calibData[:,1], yerr=calibData[:,2], linestyle='None')
                plt.gca().invert_yaxis()
                plt.title("Period: {0} d  Steps: {1}".format(pdm["stdev_minperiod"], periodsteps))
                plt.xlabel(r"Phase ($\phi$)")
                plt.ylabel(r"Calibrated " + str(filterCode) + " Magnitude")
                plt.savefig(periodPath / f"{variableName}_PDMTestPeriodPlot_Calibrated.png")
                plt.clf()
    
                tempPeriodCatOut=[]
                for g in range(len(calibData[:,0])):
                    tempPeriodCatOut.append([(calibData[g,0]/(pdm["stdev_minperiod"])) % 1, calibData[g,1], calibData[g,2]])
                tempPeriodCatOut=asarray(tempPeriodCatOut)
                savetxt(periodPath / f"{variableName}_PDM_PhasedCalibMags.csv", tempPeriodCatOut, delimiter=",", fmt='%0.8f')

        if (varData.size > 3):

            tempPeriodCatOut=[]
            for g in range(len(phaseTest)):
                tempPeriodCatOut.append([phaseTest[g],varData[g,1]])
            tempPeriodCatOut=asarray(tempPeriodCatOut)
            savetxt(periodPath / f"{variableName}_PDMTrial.csv", tempPeriodCatOut, delimiter=",", fmt='%0.8f')
    
            tempPeriodCatOut=[]
            for g in range(len(varData[:,0])):
                tempPeriodCatOut.append([(varData[g,0]/(pdm["stdev_minperiod"])) % 1, varData[g,1], varData[g,2]])
            tempPeriodCatOut=asarray(tempPeriodCatOut)
            savetxt(periodPath / f"{variableName}_PDM_PhaseddiffMags.csv", tempPeriodCatOut, delimiter=",", fmt='%0.8f')

        # Plot publication plots

        plt.figure(figsize=(5, 3))

        plt.plot(pdm["periodguess_array"], pdm["stdev_results"], linewidth=0.5)
        plt.gca().invert_yaxis()
        plt.xlabel(r"Trial Period")
        plt.ylabel(r"Likelihood of Period")
        plt.subplots_adjust(left=0.15, right=0.99, top=0.98, bottom=0.15, wspace=0.3, hspace=0.4)
        plt.savefig(periodPath / f"{variableName}_PDMLikelihoodPlot_Publication.png", dpi=300)
        plt.savefig(periodPath / f"{variableName}_PDMLikelihoodPlot_Publication.eps")

        plt.clf()

        plt.figure(figsize=(5, 3))

        plt.plot(pdm["periodguess_array"], pdm["distance_results"], linewidth=0.5)
        plt.gca().invert_yaxis()
        plt.xlabel(r"Trial Period")
        plt.ylabel(r"Likelihood of Period")
        plt.subplots_adjust(left=0.15, right=0.99, top=0.98, bottom=0.15, wspace=0.3, hspace=0.4)
        plt.savefig(periodPath / f"{variableName}_StringLikelihoodPlot_Publication.png", dpi=300)
        plt.savefig(periodPath / f"{variableName}_StringLikelihoodPlot_Publication.eps")

        plt.clf()


        # ANOVA
        
        if calibFile.exists():
            if (calibData.size > 3):
                if len(calibData[:,0]) < 75:
                    binsize=0.1
                else:
                    binsize=0.05
                minperbin=int((len(calibData[:,0])/10))
        else:
            if (varData.size > 3):
                if len(varData[:,0]) < 75:
                    binsize=0.1
                else:
                    binsize=0.05
                minperbin=int((len(varData[:,0])/10))
        

        
        if minperbin > 10:
            minperbin=10
        
        if calibFile.exists():
            if (calibData.size > 3):
                aovoutput=aov_periodfind((calibData[:,0]),(calibData[:,1]),(calibData[:,2]), sigclip=False, autofreq=False, startp=minperiod, endp=maxperiod, phasebinsize=binsize, mindetperbin=minperbin, periodPath=periodPath, variableName=variableName)
        else:
            if (varData.size > 3):
                aovoutput=aov_periodfind((varData[:,0]),(varData[:,1]),(varData[:,2]), sigclip=False, autofreq=False, startp=minperiod, endp=maxperiod, phasebinsize=binsize, mindetperbin=minperbin, periodPath=periodPath, variableName=variableName)


        logger.debug("Theta Anova Method Estimate (days): " + str(aovoutput["bestperiod"]))
        
        if calibFile.exists():
            if (calibData.size > 3):
                aovhmoutput=aovhm_periodfind((calibData[:,0]),(calibData[:,1]),(calibData[:,2]), sigclip=False, autofreq=False, startp=minperiod, endp=maxperiod, periodPath=periodPath, variableName=variableName)
        else:
            if (varData.size > 3):
                aovhmoutput=aovhm_periodfind((varData[:,0]),(varData[:,1]),(varData[:,2]), sigclip=False, autofreq=False, startp=minperiod, endp=maxperiod, periodPath=periodPath, variableName=variableName)
            
        logger.debug("Harmonic Anova Method Estimate (days): " + str(aovhmoutput["bestperiod"]))


        # LOMB SCARGLE
        for nts in range(6):
            if calibFile.exists():
                if (calibData.size > 3):
                    lscargoutput = LombScargleMultiterm('periodifile', (calibData[:, 0]), (calibData[:, 1]), (calibData[:, 2]),
                                                        nterms=nts+1,
                                                        periodlower=minperiod, periodupper=maxperiod, samples=20,
                                                        disablelightcurve=False, periodPath=periodPath, variableName=variableName)
        
                    logger.debug('Lomb-Scargle N=' + str(nts+1) + ' Period Best Estimate: ' + str(lscargoutput))
            else:
                if (varData.size > 3):
                    lscargoutput = LombScargleMultiterm('periodifile', (varData[:, 0]), (varData[:, 1]), (varData[:, 2]),
                                                        nterms=nts+1,
                                                        periodlower=minperiod, periodupper=maxperiod, samples=20,
                                                        disablelightcurve=False, periodPath=periodPath, variableName=variableName)
        
                    logger.debug('Lomb-Scargle N=' + str(nts+1) + ' Period Best Estimate: ' + str(lscargoutput))

    if 'pdm' in locals():
        return pdm["distance_minperiod"]
    else:
        return 0.0
