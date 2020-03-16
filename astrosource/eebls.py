import sys
import os
import logging
import matplotlib.pyplot as plt
from numpy import median, zeros, nan, nanmedian, sqrt, mean, std, loadtxt, linspace, \
    zeros_like, divide, asarray
from astropy.constants import G, R_sun, M_sun, R_jup, M_jup, R_earth, M_earth
from astropy.coordinates import SkyCoord
from pathlib import Path

from astrosource.utils import AstrosourceException

logger = logging.getLogger('astrosource')

def bls(t, x, qmi, qma, fmin, df, nf, nb, startPeriod, dp):
    """First trial, BLS algorithm, only minor modification from author's code
     Output parameters:
     ~~~~~~~~~~~~~~~~~~
     p    = array {p(i)}, containing the values of the BLS spectrum
            at the i-th frequency value -- the frequency values are
            computed as  f = fmin + (i-1)*df
     bper = period at the highest peak in the frequency spectrum
     bpow = value of {p(i)} at the highest peak
     depth= depth of the transit at   *bper*
     qtran= fractional transit length  [ T_transit/bper ]
     in1  = bin index at the start of the transit [ 0 < in1 < nb+1 ]
     in2  = bin index at the end   of the transit [ 0 < in2 < nb+1 ]
     Remarks:
     ~~~~~~~~
     -- *fmin* MUST be greater than  *1/total time span*
     -- *nb*   MUST be lower than  *nbmax*
     -- Dimensions of arrays {y(i)} and {ibi(i)} MUST be greater than
        or equal to  *nbmax*.
     -- The lowest number of points allowed in a single bin is equal
        to   MAX(minbin,qmi*N),  where   *qmi*  is the minimum transit
        length/trial period,   *N*  is the total number of data points,
        *minbin*  is the preset minimum number of the data points per
        bin.
    """
    n = len(t)
    rn = len(x)
    #! use try
    if n != rn:
        raise AstrosourceException("Different size of array, t and x")
    rn = float(rn) # float of n
    minbin = 5
    nbmax = 2000
    if nb > nbmax:
        raise AstrosourceException("Error: NB > NBMAX!")
    tot = t[-1] - t[0] # total time span
    if fmin < 1.0/tot:
        raise AstrosourceException("Error: fmin < 1/T")
    # parameters in binning (after folding)
    kmi = int(qmi*nb) # nb is number of bin -> a single period
    if kmi < 1:
        kmi = 1
    kma = int(qma*nb) + 1
    kkmi = rn*qmi # to check the bin size
    if kkmi < minbin:
        kkmi = minbin
    # For the extension of arrays (edge effect: transit happen at the edge of data set)
    nb1 = nb + 1
    nbkma = nb + kma
    # Data centering
    t1 = t[0]
    u = t - t1
    s = median(x) # ! Modified
    v = x - s
    bpow = 0.0
    p = zeros(nf)
    # setup array for power vs period plot
    powerPeriod=[]
    # Start period search
    for jf in range(nf):
        #f0 = fmin + df*jf # iteration in frequency not period
        #p0 = 1.0/f0
        # Actually iterate in period
        p0 = startPeriod + dp*jf
        f0 = 1.0/p0
        # Compute folded time series with p0 period
        ibi = zeros(nbkma)
        y = zeros(nbkma)
        # Median version
        yMedian = zeros(shape=(nf,n))
        yMedian.fill(nan)
        for i in range(n):
            ph = u[i]*f0 # instead of t mod P, he use t*f then calculate the phase (less computation)
            ph = ph - int(ph)
            j = int(nb*ph) # data to a bin
            ibi[j] = ibi[j] + 1 # number of data in a bin
            y[j] = y[j] + v[i] # sum of light in a bin
            yMedian[j][i]=v[i]
        # Repopulate y[j] and ibi[j] with the median value
        for i in range(nb+1):
            #logger.debug(i)
            ibi[i]=1
            y[i]=nanmedian(yMedian[i,:])
        # Extend the arrays  ibi()  and  y() beyond nb by wrapping
        for j in range(nb1, nbkma):
            jnb = j - nb
            ibi[j] = ibi[jnb]
            y[j] = y[jnb]
        # Compute BLS statictics for this trial period
        power = 0.0
        for i in range(nb): # shift the test period
            s = 0.0
            k = 0
            kk = 0
            nb2 = i + kma
            # change the size of test period (from kmi to kma)
            for j in range(i, nb2):
                k = k + 1
                kk = kk + ibi[j]
                s = s + y[j]
                if k < kmi: continue # only calculate SR for test period > kmi
                if kk < kkmi: continue #
                rn1 = float(kk)
                powo = s*s/(rn1*(rn - rn1))
                if powo > power: # save maximum SR in a test period
                    power = powo # SR value
                    jn1 = i #
                    jn2 = j
                    rn3 = rn1
                    s3 = s
        power = sqrt(power)
        p[jf] = power
        powerPeriod.append([p0,power])
        if power > bpow:
            # If it isn't an resonance of a day
            if not ((p0 > 0.95 and p0 < 1.05) or (p0 > 1.95 and p0 < 2.05) or (p0 > 2.98 and p0 < 3.02) or (p0 > 6.65 and p0 < 6.67) or (p0 > 3.32 and p0 < 3.34) or (p0 > 3.64 and p0 < 3.68)):
                bpow = power # Save the absolute maximum of SR
                in1 = jn1
                in2 = jn2
                qtran = rn3/rn
                # depth = -s3*rn/(rn3*(rn - rn3))
                # ! Modified
                high = -s3/(rn - rn3)
                low = s3/rn3
                depth = high - low
                bper = p0

    sde = (bpow - mean(p))/std(p) # signal detection efficiency
    return bpow, in1, in2, qtran, depth, bper, sde, p, high, low, powerPeriod

def plot_bls(paths, startPeriod=0.1, endPeriod=3.0, nf=1000, nb=200, qmi=0.01, qma=0.1):
    '''
     Input parameters:
     ~~~~~~~~~~~~~~~~~
     n    = number of data points
     t    = array {t(i)}, containing the time values of the time series
     x    = array {x(i)}, containing the data values of the time series
     u    = temporal/work/dummy array, must be dimensioned in the
            calling program in the same way as  {t(i)}
     v    = the same as  {u(i)}
     nf   = number of frequency points in which the spectrum is computed
     fmin = minimum frequency (MUST be > 0)
     df   = frequency step
     nb   = number of bins in the folded time series at any test period
     qmi  = minimum fractional transit length to be tested
     qma  = maximum fractional transit length to be tested
     paths = dict of Path objects
    '''
    # Get list of phot files
    trimPath = paths['parent'] / "trimcats"
    eelbsPath = paths['parent'] / "eelbs"
    # check directory structure
    if not trimPath.exists():
        os.makedirs(trimPath)
    if not eelbsPath.exists():
        os.makedirs(eelbsPath)
    fileList = paths['outcatPath'].glob('*diffExcel*csv')
    r=0
    # calculate period range
    fmin = 1/endPeriod
    fmax = 1/startPeriod
    df = (fmax-fmin)/nf
    dp = (endPeriod-startPeriod)/nf
    for filename in fileList:
        photFile = loadtxt(paths['outcatPath'] / Path(filename).name, delimiter=',')
        logger.debug('**********************')
        logger.debug(f'Testing: {filename}')
        t = photFile[:,0]
        f = photFile[:,1]
        res = bls(t, f, qmi, qma, fmin, df, nf, nb, startPeriod, dp)
        if not res:
            raise AstrosourceException("BLS fit failed")
        else: # If it did not fail, then do the rest.
            logger.debug(f'Best SR: {res[0]}')
            logger.debug(f'Ingress: {res[1]}')
            logger.debug(f'Egress: {res[2]}')
            logger.debug(f'q: {res[3]}')
            logger.debug(f'Depth: {res[4]}')
            logger.debug(f'Period: {res[5]}')
            logger.debug(f'SDE: {res[6]}')

            t1 = t[0]
            u = t - t1
            s = mean(f)
            v = f - s
            f0 = 1.0/res[5] #  freq = 1/T
            nbin = nb # number of bin
            n = len(t)
            ibi = zeros(nbin)
            y = zeros(nbin)
            phase = linspace(0.0, 1.0, nbin)
            for i in range(n):
                ph = u[i]*f0
                ph = ph - int(ph)
                j = int(nbin*ph) # data to a bin
                ibi[j] += 1.0 # number of data in a bin
                y[j] = y[j] + v[i] # sum of light in a bin

            plt.figure(figsize=(15,6))

            powerPeriod=asarray(res[10])
            plt.subplot(1, 2, 1)
            plt.plot(powerPeriod[:,0], powerPeriod[:,1], 'r.')

            plt.title("EELBS Period Trials")
            plt.xlabel(r"Trialled Period")
            plt.ylabel(r"Likelihood")

            plt.subplot(1, 2, 2)
            plt.plot(phase, divide(y, ibi, out=zeros_like(y), where=ibi!=0), 'r.')
            fite = zeros(nbin) + res[8] # H
            fite[res[1]:res[2]+1] = res[9] # L
            plt.plot(phase, fite)
            plt.gca().invert_yaxis()
            plt.title("\nDepth: "+ str(-res[4]) + "     " + "Period: {0} d  bin: {1}".format(1/f0, nbin))
            plt.xlabel(r"Phase ($\phi$)")
            plt.ylabel(r"Mean value of $x(\phi)$ in a bin")
            plt.tight_layout()
            filebase = str(filename).split("/")[-1].split("\\")[-1].replace(".csv","").replace("_calibExcel","")
            plot_filename = "{}_EELBS_Plot.png".format(filebase)
            plt.savefig(eelbsPath / plot_filename)

            logger.info("Saved {}".format(plot_filename))
            plt.clf()
            # Write text file
            texFileName=eelbsPath / '{}_EELBS_Statistics.txt'.format(filebase)
            logger.info("Saved {}".format(texFileName))
            with open(texFileName, "w") as f:
                f.write("Best SR: " +str(res[0])+"\n")
                f.write("Ingress: " + str(res[1])+"\n")
                f.write("Egress: "+ str(res[2])+"\n")
                f.write("nq: "+ str(res[3])+"\n")
                f.write("Depth: "+ str(-res[4])+"\n")
                f.write("Period: "+ str(res[5])+"\n")
                f.write("SDE: "+ str(res[6])+"\n")
    return
