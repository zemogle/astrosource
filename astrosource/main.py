from pathlib import Path
import click
import sys
import logging

from numpy import array, all
from astropy.utils.exceptions import AstropyWarning, AstropyDeprecationWarning

from astrosource.astrosource import TimeSeries

from astrosource.utils import get_targets, folder_setup, AstrosourceException, cleanup, convert_coords

logger = logging.getLogger('astrosource')

@click.command()
@click.option('--full', is_flag=True, help='Perform full analysis, i.e. steps 1-5')
@click.option('--stars', is_flag=True, help='Step 1: Identify and match stars from each data file')
@click.option('--comparison', is_flag=True, help='Step 2: Identify non-varying stars to use for comparisons')
@click.option('--calc', is_flag=True, help='Step 3: Calculate the brightness change of the target')
@click.option('--phot', is_flag=True, help='Step 4: Photometry calculations for either differential or calibrated')
@click.option('--plot', is_flag=True, help='Step 5: Produce lightcurve plots')

@click.option('--indir', default=None, type=str, required=True, help='Path of directory containing LCO data files. If none is given, astrosource assumes the current directory')
@click.option('--ra', type=str, help='Right Ascension of the target (in decimal or H:M:S)')
@click.option('--dec', type=str, help='Declination of the target (in decimal or D:M:S)')

@click.option('--target-file', default=None, type=str, help='Textfile in csv with RA and Dec in decimal')
@click.option('--format', default='fz', type=str, help='Input file format. If not `fz`, `fits`, or `fit` assumes the input files are photometry files with correct headers. If image files given, code looks for photometry in FITS Table extension.')

@click.option('--verbose', '-v', is_flag=True, help='Show all system messages for AstroSource')
@click.option('--clean', is_flag=True, help='Remove all generated files. Reset `indir` to initial state')

@click.option('--usescreenedcomps', is_flag=True, help='Use screenedComps.csv as the candidate comparison list rather than detecting them')
@click.option('--usecompsused', is_flag=True, help='Use compsUsed.csv as the final candidate comparison list rather than calculating them')

@click.option('--calib', is_flag=True, help='Perform calibrations for absolute photometry')
@click.option('--usecompletedcalib', is_flag=True, help='Use the calibration already calculated')
@click.option('--calibsave', is_flag=True, help='Save Calibration Files for each image')

@click.option('--detrend', is_flag=True, help='Detrend exoplanet data')
@click.option('--detrendfraction', type=float, default=0.1, help='Fraction of data range from the edges to use to detrend data')

@click.option('--eebls', is_flag=True, help='Box fitting to search for periodic transits')

@click.option('--racut', type=float, default=-99.9, help='Right Ascension of the centre of the aperture cutout (in decimal or H:M:S)')
@click.option('--deccut', type=float, default=-99.9, help='Declination of the centre of the aperture cutout (in decimal or D:M:S)')
@click.option('--radiuscut', type=float, default=-99.9, help='Radius centre of the aperture cutout (in arcminutes')

@click.option('--imgreject', '-ir', type=float, default=0.05, help=' Image fraction rejection allowance based on image size starting value.')
@click.option('--bjd', is_flag=True, help='Convert the MJD time into BJD time for LCO fz images')


@click.option('--period', is_flag=True, type=float, help='Search for periodicity in the data, currently with PDM and String methods. This will autoselect a reasonable search range if not provided a range.')
@click.option('--periodlower', '-pl', type=float, default=-99.9, help='Shortest period to trial in days. Default is 0.05 days')
@click.option('--periodupper', '-pu', type=float, default=-99.9, help='Longest period to trial in days. Default is one-third the observational baseline of your dataset.')
@click.option('--periodtests', '-pt', type=int, default=10000, help='Number of different trial periods to run')
#@click.option('--rejectbrighter', '-rb', type=float, default=99, help='')
#@click.option('--rejectdimmer', '-rd', type=float, default=99, help='')
@click.option('--thresholdcounts', type=int, default=1000000, help='number of counts at which to stop adding identified comparison stars to the ensemble')
@click.option('--hicounts', type=int, default=3000000, help='Count rate above which to reject a comparison star as a candidate')
@click.option('--lowcounts', type=int, default=5000, help='Count rate above which to accept a comparison star as a candidate')

@click.option('--lowestcounts', type=int, default=1800, help='Lowest Count rate to consider any star for anything (removes these stars entirely from photfiles)')

@click.option('--starreject', type=float, default=0.3, help='Limit on fraction of stars which can be rejected without image being rejected starting value')
@click.option('--closerejectd',  type=float, default=5.0, help='Limiting distance (arcsecs) between potential comparisons and nearby stars')
@click.option('--targetradius',  type=float, default=1.5, help='Limiting distance (arcsecs) away from requested target star position.')
@click.option('--matchradius',  type=float, default=1.0, help='Limiting distance (arcsecs) when matching general stars, usually comparisons.')
@click.option('--mincompstars', type=float, default=0.1, help='Minimum number of candidate comparison stars to identify as a fraction of detected stars')
@click.option('--mincompstarstotal', type=float, default=-99, help='Minimum comparison stars as a given number')
@click.option('--maxcandidatestars', type=float, default=10000, help='Maximum number of candidate stars to begin search with')
@click.option('--nopanstarrs',  is_flag=True, help='Do not use the PanSTARRS catalogue for calibration')
@click.option('--nosdss',  is_flag=True, help='Do not use the SDSS catalogue for calibration')
@click.option('--varsearch',  is_flag=True, help='Undertake variability calculations for identified stars')
@click.option('--varsearchthresh',  type=float, default=10000, help='Threshold counts above which to detect variability')
@click.option('--varsearchstdev',  type=float, default=1.5, help='How many stdev above mean of each bin to detect variables')
@click.option('--varsearchmagwidth',  type=float, default=0.5, help='Size of magnitude bin to detect variables')
@click.option('--varsearchminimages', type=float, default=0.3, help='Fraction of number of images a variable star has to be in to be detected')
@click.option('--ignoreedgefraction', type=float, default=0.05, help='Ignore stars that are within this fraction of the edge of the RA and DEC range')

@click.option('--outliererror',  type=float, default=4, help='Ignore measurements that have an error this many standard deviatons above the mean')
@click.option('--outlierstdev', type=float, default=4, help='Ignore measurements that have a magnitude this many standard deviatons outside the mean')

@click.option('--variablehunt', is_flag=True, help='Do a variable search in astrosource and then use the potentialVariables as a targetlist')
@click.option('--usepreviousvarsearch', is_flag=True, help='Do not repeat the variability measurements for all stars, use the previously calculated ones')

@click.option('--notarget',  is_flag=True, help='Do not provide a target and use astrosource for other means')

@click.option('--colourdetect',  is_flag=True, help='Run a routine to detect the colour term for the filter')
@click.option('--linearise',  is_flag=True, help='Use the calibration to make the measurements linear across the magnitude range (for slopes > 0.01 or 0.02 in mag)')
@click.option('--colourterm',  type=float, default=0.0, help='Provide a known colour term for the filter')
@click.option('--colourerror',  type=float, default=0.0, help='Provide a known colour error for the filter')
@click.option('--targetcolour',  type=float, default=-99.0, help='Provide a known colour for the target')


@click.option('--restrictcompcolourcentre', type=float, default=-99.0, help='The middle of the range of comparison colours to restrict')
@click.option('--restrictcompcolourrange', type=float, default=-99.0, help='The range of the comparison colours to restrict')

@click.option('--restrictmagbrightest', type=float, default=-99.0, help='Do not use catalogue magnitudes brighter than this for calibration')
@click.option('--restrictmagdimmest', type=float, default=99.0, help='Do not use catalogue magnitudes dimmer than this for calibration')
@click.option('--rejectmagbrightest', type=float, default=-99.0, help='Remove calibrated measurements brighter than this from target results')
@click.option('--rejectmagdimmest', type=float, default=99.0, help='Remove calibrated measurements dimmer than this from target results')

def main(full, stars, comparison, variablehunt, notarget, lowestcounts, usescreenedcomps, usepreviousvarsearch, calibsave, outliererror, outlierstdev, varsearchstdev, varsearchmagwidth, varsearchminimages, ignoreedgefraction, usecompsused, usecompletedcalib, mincompstarstotal, calc, calib, phot, plot, detrend, eebls, period, indir, ra, dec, target_file, format, imgreject, mincompstars, maxcandidatestars, closerejectd, bjd, clean, verbose, periodlower, periodupper, periodtests,  thresholdcounts, nopanstarrs, nosdss, varsearch, varsearchthresh, starreject, hicounts, lowcounts, colourdetect, linearise, colourterm, colourerror, targetcolour, restrictmagbrightest, restrictmagdimmest, rejectmagbrightest, rejectmagdimmest,targetradius, matchradius, racut, deccut, radiuscut, restrictcompcolourcentre, restrictcompcolourrange, detrendfraction):

    try:
        parentPath = Path(indir)
        if clean:
            cleanup(parentPath)
            logger.info('All output files removed')
            return
        if not (ra and dec) and not target_file and not variablehunt:
            #logger.error("Either RA and Dec or a targetfile must be specified")
            logger.error("No specified RA or Dec nor targetfile nor request for a variable hunt. It is assumed you have no target to analyse.")
            notarget=True
            #return



        if ra and dec:
            ra, dec = convert_coords(ra, dec)
            targets = array([(ra, dec, 0, 0)])
        elif target_file:
            target_file = parentPath / target_file
            targets = get_targets(target_file)
        elif notarget == True or variablehunt == True:
            targets = None
            
        if variablehunt == True:
            varsearch=True
            full=True

        if usecompletedcalib == True:
            usecompsused = True

        if usecompsused == True:
            usescreenedcomps = True

        ts = TimeSeries(indir=parentPath,
                        targets=targets,
                        format=format,
                        imgreject=imgreject,
                        periodupper=periodupper,
                        periodlower=periodlower,
                        periodtests=periodtests,                        
                        thresholdcounts=thresholdcounts,
                        lowcounts=lowcounts,
                        hicounts=hicounts,
                        starreject=starreject,
                        nopanstarrs=nopanstarrs,
                        nosdss=nosdss,
                        closerejectd=closerejectd,
                        maxcandidatestars=maxcandidatestars,
                        verbose=verbose,
                        bjd=bjd,
                        mincompstars=mincompstars,
                        mincompstarstotal=mincompstarstotal,
                        colourdetect=colourdetect,
                        linearise=linearise,
                        variablehunt=variablehunt,
                        calibsave=calibsave,
                        notarget=notarget,
                        usescreenedcomps=usescreenedcomps,
                        usepreviousvarsearch=usepreviousvarsearch,
                        colourterm=colourterm,
                        colourerror=colourerror,
                        targetcolour=targetcolour,
                        restrictmagbrightest=restrictmagbrightest,
                        restrictmagdimmest=restrictmagdimmest,
                        rejectmagbrightest=rejectmagbrightest,
                        rejectmagdimmest=rejectmagdimmest,
                        targetradius=targetradius,
                        matchradius=matchradius,
                        varsearchthresh=varsearchthresh,
                        varsearchstdev=varsearchstdev, 
                        varsearchmagwidth=varsearchmagwidth,
                        varsearchminimages=varsearchminimages,
                        ignoreedgefraction=ignoreedgefraction,
                        outliererror=outliererror,
                        outlierstdev=outlierstdev,
                        lowestcounts=lowestcounts,
                        racut=racut,
                        deccut=deccut,
                        radiuscut=radiuscut,
                        restrictcompcolourcentre=restrictcompcolourcentre,
                        restrictcompcolourrange=restrictcompcolourrange,
                        detrendfraction=detrendfraction
                        )

        if full or comparison:
            ts.analyse(usescreenedcomps=usescreenedcomps, usecompsused=usecompsused, usecompletedcalib=usecompletedcalib)
        if (full or calc) and varsearch and not usepreviousvarsearch:
            ts.find_variables()

        if variablehunt == True:
            targets = get_targets(parentPath / 'potentialVariables.csv')
        

        
        if targets is not None:
            if full or phot:
                ts.photometry(filesave=True, targets=targets)
            if full or plot:
                ts.plot(detrend=detrend, period=period, eebls=eebls, filesave=True)

        sys.stdout.write("✅ AstroSource analysis complete\n")

    except AstrosourceException as e:
        logger.critical(e)
    return


if __name__ == '__main__':
    main()
