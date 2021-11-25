from pathlib import Path
import click
import sys
import logging

from numpy import array
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
@click.option('--calib', is_flag=True, help='Perform calibrations for absolute photometry')
@click.option('--detrend', is_flag=True, help='Detrend exoplanet data')
@click.option('--eebls', is_flag=True, help='Box fitting to search for periodic transits')
@click.option('--indir', default=None, type=str, required=True, help='Path of directory containing LCO data files. If none is given, astrosource assumes the current directory')
@click.option('--ra', type=str, help='Right Ascension of the target (in decimal or H:M:S)')
@click.option('--dec', type=str, help='Declination of the target (in decimal or D:M:S)')
@click.option('--target-file', default=None, type=str, help='Textfile in csv with RA and Dec in decimal')
@click.option('--format', default='fz', type=str, help='Input file format. If not `fz`, `fits`, or `fit` assumes the input files are photometry files with correct headers. If image files given, code looks for photometry in FITS Table extension.')
@click.option('--imgreject', '-ir', type=float, default=0.05, help=' Image fraction rejection allowance based on image size starting value.')
@click.option('--bjd', is_flag=True, help='Convert the MJD time into BJD time for LCO images')
@click.option('--clean', is_flag=True, help='Remove all generated files. Reset `indir` to initial state')
@click.option('--verbose', '-v', is_flag=True, help='Show all system messages for AstroSource')
@click.option('--period', is_flag=True, type=float, help='Search for periodicity in the data, currently with PDM and String methods. This will autoselect a reasonable search range if not provided a range.')
@click.option('--periodlower', '-pl', type=float, default=0.05, help='Shortest period to trial in days.')
@click.option('--periodupper', '-pu', type=float, help='Longest period to trial in days. Default is one-third the observational baseline of your dataset.')
@click.option('--periodtests', '-pt', type=int, default=10000, help='Number of different trial periods to run')
@click.option('--rejectbrighter', '-rb', type=float, default=99, help='')
@click.option('--rejectdimmer', '-rd', type=float, default=99, help='')
@click.option('--thresholdcounts', '-tc', type=int, default=1000000, help='number of counts at which to stop adding identified comparison stars to the ensemble')
@click.option('--hicounts', '-hc', type=int, default=1500000, help='Count rate above which to reject a comparison star as a candidate')
@click.option('--lowcounts',  '-lc', type=int, default=1000, help='Count rate above which to accept a comparison star as a candidate')
@click.option('--starreject', '-sr', type=float, default=0.3, help='Limit on fraction of stars which can be rejected without image being rejected')
@click.option('--closerejectd', '-cr', type=float, default=5.0, help='Limiting distance (arcsecs) between potential comparisons and nearby stars')
@click.option('--mincompstars', '-mc', type=float, default=0.1, help='Minimum comparison stars')
@click.option('--nopanstarrs', '-np', is_flag=True, help='Do not use the PanSTARRS catalogue for calibration')
@click.option('--nosdss', '-ns', is_flag=True, help='Do not use the SDSS catalogue for calibration')
@click.option('--skipvarsearch', '-sv', is_flag=True, help='Skip variability calculations for identified stars')
@click.option('--colourdetect', '-cc', is_flag=True)
@click.option('--linearise', '-cc', is_flag=True)
@click.option('--colourterm', '-ct', type=float, default=0.0)
@click.option('--colourerror', '-ce', type=float, default=0.0)
@click.option('--targetcolour', '-tc', type=float, default=-99.0)
@click.option('--restrictmagbrightest', type=float, default=-99.0)
@click.option('--restrictmagdimmest', type=float, default=99.0)
def main(full, stars, comparison, calc, calib, phot, plot, detrend, eebls, period, indir, ra, dec, target_file, format, imgreject, mincompstars, closerejectd, bjd, clean, verbose, periodlower, periodupper, periodtests, rejectbrighter, rejectdimmer, thresholdcounts, nopanstarrs, nosdss, skipvarsearch, starreject, hicounts, lowcounts, colourdetect, linearise, colourterm, colourerror, targetcolour, restrictmagbrightest, restrictmagdimmest):

    try:
        parentPath = Path(indir)
        if clean:
            cleanup(parentPath)
            logger.info('All output files removed')
            return
        if not (ra and dec) and not target_file:
            logger.error("Either RA and Dec or a targetfile must be specified")
            return

        if ra and dec:
            ra, dec = convert_coords(ra, dec)
            targets = array([(ra, dec, 0, 0)])
        elif target_file:
            target_file = parentPath / target_file
            targets = get_targets(target_file)

        ts = TimeSeries(indir=parentPath,
                        targets=targets,
                        format=format,
                        imgreject=imgreject,
                        periodupper=periodupper,
                        periodlower=periodlower,
                        periodtests=periodtests,
                        rejectbrighter=rejectbrighter,
                        rejectdimmer=rejectdimmer,
                        thresholdcounts=thresholdcounts,
                        lowcounts=lowcounts,
                        hicounts=hicounts,
                        starreject=starreject,
                        nopanstarrs=nopanstarrs,
                        nosdss=nosdss,
                        closerejectd=closerejectd,
                        verbose=verbose,
                        bjd=bjd,
                        mincompstars=mincompstars,
                        colourdetect=colourdetect,
                        linearise=linearise,
                        colourterm=colourterm,
                        colourerror=colourerror,
                        targetcolour=targetcolour,
                        restrictmagbrightest=restrictmagbrightest,
                        restrictmagdimmest=restrictmagdimmest
                        )

        if full or comparison:
            ts.analyse()
        if (full or calc) and not skipvarsearch:
            ts.find_variables()
        if full or phot:
            ts.photometry(filesave=True)
        if full or plot:
            ts.plot(detrend=detrend, period=period, eebls=eebls, filesave=True)

        sys.stdout.write("✅ AstroSource analysis complete\n")

    except AstrosourceException as e:
        logger.critical(e)
    return


if __name__ == '__main__':
    main()
