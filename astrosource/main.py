from pathlib import Path
import click
import sys
import logging

from numpy import array
from astropy.utils.exceptions import AstropyWarning, AstropyDeprecationWarning

from astrosource.astrosource import TimeSeries

from astrosource.utils import get_targets, folder_setup, AstrosourceException, cleanup

logger = logging.getLogger('astrosource')

@click.command()
@click.option('--full', is_flag=True)
@click.option('--stars', is_flag=True)
@click.option('--comparison', is_flag=True)
@click.option('--calc', is_flag=True)
@click.option('--calib', is_flag=True)
@click.option('--phot', is_flag=True)
@click.option('--plot', is_flag=True)
@click.option('--detrend', is_flag=True)
@click.option('--eebls', is_flag=True)
@click.option('--period', is_flag=True, type=float)
@click.option('--indir', default=None, type=str, required=True)
@click.option('--ra', type=float)
@click.option('--dec', type=float)
@click.option('--target-file', default=None, type=str)
@click.option('--format', default='fz', type=str)
@click.option('--imgreject', '-ir', type=float, default=0.05)
@click.option('--bjd', is_flag=True)
@click.option('--clean', is_flag=True)
@click.option('--verbose', '-v', is_flag=True)
@click.option('--periodlower', '-pl', type=float, default=-99.9)
@click.option('--periodupper', '-pu', type=float, default=-99.9)
@click.option('--periodtests', '-pt', type=int, default=-99)
@click.option('--rejectbrighter', '-rb', type=float, default=99)
@click.option('--rejectdimmer', '-rd', type=float, default=99)
@click.option('--thresholdcounts', '-tc', type=int, default=1000000)
@click.option('--hicounts', '-hc', type=int, default=1500000)
@click.option('--lowcounts',  '-lc', type=int, default=1000)
@click.option('--starreject', '-sr', type=float, default=0.3)
@click.option('--closerejectd', '-sr', type=float, default=5.0)
@click.option('--nopanstarrs', '-np', is_flag=True)
@click.option('--mincompstars', '-mc', type=float, default=0.1)
@click.option('--nosdss', '-ns', is_flag=True)
@click.option('--skipvarsearch', '-sv', is_flag=True)
def main(full, stars, comparison, calc, calib, phot, plot, detrend, eebls, period, indir, ra, dec, target_file, format, imgreject, mincompstars, closerejectd, bjd, clean, verbose, periodlower, periodupper, periodtests, rejectbrighter, rejectdimmer, thresholdcounts, nopanstarrs, nosdss, skipvarsearch, starreject, hicounts, lowcounts):

    try:
        parentPath = Path(indir)
        if clean:
            cleanup(parentPath)
            logger.info('All temporary files removed')
            return
        if not (ra and dec) and not target_file:
            logger.error("Either RA and Dec or a targetfile must be specified")
            return

        if ra and dec:
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
                        mincompstars=mincompstars
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
