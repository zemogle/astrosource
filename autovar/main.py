from pathlib import Path
import click
import sys
import logging

from numpy import array

from autovar.identify import find_stars
from autovar.comparison import find_comparisons
from autovar.analyse import calculate_curves, photometric_calculations
from autovar.plots import make_plots, calibrated_plots
from autovar.eebls import plot_bls

from autovar.utils import get_targets, folder_setup, AutovarException, cleanup


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option('--full', is_flag=True)
@click.option('--stars', is_flag=True)
@click.option('--comparison', is_flag=True)
@click.option('--calc', is_flag=True)
@click.option('--calib', is_flag=True)
@click.option('--phot', is_flag=True)
@click.option('--plot', is_flag=True)
@click.option('--eebls', is_flag=True)
@click.option('--indir', default=None, type=str, required=True)
@click.option('--ra', type=float)
@click.option('--dec', type=float)
@click.option('--target-file', default=None, type=str)
@click.option('--format', default='fz', type=str)
@click.option('--clean', is_flag=True)
def main(full, stars, comparison, calc, calib, phot, plot, eebls, indir, ra, dec, target_file, format, clean):
    try:
        parentPath = Path(indir)
        if clean:
            cleanup(parentPath)
            logger.info('All temporary files removed')
            return
        if not (ra and dec) and not target_file:
            logger.error("Either RA and Dec or a targetfile must be specified")
            return

        paths = folder_setup(parentPath)
        if ra and dec:
            targets = array([(ra,dec,0,0)])
        elif target_file:
            target_file = parentPath / target_file
            targets = get_targets(target_file)

        # sys.tracebacklimit = 0

        if full or stars:
            find_stars(targets, parentPath, filetype=format)
        if full or comparison:
            find_comparisons(parentPath)
        if full or calc:
            calculate_curves(targets, parentPath=parentPath)
        if full or phot:
            photometric_calculations(targets, paths=paths)
        if full or plot:
            make_plots(filterCode='r', paths=paths)
        if full or eebls:
            plot_bls(paths=paths)
        if full or calib:
            calibrated_plots(filterCode='r', paths=paths)

    except AutovarException as e:
        logger.critical(e)

if __name__ == '__main__':
    main()
