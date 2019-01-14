from identify import find_stars
from comparison import find_comparisons
from analyse import calculate_curves, plot_lightcurves
import click

import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


@click.command()
@click.option('--full', is_flag=True)
@click.option('--stars', is_flag=True)
@click.option('--comparison', is_flag=True)
@click.option('--calc', is_flag=True)
@click.option('--plot', is_flag=True)
@click.option('--indir', type=str)
@click.option('--ra', required=True, type=float)
@click.option('--dec', required=True, type=float)
def main(full, stars, comparison, calc, plot, indir, ra, dec):
    if full or stars:
        find_stars(indir, ra, dec)
    if full or comparison:
        find_comparisons(indir)
    if full or calc:
        calculate_curves(indir)
    if full or plot:
        plot_lightcurves(indir)
    return

if __name__ == '__main__':
    main()
