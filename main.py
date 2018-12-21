import identify
import comparison
import analyse
import click

import logging

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


@click.command()
@click.option('--full', is_flag=True)
@click.option('--stars', is_flag=True)
@click.option('--comparison', is_flag=True)
@click.option('--calc', is_flag=True)
@click.option('--plot', is_flag=True)
@click.option('--ra', required=True, type=float)
@click.option('--dec', required=True, type=float)
def main(full, stars, comparison, calc, plot, ra, dec):
    if full or stars:
        identify.find_stars(ra, dec)
    if full or comparison:
        comparison.find_comparisons()
    if full or calc:
        analyse.calculate_curves()
    if full or plot:
        analyse.plot_lightcurves()
    return

if __name__ == '__main__':
    main()
