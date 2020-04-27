from astropy.io import fits
import numpy
import os
from pathlib import Path

from astrosource.identify import (rename_data_file, export_photometry_files,
    extract_photometry, gather_files, find_stars)


TEST_PATH_PARENT = Path(os.path.dirname(__file__)) / 'test_files'

TEST_PATHS = {'parent': TEST_PATH_PARENT / 'stars'}

def test_rename_object():
    header = {  "OBJECT"    : "M1",
                "FILTER1"   : "ip",
                "FILTER2"   : "air",
                "FILTER3"   : "air",
                'EXPTIME'   : 20.0,
                'DATE-OBS'      : "2019-01-25T15:54:10.861857",
                'AIRMASS'   : 1.6,
                'INSTRUME'  : 'kb92',
                'MJD-OBS'   : 58508.3265502,
                }
    name = rename_data_file(header)
    exp_name = "M1_ip_58508d3265502000_2019d01d25T15d54d10d861857_1a6_20d0_kb92.npy"
    assert name == exp_name

def test_rename_noobject():
    header = {
                "FILTER1"   : "ip",
                "FILTER2"   : "air",
                "FILTER3"   : "air",
                'EXPTIME'   : 20.0,
                'DATE-OBS'      : "2019-01-25T15:54:10.861857",
                'AIRMASS'   : 1.6,
                'INSTRUME'  : 'kb92',
                'MJD-OBS'   : 58508.3265502,
                }
    name = rename_data_file(header)
    exp_name = "UNKNOWN_ip_58508d3265502000_2019d01d25T15d54d10d861857_1a6_20d0_kb92.npy"
    assert name == exp_name

def test_rename_nomjd():
    header = {  "OBJECT"    : "M1",
                "FILTER1"   : "ip",
                "FILTER2"   : "air",
                "FILTER3"   : "air",
                'EXPTIME'   : 20.0,
                'DATE-OBS'      : "2019-01-25T15:54:10.861857",
                'AIRMASS'   : 1.6,
                'INSTRUME'  : 'kb92',
                'MJD-OBS'   :'UNKNOWN',
                }
    name = rename_data_file(header)
    exp_name = "M1_ip_UNKNOWN_2019d01d25T15d54d10d861857_1a6_20d0_kb92.npy"
    assert name == exp_name

def test_extract_photometry(tmp_path):
    # tmp_path is a Path object for a temporary directory
    infile = TEST_PATHS['parent'] / 'photometry_test.fits'
    result_file = extract_photometry(infile, tmp_path, "test.npy")
    # Test returned file is where we expect it for given filename
    assert result_file == tmp_path / "test.npy"

    result_phot = numpy.load(result_file)
    test_photfile = TEST_PATHS['parent'] / 'photometry_test.csv'
    test_phot = numpy.genfromtxt(test_photfile, dtype=float, delimiter=',')
    # Test if csv file is as we expect
    assert result_phot.all() == test_phot.all()

def test_gather_files():

    phot_files, filtercode = gather_files(TEST_PATHS, filetype="fits")
    test_files = {'XOd2_ip_57757d0532642000_2017d01d04T01d16d43d571_1a089113_22d284_kb29.npy': 'photometry_test2.fits',
    'XOd2_ip_57757d0522793000_2017d01d04T01d15d18d519_1a0899013_22d293_kb29.npy': 'photometry_test.fits'}
    assert phot_files == test_files
    # Clean up
    # for tf in test_files:
    #     os.remove(tf)

def test_find_stars():
    target = [[117.0269708, 50.2258111, 0,0]]
    phot_files, filtercode = gather_files(TEST_PATHS, filetype="fits")
    print(phot_files)
    usedImages = find_stars(target, TEST_PATHS, phot_files)
    images_list = [str(u) for u in usedImages]
    # Check the right files are saved
    test_list = (TEST_PATHS['parent'] / 'usedImages_test.txt').read_text().strip().split('\n')
    assert images_list.sort() == test_list.sort()
    # Clean up
    (TEST_PATHS['parent'] / 'usedImages.txt').unlink()
    test_files = ['XOd2_ip_57757d0522793000_2017d01d04_1a0899013_22d293_kb29.npy',
                  'XOd2_ip_57757d0532642000_2017d01d04_1a089113_22d284_kb29.npy',
                  'screenedComps.csv']
    # for tf in test_files:
    #     (TEST_PATHS['parent'] / tf).unlink()
