from astropy.io import fits
import numpy
import os
import glob

from .identify import rename_data_file, export_photometry_files, extract_photometry, \
                    gather_files, find_stars

TEST_DATA_PATH = os.environ.get('AUTOVAR_TEST_DATA_PATH','/usr/var/data/autovar')


def test_rename_object():
    header = {  "OBJECT"    : "M1",
                "FILTER1"   : "ip",
                "FILTER2"   : "air",
                "FILTER3"   : "air",
                'EXPTIME'   : 20.0,
                'DATE'      : "2019-01-25T15:54:10.861857",
                'AIRMASS'   : 1.6,
                'INSTRUME'  : 'kb92',
                'MJD-OBS'   : 58508.3265502,
                }
    name = rename_data_file(header)
    exp_name = "M1_ip_20d0_2019d01d25T15d54d10d861857_1a6_58508d3265502000_kb92.csv"
    assert name == exp_name

def test_rename_noobject():
    header = {
                "FILTER1"   : "ip",
                "FILTER2"   : "air",
                "FILTER3"   : "air",
                'EXPTIME'   : 20.0,
                'DATE'      : "2019-01-25T15:54:10.861857",
                'AIRMASS'   : 1.6,
                'INSTRUME'  : 'kb92',
                'MJD-OBS'   : 58508.3265502,
                }
    name = rename_data_file(header)
    exp_name = "UNKNOWN_ip_20d0_2019d01d25T15d54d10d861857_1a6_58508d3265502000_kb92.csv"
    assert name == exp_name

def test_rename_nomjd():
    header = {  "OBJECT"    : "M1",
                "FILTER1"   : "ip",
                "FILTER2"   : "air",
                "FILTER3"   : "air",
                'EXPTIME'   : 20.0,
                'DATE'      : "2019-01-25T15:54:10.861857",
                'AIRMASS'   : 1.6,
                'INSTRUME'  : 'kb92',
                'MJD-OBS'   :'UNKNOWN',
                }
    name = rename_data_file(header)
    exp_name = "M1_ip_20d0_2019d01d25T15d54d10d861857_1a6_UNKNOWN_kb92.csv"
    assert name == exp_name

def test_extract_photometry(tmpdir):
    infile = os.path.join(TEST_DATA_PATH,'photometry_test.fits')
    indir = tmpdir.mkdir("autovar")
    result_file = extract_photometry(infile, indir, "test.csv")
    # Test returned file is where we expect it for given filename
    assert result_file == indir.join("test.csv")

    result_phot = numpy.genfromtxt(result_file, dtype=float, delimiter=',')
    test_photfile = os.path.join(TEST_DATA_PATH,'photFile_test.csv')
    test_phot = numpy.genfromtxt(test_photfile, dtype=float, delimiter=',')
    # Test if csv file is as we expect
    assert result_phot.all() == test_phot.all()

def test_gather_files():
    phot_files = gather_files(TEST_DATA_PATH, filetype="fits")
    test_files = [os.path.join(TEST_DATA_PATH,'XOd2_ip_22d293_2017d01d04_1a0899013_57757d0522793000_kb29.csv'), os.path.join(TEST_DATA_PATH, 'XOd2_ip_22d284_2017d01d04_1a089113_57757d0532642000_kb29.csv')]
    assert phot_files.sort() == test_files.sort()
    # Clean up
    for tf in test_files:
        os.remove(tf)

def test_find_stars():
    ra = 117.0269708
    dec = 50.2258111
    used_file = find_stars(TEST_DATA_PATH, ra, dec, filetype='fits')
    # Check the targets are in targetstars.csv
    test_targets = file_to_list(os.path.join(TEST_DATA_PATH, 'targetstars.csv'))
    assert test_targets == ['0.00000000,0.00000000,0.00000000,0.00000000', '117.02697080,50.22581110,0.00000000,0.00000000']
    # Check the right files are saved
    result_list = file_to_list(used_file)
    test_list = file_to_list(os.path.join(TEST_DATA_PATH, 'usedImages_test.txt'))
    assert result_list.sort() == test_list.sort()
    # Clean up
    os.remove(os.path.join(TEST_DATA_PATH,'usedImages.txt'))
    test_files = ['XOd2_ip_22d293_2017d01d04_1a0899013_57757d0522793000_kb29.csv',
                  'XOd2_ip_22d284_2017d01d04_1a089113_57757d0532642000_kb29.csv',
                  'screenedComps.csv',
                  'targetstars.csv']
    for tf in test_files:
        os.remove(os.path.join(TEST_DATA_PATH,tf))

def file_to_list(infile):
    fileList=[]
    used_file = os.path.join(infile)
    with open(used_file, "r") as f:
      for line in f:
        fileList.append(line.strip())
    return fileList
