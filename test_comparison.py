from astropy.io import fits
import numpy
import os
import glob

from .comparison import find_comparisons

TEST_DATA_PATH = os.environ.get('AUTOVAR_TEST_DATA_PATH','/usr/var/data/autovar')

COMP_DATA_PATH = os.path.join(TEST_DATA_PATH, 'comparison')

def test_files_exist():
    files = os.listdir(COMP_DATA_PATH)
    assert 'usedImages.txt' in files
    assert 'screenedComps.csv' in files
    assert 'targetstars.csv' in files
    # All files are present so we are ready to continue
    outfile = find_comparisons(COMP_DATA_PATH)

    assert outfile == os.path.join(COMP_DATA_PATH,"compsUsed.csv")
