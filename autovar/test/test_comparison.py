from astropy.io import fits
import numpy
import os
from pathlib import Path

from autovar.comparison import find_comparisons, read_data_files, find_reference_frame


TEST_PATH_PARENT = Path(os.path.dirname(__file__)) / 'test_files'

TEST_PATHS = {'parent': TEST_PATH_PARENT / 'comparison'}


def test_setup():
    used_files = TEST_PATHS['parent'] / 'usedImages.txt'
    if used_files.exists():
        used_files.unlink()
    files = TEST_PATHS['parent'].glob('*.psx')
    with used_files.open(mode='w') as fid:
        for f in files:
            fid.write("{}\n".format(f))

# def test_ensemble():
#     fileCount = [ 2797858.97, 3020751.97, 3111426.77, 3115947.86]

def test_read_data_files():
    files = os.listdir(TEST_PATHS['parent'])
    assert 'screenedComps.csv' in files
    assert 'targetstars.csv' in files
    compFile, photFileArray, fileList = read_data_files(TEST_PATHS['parent'])
    referenceFrame, fileRaDec = find_reference_frame(photFileArray)
    assert list(referenceFrame[0]) == [154.7583434, -9.6660181000000005, 271.47230000000002, 23.331099999999999, 86656.100000000006, 319.22829999999999]
    assert (fileRaDec[0].ra.degree, fileRaDec[0].dec.degree) == ( 154.7583434,  -9.6660181)
    assert len(referenceFrame) == 227
    assert len(fileRaDec) == 227

def test_comparison():
    # All files are present so we are ready to continue
    outfile, num_cands = find_comparisons(TEST_PATHS['parent'])

    assert outfile == TEST_PATHS['parent'] / "compsUsed.csv"
    assert num_cands == 11
