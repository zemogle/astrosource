from astropy.io import fits
import numpy
import os
from pathlib import Path
from mock import patch

from astrosource.comparison import find_comparisons, read_data_files, find_reference_frame, \
    remove_targets, find_comparisons_calibrated

from astrosource.test.mocks import mock_vizier_query_region_vsx, mock_vizier_query_region_apass_b,\
    mock_vizier_query_region_apass_v


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

@patch('astrosource.comparison.Vizier.query_region',mock_vizier_query_region_vsx)
def test_remove_targets_calibrated():
    parentPath = TEST_PATHS['parent']
    compFile, photFileArray, fileList = read_data_files(parentPath)
    assert compFile.shape == (60,2)
    compFile_out = remove_targets(parentPath, compFile, acceptDistance=5.0)
    # 3 stars are removed because they are variable
    assert compFile_out.shape == (57,2)

@patch('astrosource.comparison.Vizier.query_region',mock_vizier_query_region_apass_b)
def test_find_comparisons_calibrated_b():
    compFile = find_comparisons_calibrated('B', paths=TEST_PATHS)
    assert compFile.shape == (11,5)

@patch('astrosource.comparison.Vizier.query_region',mock_vizier_query_region_apass_v)
def test_find_comparisons_calibrated_v():
    compFile = find_comparisons_calibrated('V', paths=TEST_PATHS)
    assert compFile.shape == (11,5)
