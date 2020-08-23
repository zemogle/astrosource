import numpy as np
from pathlib import Path
import os
import pytest
from unittest.mock import patch, Mock

from astrosource.identify import find_stars, read_data_files
from astrosource.comparison import find_comparisons, find_comparisons_calibrated, \
    calibrate_photometry

TEST_PATH_PARENT = Path(os.path.dirname(__file__)) / 'test_files'

TEST_PATHS = {'parent': TEST_PATH_PARENT / 'speedup'}

INPUT_FILES = ['RXEri_B_2458372d8045823071_2018d09d11T07d09d16d123_1a3954565_17d286_kb81_michaelfitzgerald.npy',
 'RXEri_B_2458547d2593600489_2019d03d04T18d20d17d106_1a1375401_13d283_kb84_timdjones7.npy',
 'RXEri_B_2458549d2488945760_2019d03d06T18d05d17d487_1a124071_13d284_kb84_timdjones7.npy',
 'rxeri_B_2458562d2884747311_2019d03d19T19d02d33d828_1a5178674_13d285_kb84_timdjones7.npy',
 'RXEri_B_2458372d1298749419_2018d09d10T14d57d43d008_1a2899937_16d942_kb27_michaelfitzgerald.npy',
 'RXEri_B_2458371d6046093008_2018d09d10T02d21d20d615_1a1529288_17d283_kb96_michaelfitzgerald.npy',
 'RXEri_B_2458371d8837984502_2018d09d10T09d03d22d513_1a0790084_17d284_kb26_michaelfitzgerald.npy',
 'RXEri_B_2458546d2483008588_2019d03d03T18d04d18d732_1a1029248_13d409_kb84_timdjones7.npy',
 'rxeri_B_2458567d2286917502_2019d03d24T17d36d28d604_1a2230736_13d284_kb84_timdjones7.npy',
 'RXEri_B_2458371d6050731102_2018d09d10T02d22d00d686_1a1514762_17d282_kb96_michaelfitzgerald.npy',
 'RXEri_B_2458371d7887208089_2018d09d10T06d46d27d791_1a5547859_17d285_kb26_michaelfitzgerald.npy',
 'RXEri_B_2458372d2446666160_2018d09d10T17d43d00d539_1a153668_17d238_kb24_michaelfitzgerald.npy',
 'RXEri_B_2458371d6041340921_2018d09d10T02d20d40d435_1a1543828_17d283_kb96_michaelfitzgerald.npy',
 'RXEri_B_2458550d2448007651_2019d03d07T17d59d26d062_1a1206428_13d284_kb84_timdjones7.npy',
 'RXEri_B_2458553d3472028011_2019d03d10T20d26d58d726_1a5740949_13d338_kb25_timdjones7.npy',
 'rxeri_B_2458563d4888034821_2019d03d20T23d51d02d943_1a1935013_13d284_kb95_timdjones7.npy',
 'RXEri_B_2458556d7220212580_2019d03d14T05d26d47d877_1a3792819_12d941_kb82_timdjones7.npy',
 'rxeri_B_2458564d2591135348_2019d03d21T18d20d18d252_1a3380838_13d286_kb84_timdjones7.npy',
 'rxeri_B_2458572d4780848781_2019d03d29T23d35d31d896_1a2556188_13d27_kb95_timdjones7.npy',
 'RXEri_B_2458371d7881178972_2018d09d10T06d45d36d244_1a5607974_17d282_kb26_michaelfitzgerald.npy',
 'RXEri_B_2458371d7884193030_2018d09d10T06d46d01d955_1a557816_17d281_kb26_michaelfitzgerald.npy',
 'RXEri_B_2458546d5005480940_2019d03d04T00d07d33d062_1a0886046_13d274_kb95_timdjones7.npy',
 'RXEri_B_2458538d9453712050_2019d02d24T10d47d41d844_1a1967743_12d236_kb24_timdjones7.npy',
 'RXEri_B_2458373d2712414172_2018d09d11T18d21d15d095_1a0806183_17d284_kb97_michaelfitzgerald.npy',
 'RXEri_B_2458551d5023669698_2019d03d09T00d10d21d803_1a1273514_13d284_kb26_timdjones7.npy',
 'RXEri_B_2458546d7165800710_2019d03d04T05d18d39d063_1a275722_12d938_kb82_timdjones7.npy',
 'RXEri_B_2458372d2443031101_2018d09d10T17d42d29d196_1a1548354_17d231_kb24_michaelfitzgerald.npy',
 'rxeri_B_2458565d7208396262_2019d03d23T05d25d10d720_1a5115732_12d935_kb82_timdjones7.npy',
 'rxeri_B_2458561d7279610210_2019d03d19T05d35d25d127_1a4858374_12d938_kb82_timdjones7.npy',
 'rxeri_B_2458563d2439167080_2019d03d20T17d58d24d326_1a2422401_13d284_kb84_timdjones7.npy',
 'rxeri_B_2458562d2941195020_2019d03d19T19d10d42d392_1a5720272_13d285_kb84_timdjones7.npy',
 'RXEri_B_2458372d5559694339_2018d09d11T01d11d17d291_1a3677274_17d284_kb96_michaelfitzgerald.npy',
 'rxeri_B_2458569d4848327660_2019d03d26T23d45d17d809_1a2486176_13d271_kb95_timdjones7.npy',
 'RXEri_B_2458372d8840406071_2018d09d11T09d03d41d555_1a0735748_17d284_kb26_michaelfitzgerald.npy',
 'RXEri_B_2458556d2388501181_2019d03d13T17d51d02d210_1a1501603_13d414_kb84_timdjones7.npy',
 'RXEri_B_2458371d8843895621_2018d09d10T09d04d13d923_1a0778809_17d284_kb26_michaelfitzgerald.npy',
 'RXEri_B_2458550d7245730362_2019d03d08T05d30d18d903_1a3255874_12d936_kb82_timdjones7.npy',
 'RXEri_B_2458549d7269251840_2019d03d07T05d33d40d042_1a3242769_12d942_kb82_timdjones7.npy',
 'RXEri_B_2458546d9721883661_2019d03d04T11d26d44d299_1a4939507_13d241_kb24_timdjones7.npy',
 'RXEri_B_2458551d7279483438_2019d03d09T05d35d12d710_1a3472418_12d937_kb82_timdjones7.npy',
 'RXEri_B_2458372d5550421169_2018d09d11T01d09d56d831_1a3739617_17d284_kb96_michaelfitzgerald.npy',
 'RXEri_B_2458545d7314822222_2019d03d03T05d40d03d579_1a3046662_12d943_kb27_timdjones7.npy',
 'RXEri_B_2458550d9097424028_2019d03d08T09d56d57d913_1a1872191_13d364_kb24_timdjones7.npy',
 'RXEri_B_2458372d6274012239_2018d09d11T02d54d08d219_1a0896772_17d283_kb96_michaelfitzgerald.npy',
 'RXEri_B_2458554d7210935010_2019d03d12T05d25d25d588_1a3525922_12d937_kb82_timdjones7.npy',
 'RXEri_B_2458372d1302230479_2018d09d10T14d58d13d684_1a2891454_16d941_kb27_michaelfitzgerald.npy',
 'RXEri_B_2458555d2558344072_2019d03d12T18d15d27d965_1a1997096_13d415_kb84_timdjones7.npy',
 'rxeri_B_2458562d4868777301_2019d03d19T23d48d16d132_1a1755274_13d28_kb95_timdjones7.npy',
 'rxeri_B_2458568d5292426851_2019d03d26T00d49d15d429_1a5423496_13d271_kb95_timdjones7.npy',
 'RXEri_B_2458372d1305710552_2018d09d10T14d58d43d755_1a2883226_16d94_kb27_michaelfitzgerald.npy',
 'rxeri_B_2458571d2316264221_2019d03d28T17d40d40d162_1a2897261_13d285_kb84_timdjones7.npy',
 'RXEri_B_2458372d5555057260_2018d09d11T01d10d37d014_1a3708549_17d281_kb96_michaelfitzgerald.npy',
 'rxeri_B_2458566d2378424001_2019d03d23T17d49d39d812_1a252196_13d288_kb84_timdjones7.npy',
 'RxEri_B_2458537d9042287529_2019d02d23T09d48d23d151_1a075957_12d418_kb56_timdjones7.npy',
 'rxeri_B_2458563d7303392440_2019d03d21T05d38d51d344_1a5410063_12d938_kb82_timdjones7.npy',
 'rxeri_B_2458570d2310841158_2019d03d27T17d39d53d781_1a2726041_13d284_kb84_timdjones7.npy',
 'RXEri_B_2458371d8840998560_2018d09d10T09d03d48d204_1a0784428_17d283_kb26_michaelfitzgerald.npy',
 'RXEri_B_2458372d8049290129_2018d09d11T07d09d46d356_1a3929148_17d282_kb81_michaelfitzgerald.npy',
 'RXEri_B_2458537d9070002250_2019d02d23T09d52d22d078_1a080858_14d283_kb56_timdjones7.npy',
 'RXEri_B_2458372d8846783182_2018d09d11T09d04d36d444_1a0724503_17d285_kb26_michaelfitzgerald.npy',
 'RXEri_B_2458545d2890574918_2019d03d02T19d02d57d440_1a223001_13d285_kb84_timdjones7.npy',
 'RXEri_B_2458372d6269252161_2018d09d11T02d53d27d189_1a0905814_17d284_kb96_michaelfitzgerald.npy',
 'RXEri_B_2458548d2558790790_2019d03d05T18d15d18d099_1a1354742_13d281_kb84_timdjones7.npy',
 'RXEri_B_2458372d6278648321_2018d09d11T02d54d48d408_1a0888027_17d285_kb96_michaelfitzgerald.npy',
 'RXEri_B_2458372d8843422122_2018d09d11T09d04d07d478_1a0730405_17d281_kb26_michaelfitzgerald.npy',
 'RXEri_B_2458547d9144644300_2019d03d05T10d03d39d075_1a1740964_13d284_kb56_timdjones7.npy',
 'RXEri_B_2458546d8903485439_2019d03d04T09d28d53d132_1a0973914_13d247_kb24_timdjones7.npy',
 'rxeri_B_2458564d7275050352_2019d03d22T05d34d46d642_1a5402195_12d942_kb82_timdjones7.npy',
 'RXEri_B_2458372d2439407031_2018d09d10T17d41d57d938_1a1560202_17d231_kb24_michaelfitzgerald.npy',
 'rxeri_B_2458561d2342808051_2019d03d18T17d44d31d319_1a180549_13d284_kb84_timdjones7.npy',
 'rxeri_B_2458564d4847789630_2019d03d21T23d45d15d295_1a1884099_13d271_kb95_timdjones7.npy',
 'RXEri_B_2458372d8042353010_2018d09d11T07d08d46d204_1a3980143_17d284_kb81_michaelfitzgerald.npy']

class TestSetup:
    def __init__(self):
        # Add targets to the TestSetup object
        self.targets = np.array([(72.43451,  -15.74109, 0.00000000, 0.00000000)])
        self.screenedComps = np.array([[72.40802640,-15.80937480],
                    [72.47512030,-15.70976510],
                    [72.39157640,-15.62051080],
                    [72.40985350,-15.74274110],
                    [72.49152350,-15.72360160],
                    [72.48819650,-15.72387280],
                    [72.47163550,-15.89579990],
                    [72.31605150,-15.55956820],
                    [72.48506360,-15.61679210],
                    [72.37477530,-15.85890660],
                    [72.46545850,-15.92890730],
                    [72.43951170,-15.90962870],
                    [72.56781390,-15.67062440]])
        fileList = []
        for line in (TEST_PATHS['parent'] / "usedImages.txt").read_text().strip().split('\n'):
            fileList.append(line.strip())
        self.imagesUsed = fileList
        self.compusedfile = np.array([[ 72.4080264, -15.8093748, 0.19196379],
                     [ 72.4751203, -15.7097651, 0.30405236],
                     [ 72.3915764, -15.6205108, 0.3101752 ]])
        self.calibstands = np.array([[72.39157640,-15.62051080,0.31017520,8.87300014,0.00000000],
                    [72.40802640,-15.80937480,0.19196379,8.75199986,0.00000000],
                    [72.40985350,-15.74274110,0.32368866,11.68500042,0.03000000],
                    [72.47512030,-15.70976510,0.30405236,11.13399982,0.04800000]])
        self.calibcompused = np.array([[72.40802640,-15.80937480,0.19196379,8.73505216,0.27071088],
                    [72.47512030,-15.70976510,0.30405236,11.12645864,0.15221891],
                    [72.39157640,-15.62051080,0.31017520,8.87137490,0.44098755]])

@pytest.fixture()
def setup():
    return TestSetup()


def mock_read_data_files(paths,filelist):
    photFileArray = np.load(TEST_PATHS['parent'] / 'speedup_photarray.npy', allow_pickle=True)
    return photFileArray, INPUT_FILES

@patch('astrosource.identify.read_data_files', mock_read_data_files)
def test_find_stars(setup):
    usedImages, stars = find_stars(setup.targets, TEST_PATHS, INPUT_FILES)
    assert usedImages == setup.imagesUsed
    # assert np.any(stars) == False
    assert stars.shape[1] == len(setup.screenedComps)
    assert stars[0].all() == setup.screenedComps.all()
    stars, comparisons, compFile = find_comparisons(setup.targets, TEST_PATHS['parent'], usedImages, photlist=stars)
    print(stars.shape)
    calibStands, goodcalib, cat_used = calibrate_photometry(setup.targets, filterCode='B', variabilityMultiplier=2, starvar=compFile, closerejectd=0.5)
    assert calibStands[goodcalib,:].shape == setup.calibstands.shape
    assert calibStands[goodcalib,:].all() == setup.calibstands.all()

    compfile = find_comparisons_calibrated(targets=setup.targets, filterCode='B', paths=TEST_PATHS, photlist=stars, starvar=compFile, comparisons=comparisons)
    assert compfile.shape == setup.calibcompused.shape
