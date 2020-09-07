import glob
import sys
import os
from pathlib import Path
from collections import namedtuple
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from numpy import min, max, median, std, isnan, delete, genfromtxt, savetxt, load, \
    asarray, add, append, log10, average, array, where
from astropy.units import degree, arcsecond
from astropy.coordinates import SkyCoord
from astroquery.sdss import SDSS
from astroquery.vo_conesearch import ConeSearch
from astroquery.vo_conesearch.exceptions import VOSError
from astroquery.vizier import Vizier


from astrosource.utils import AstrosourceException

import logging

logger = logging.getLogger('astrosource')


def find_comparisons(targets, parentPath=None, fileList=None, stdMultiplier=2.5, thresholdCounts=10000000, variabilityMultiplier=2.5, removeTargets=True, acceptDistance=1.0):
    '''
    Find stable comparison stars for the target photometry

    Parameters
    ----------
    parentPath : str or Path object
            Path to the data files
    stdMultiplier : int
            Number of standard deviations above the mean to cut off the top. The cycle will continue until there are no stars this many std.dev above the mean
    thresholdCounts : int
            Target countrate for the ensemble comparison. The lowest variability stars will be added until this countrate is reached.
    variabilityMax : float
            This will stop adding ensemble comparisons if it starts using stars higher than this variability
    removeTargets : int
            Set this to 1 to remove targets from consideration for comparison stars
    acceptDistance : float
            Furthest distance in arcseconds for matches

    Returns
    -------
    outfile : str

    '''
    sys.stdout.write("⭐️ Find stable comparison stars for differential photometry\n")
    sys.stdout.flush()
    # Get list of phot files
    if not parentPath:
        parentPath = Path(os.getcwd())
    if type(parentPath) == 'str':
        parentPath = Path(parentPath)

    compFile, photFileArray = read_data_files(parentPath, fileList)

    compFile = remove_stars_targets(parentPath, compFile, acceptDistance, targets, removeTargets)

    while True:
        # First half of Loop: Add up all of the counts of all of the comparison stars
        # To create a gigantic comparison star.

        logger.debug("Please wait... calculating ensemble comparison star for each image")
        fileCount = ensemble_comparisons(photFileArray, compFile)

        # Second half of Loop: Calculate the variation in each candidate comparison star in brightness
        # compared to this gigantic comparison star.
        rejectStar=[]
        stdCompStar, sortStars = calculate_comparison_variation(compFile, photFileArray, fileCount)

        variabilityMax=(min(stdCompStar)*variabilityMultiplier)

        # Calculate and present the sample statistics
        stdCompMed=median(stdCompStar)
        stdCompStd=std(stdCompStar)

        logger.debug(fileCount)
        logger.debug(stdCompStar)
        logger.debug(f"Median of comparisons = {stdCompMed}")
        logger.debug(f"STD of comparisons = {stdCompStd}")

        # Delete comparisons that have too high a variability

        starRejecter=[]
        if min(stdCompStar) > 0.002:
            for j in range(len(stdCompStar)):
                #logger.debug(stdCompStar[j])
                if ( stdCompStar[j] > (stdCompMed + (stdMultiplier*stdCompStd)) ):
                    logger.debug(f"Star {j} Rejected, Variability too high!")
                    starRejecter.append(j)

                if ( isnan(stdCompStar[j]) ) :
                    logger.debug("Star Rejected, Invalid Entry!")
                    starRejecter.append(j)
                sys.stdout.write('.')
                sys.stdout.flush()
            if starRejecter:
                logger.warning("Rejected {} stars".format(len(starRejecter)))
        else:
            logger.info("Minimum variability is too low for comparison star rejection by variability.")


        compFile = delete(compFile, starRejecter, axis=0)
        sortStars = delete(sortStars, starRejecter, axis=0)

        # Calculate and present statistics of sample of candidate comparison stars.
        logger.info("Median variability {:.6f}".format(median(stdCompStar)))
        logger.info("Std variability {:.6f}".format(std(stdCompStar)))
        logger.info("Min variability {:.6f}".format(min(stdCompStar)))
        logger.info("Max variability {:.6f}".format(max(stdCompStar)))
        logger.info("Number of Stable Comparison Candidates {}".format(compFile.shape[0]))
        # Once we have stopped rejecting stars, this is our final candidate catalogue then we start to select the subset of this final catalogue that we actually use.
        if (starRejecter == []):
            break
        else:
            logger.warning("Trying again")
            sys.stdout.write('💫')
            sys.stdout.flush()

    sys.stdout.write('\n')
    logger.info('Statistical stability reached.')
    
    outfile, num_comparisons = final_candidate_catalogue(parentPath, photFileArray, sortStars, thresholdCounts, variabilityMax)
    return outfile, num_comparisons

def final_candidate_catalogue(parentPath, photFileArray, sortStars, thresholdCounts, variabilityMax):

    logger.info('List of stable comparison candidates output to stdComps.csv')

    savetxt(parentPath / "stdComps.csv", sortStars, delimiter=",", fmt='%0.8f')

    # The following process selects the subset of the candidates that we will use (the least variable comparisons that hopefully get the request countrate)

    # Sort through and find the largest file and use that as the reference file
    referenceFrame, fileRaDec = find_reference_frame(photFileArray)

    # SORT THE COMP CANDIDATE FILE such that least variable comparison is first

    sortStars=(sortStars[sortStars[:,2].argsort()])

    # PICK COMPS UNTIL OVER THE THRESHOLD OF COUNTS OR VRAIABILITY ACCORDING TO REFERENCE IMAGE
    logger.debug("PICK COMPARISONS UNTIL OVER THE THRESHOLD ACCORDING TO REFERENCE IMAGE")
    compFile=[]
    tempCountCounter=0.0
    finalCountCounter=0.0
    for j in range(sortStars.shape[0]):
        if sortStars.size == 13 and sortStars.shape[0] == 1:
            matchCoord=SkyCoord(ra=sortStars[0][0]*degree, dec=sortStars[0][1]*degree)
        else:
            matchCoord=SkyCoord(ra=sortStars[j][0]*degree, dec=sortStars[j][1]*degree)
        idx, d2d, d3d = matchCoord.match_to_catalog_sky(fileRaDec)

        if tempCountCounter < thresholdCounts:
            if sortStars.size == 13 and sortStars.shape[0] == 1:
                compFile.append([sortStars[0][0],sortStars[0][1],sortStars[0][2]])
                logger.debug("Comp " + str(j+1) + " std: " + str(sortStars[0][2]))
                logger.debug("Cumulative Counts thus far: " + str(tempCountCounter))
                finalCountCounter=add(finalCountCounter,referenceFrame[idx][4])

            elif sortStars[j][2] < variabilityMax:
                compFile.append([sortStars[j][0],sortStars[j][1],sortStars[j][2]])
                logger.debug("Comp " + str(j+1) + " std: " + str(sortStars[j][2]))
                logger.debug("Cumulative Counts thus far: " + str(tempCountCounter))
                finalCountCounter=add(finalCountCounter,referenceFrame[idx][4])

        tempCountCounter=add(tempCountCounter,referenceFrame[idx][4])

    logger.debug("Selected stars listed below:")
    logger.debug(compFile)

    logger.info("Finale Ensemble Counts: " + str(finalCountCounter))
    compFile=asarray(compFile)

    logger.info(str(compFile.shape[0]) + " Stable Comparison Candidates below variability threshold output to compsUsed.csv")

    outfile = parentPath / "compsUsed.csv"
    savetxt(outfile, compFile, delimiter=",", fmt='%0.8f')

    return outfile, compFile.shape[0]

def find_reference_frame(photFileArray):
    fileSizer = 0
    logger.info("Finding image with most stars detected")
    for photFile in photFileArray:
        if photFile.size > fileSizer:
            referenceFrame = photFile
            logger.debug(photFile.size)
            fileSizer = photFile.size
    logger.info("Setting up reference Frame")
    fileRaDec = SkyCoord(ra=referenceFrame[:,0]*degree, dec=referenceFrame[:,1]*degree)
    return referenceFrame, fileRaDec

def read_data_files(parentPath, fileList):
    # LOAD Phot FILES INTO LIST
    photFileArray = []
    for file in fileList:
        photFileArray.append(load(parentPath / file))
    photFileArray = asarray(photFileArray)

    #Grab the candidate comparison stars
    screened_file = parentPath / "screenedComps.csv"
    compFile = genfromtxt(screened_file, dtype=float, delimiter=',')
    return compFile, photFileArray

def ensemble_comparisons(photFileArray, compFile):
    fileCount = []
    for photFile in photFileArray:
        allCounts = 0.0
        fileRaDec = SkyCoord(ra=photFile[:,0]*degree, dec=photFile[:,1]*degree)
        if compFile.size ==2 and compFile.shape[0]==2:
            matchCoord = SkyCoord(ra=compFile[0]*degree, dec=compFile[1]*degree)
            idx, d2d, d3d = matchCoord.match_to_catalog_sky(fileRaDec)
            allCounts = add(allCounts,photFile[idx][4])
        else:
            for cf in compFile:
                matchCoord = SkyCoord(ra=cf[0]*degree, dec=cf[1]*degree)
                idx, d2d, d3d = matchCoord.match_to_catalog_sky(fileRaDec)
                allCounts = add(allCounts,photFile[idx][4])
        logger.debug("Total Counts in Image: {:.2f}".format(allCounts))
        fileCount.append(allCounts)
    logger.debug("Total total {}".format(np.sum(np.array(fileCount))))
    return fileCount

def calculate_comparison_variation(compFile, photFileArray, fileCount):
    stdCompStar=[]
    sortStars=[]

    if compFile.size ==2 and compFile.shape[0]==2:
        compDiffMags = []
        logger.debug("*************************")
        logger.debug("RA : " + str(compFile[0]))
        logger.debug("DEC: " + str(compFile[1]))
        for q, photFile in enumerate(photFileArray):
            fileRaDec = SkyCoord(ra=photFile[:,0]*degree, dec=photFile[:,1]*degree)
            matchCoord = SkyCoord(ra=compFile[0]*degree, dec=compFile[1]*degree)
            idx, d2d, d3d = matchCoord.match_to_catalog_sky(fileRaDec)
            compDiffMags = append(compDiffMags,2.5 * log10(photFile[idx][4]/fileCount[q]))
        
        stdCompDiffMags=std(compDiffMags)
        logger.debug("VAR: " +str(stdCompDiffMags))
        if np.isnan(stdCompDiffMags) :
            logger.error("Star Variability non rejected")
            stdCompDiffMags=99
        stdCompStar.append(stdCompDiffMags)
        sortStars.append([compFile[0],compFile[1],stdCompDiffMags,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

            

    else:
        for cf in compFile:
            compDiffMags = []
            logger.debug("*************************")
            logger.debug("RA : " + str(cf[0]))
            logger.debug("DEC: " + str(cf[1]))
            for q, photFile in enumerate(photFileArray):
                fileRaDec = SkyCoord(ra=photFile[:,0]*degree, dec=photFile[:,1]*degree)
                matchCoord = SkyCoord(ra=cf[0]*degree, dec=cf[1]*degree)
                idx, d2d, d3d = matchCoord.match_to_catalog_sky(fileRaDec)
                compDiffMags = append(compDiffMags,2.5 * log10(photFile[idx][4]/fileCount[q]))

            stdCompDiffMags=std(compDiffMags)
            logger.debug("VAR: " +str(stdCompDiffMags))
            
            if np.isnan(stdCompDiffMags) :
                logger.error("Star Variability non rejected")
                stdCompDiffMags=99
            stdCompStar.append(stdCompDiffMags)
            sortStars.append([cf[0],cf[1],stdCompDiffMags,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])


    return stdCompStar, sortStars

def remove_stars_targets(parentPath, compFile, acceptDistance, targetFile, removeTargets):
    max_sep=acceptDistance * arcsecond
    logger.info("Removing Target Stars from potential Comparisons")

    if not (compFile.shape[0] == 2 and compFile.size ==2):
        fileRaDec = SkyCoord(ra=compFile[:,0]*degree, dec=compFile[:,1]*degree)
    else:
        fileRaDec = SkyCoord(ra=compFile[0]*degree, dec=compFile[1]*degree)

    # Remove any nan rows from targetFile
    targetRejecter=[]
    if not (targetFile.shape[0] == 4 and targetFile.size ==4):
        for z in range(targetFile.shape[0]):
          if isnan(targetFile[z][0]):
            targetRejecter.append(z)
        targetFile=delete(targetFile, targetRejecter, axis=0)

    # Get Average RA and Dec from file
    if compFile.shape[0] == 2 and compFile.size == 2:
        logger.debug(compFile[0])
        logger.debug(compFile[1])
        avgCoord=SkyCoord(ra=(compFile[0])*degree, dec=(compFile[1]*degree))

    else:
        logger.debug(average(compFile[:,0]))
        logger.debug(average(compFile[:,1]))
        avgCoord=SkyCoord(ra=(average(compFile[:,0]))*degree, dec=(average(compFile[:,1]))*degree)


    # Check VSX for any known variable stars and remove them from the list
    try:
        v=Vizier(columns=['all']) # Skymapper by default does not report the error columns
        v.ROW_LIMIT=-1
        variableResult=v.query_region(avgCoord, '0.33 deg', catalog='VSX')['B/vsx/vsx']
    except ConnectionError:
        connected=False
        logger.info("Connection failed, waiting and trying again")
        while connected==False:
            try:
                v=Vizier(columns=['all']) # Skymapper by default does not report the error columns
                v.ROW_LIMIT=-1
                variableResult=v.query_region(avgCoord, '0.33 deg', catalog='VSX')['B/vsx/vsx']
                connected=True
            except ConnectionError:
                time.sleep(10)
                logger.info("Failed again.")
                connected=False

    logger.debug(variableResult)

    logger.debug(variableResult.keys())

    raCat=array(variableResult['RAJ2000'].data)
    logger.debug(raCat)
    decCat=array(variableResult['DEJ2000'].data)
    logger.debug(decCat)
    varStarReject=[]
    for t in range(raCat.size):

        compCoord=SkyCoord(ra=raCat[t]*degree, dec=decCat[t]*degree)

        if not (compFile.shape[0] == 2 and compFile.size == 2):
            catCoords=SkyCoord(ra=compFile[:,0]*degree, dec=compFile[:,1]*degree)
            idxcomp,d2dcomp,d3dcomp=compCoord.match_to_catalog_sky(catCoords)
        elif not (raCat.shape[0] == 2 and raCat.size == 2): ### this is effictively the same as below
            catCoords=SkyCoord(ra=compFile[0]*degree, dec=compFile[1]*degree)
            idxcomp,d2dcomp,d3dcomp=compCoord.match_to_catalog_sky(catCoords)
        else:
            if abs(compFile[0]-raCat[0]) > 0.0014 and abs(compFile[1]-decCat[0]) > 0.0014:
                d2dcomp = 9999

        #logger.debug(d2dcomp)
        if d2dcomp != 9999:
            if d2dcomp.arcsecond[0] < max_sep.value:
                logger.debug("match!")
                varStarReject.append(t)
        #    else:
        #        logger.debug("no match!")

    logger.debug("Number of stars prior to VSX reject")
    logger.debug(compFile.shape[0])
    compFile=delete(compFile, varStarReject, axis=0)
    logger.debug("Number of stars post to VSX reject")
    logger.debug(compFile.shape[0])


    if (compFile.shape[0] ==1):
        compFile=[[compFile[0][0],compFile[0][1],0.01]]
        compFile=asarray(compFile)
        savetxt(parentPath / "compsUsed.csv", compFile, delimiter=",", fmt='%0.8f')
        sortStars=[[compFile[0][0],compFile[0][1],0.01,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]]
        sortStars=asarray(sortStars)
        savetxt("stdComps.csv", sortStars, delimiter=",", fmt='%0.8f')
        raise AstrosourceException("Looks like you have a single comparison star!")
    return compFile


def catalogue_call(avgCoord, opt, cat_name, targets, closerejectd):
    data = namedtuple(typename='data',field_names=['ra','dec','mag','emag','cat_name'])

    TABLES = {'APASS':'II/336/apass9',
              'SDSS' :'V/147/sdss12',
              'PanSTARRS' : 'II/349/ps1',
              'SkyMapper' : 'II/358/smss'
              }

    tbname = TABLES.get(cat_name, None)
    kwargs = {'radius':'0.33 deg'}
    kwargs['catalog'] = cat_name

    try:
        v=Vizier(columns=['all']) # Skymapper by default does not report the error columns
        v.ROW_LIMIT=-1
        query = v.query_region(avgCoord, **kwargs)
    except VOSError:
        raise AstrosourceException("Could not find RA {} Dec {} in {}".format(avgCoord.ra.value,avgCoord.dec.value, cat_name))
    except ConnectionError:
        connected=False
        logger.info("Connection failed, waiting and trying again")
        while connected==False:
            try:
                v=Vizier(columns=['all']) # Skymapper by default does not report the error columns
                v.ROW_LIMIT=-1
                query = v.query_region(avgCoord, **kwargs)
                connected=True
            except ConnectionError:
                time.sleep(10)
                logger.info("Failed again.")
                connected=False

    if query.keys():
        resp = query[tbname]
    else:
        raise AstrosourceException("Could not find RA {} Dec {} in {}".format(avgCoord.ra.value,avgCoord.dec.value, cat_name))


    logger.debug(f'Looking for sources in {cat_name}')
    if cat_name in ['APASS','PanSTARRS']:
        radecname = {'ra' :'RAJ2000', 'dec': 'DEJ2000'}
    elif cat_name == 'SDSS':
        radecname = {'ra' :'RA_ICRS', 'dec': 'DE_ICRS'}
    elif cat_name == 'SkyMapper':
        radecname = {'ra' :'RAICRS', 'dec': 'DEICRS'}
    else:
        radecname = {'ra' :'raj2000', 'dec': 'dej2000'}

    # Filter out bad data from catalogues
    if cat_name == 'PanSTARRS':
        resp = resp[where((resp['Qual'] == 52) | (resp['Qual'] == 60) | (resp['Qual'] == 61))]
    elif cat_name == 'SDSS':
        resp = resp[resp['Q'] == 3]
    elif cat_name == 'SkyMapper':
        resp = resp[resp['flags'] == 0]

    logger.info("Original high quality sources in calibration catalogue: "+str(len(resp)))

    # Remove any objects close to targets from potential calibrators
    if targets.shape == (4,):
        targets = [targets]
    for tg in targets:
        resp = resp[where(np.abs(resp[radecname['ra']]-tg[0]) > 0.0014) and where(np.abs(resp[radecname['dec']]-tg[1]) > 0.0014)]

    logger.info("Number of calibration sources after removal of sources near targets: "+str(len(resp)))

    # Remove any star from calibration catalogue that has another star in the catalogue within closerejectd arcseconds of it.
    while True:
        fileRaDec = SkyCoord(ra=resp[radecname['ra']].data*degree, dec=resp[radecname['dec']].data*degree)
        idx, d2d, _ = fileRaDec.match_to_catalog_sky(fileRaDec, nthneighbor=2) # Closest matches that isn't itself.
        catReject = []
        for q in range(len(d2d)):
            if d2d[q] < closerejectd*arcsecond:
                catReject.append(q)
        if catReject == []:
            break
        del resp[catReject]
        logger.info(f"Stars rejected that are too close (<5arcsec) in calibration catalogue: {len(catReject)}")

    logger.info(f"Number of calibration sources after removal of sources near other sources: {len(resp)}")


    data.cat_name = cat_name
    data.ra = array(resp[radecname['ra']].data)
    data.dec = array(resp[radecname['dec']].data)

    # extract RA, Dec, Mag and error as arrays
    data.mag = array(resp[opt['filter']].data)
    data.emag = array(resp[opt['error']].data)
    return data

def find_comparisons_calibrated(targets, paths, filterCode, nopanstarrs=False, nosdss=False, closerejectd=5.0, max_magerr=0.05, stdMultiplier=2, variabilityMultiplier=2):
    sys.stdout.write("⭐️ Find comparison stars in catalogues for calibrated photometry\n")

    FILTERS = {
                'B' : {'APASS' : {'filter' : 'Bmag', 'error' : 'e_Bmag'}},
                'V' : {'APASS' : {'filter' : 'Vmag', 'error' : 'e_Vmag'}},
                'up' : {'SDSS' : {'filter' : 'umag', 'error' : 'e_umag'},
                        'SkyMapper' : {'filter' : 'uPSF', 'error' : 'e_uPSF'}},
                'gp' : {'SDSS' : {'filter' : 'gmag', 'error' : 'e_mag'},
                        'SkyMapper' : {'filter' : 'gPSF', 'error' : 'e_gPSF'},
                        'PanSTARRS': {'filter' : 'gmag', 'error' : 'e_gmag'}},
                'rp' : {'SDSS' : {'filter' : 'rmag', 'error' : 'e_rmag'},
                        'SkyMapper' : {'filter' : 'rPSF', 'error' : 'e_rPSF'},
                        'PanSTARRS': {'filter' : 'rmag', 'error' : 'e_rmag'}},
                'ip' : {'SDSS' : {'filter' : 'imag', 'error' : 'e_imag'},
                        'SkyMapper' : {'filter' : 'iPSF', 'error' : 'e_iPSF'},
                        'PanSTARRS': {'filter' : 'imag', 'error' : 'e_imag'}},
                'zs' : {'PanSTARRS': {'filter' : 'zmag', 'error' : 'e_zmag'},
                        'SkyMapper' : {'filter' : 'zPSF', 'error' : 'e_zPSF'},
                        'SDSS' : {'filter' : 'zmag', 'error' : 'e_zmag'}},
                }


    parentPath = paths['parent']
    calibPath = parentPath / "calibcats"
    if not calibPath.exists():
        os.makedirs(calibPath)

    #Vizier.ROW_LIMIT = -1

    # Get List of Files Used
    fileList=[]
    for line in (parentPath / "usedImages.txt").read_text().strip().split('\n'):
        fileList.append(line.strip())

    logger.debug("Filter Set: " + filterCode)

    # Load compsused
    compFile = genfromtxt(parentPath / 'stdComps.csv', dtype=float, delimiter=',')
    logger.debug(compFile.shape[0])

    if compFile.shape[0] == 13 and compFile.size == 13:
        compCoords=SkyCoord(ra=compFile[0]*degree, dec=compFile[1]*degree)
    else:
        compCoords=SkyCoord(ra=compFile[:,0]*degree, dec=compFile[:,1]*degree)

    # Get Average RA and Dec from file
    if compFile.shape[0] == 13 and compFile.size == 13:
        logger.debug(compFile[0])
        logger.debug(compFile[1])
        avgCoord=SkyCoord(ra=(compFile[0])*degree, dec=(compFile[1]*degree))

    else:
        logger.debug(average(compFile[:,0]))
        logger.debug(average(compFile[:,1]))
        avgCoord=SkyCoord(ra=(average(compFile[:,0]))*degree, dec=(average(compFile[:,1]))*degree)

    try:
        catalogues = FILTERS[filterCode]
    except IndexError:
        raise AstrosourceException(f"{filterCode} is not accepted at present")

    # Look up in online catalogues

    coords=[]
    for cat_name, opt in catalogues.items():
        try:
            if coords ==[]: #SALERT - Do not search if a suitable catalogue has already been found
                logger.info("Searching " + str(cat_name))
                if cat_name == 'PanSTARRS' and nopanstarrs==True:
                    logger.info("Skipping PanSTARRS")
                elif cat_name == 'SDSS' and nosdss==True:
                    logger.info("Skipping SDSS")
                else:
                    coords = catalogue_call(avgCoord, opt, cat_name, targets=targets, closerejectd=closerejectd)
                    if coords.cat_name == 'PanSTARRS' or coords.cat_name == 'APASS':
                        max_sep=2.5 * arcsecond
                    else:
                        max_sep=1.5 * arcsecond
                    if coords !=[]:
                        cat_used=cat_name


        except AstrosourceException as e:
            logger.debug(e)

    if not coords:
        raise AstrosourceException(f"Could not find coordinate match in any catalogues for {filterCode}")

    #Setup standard catalogue coordinates
    catCoords=SkyCoord(ra=coords.ra*degree, dec=coords.dec*degree)

    #Get calib mags for least variable IDENTIFIED stars.... not the actual stars in compUsed!! Brighter, less variable stars may be too bright for calibration!
    #So the stars that will be used to calibrate the frames to get the OTHER stars.
    calibStands=[]

    if compFile.shape[0] ==13 and compFile.size ==13:
        lenloop=1
    else:
        lenloop=len(compFile[:,0])

    for q in range(lenloop):
        if compFile.shape[0] ==13 and compFile.size ==13:
            compCoord=SkyCoord(ra=compFile[0]*degree, dec=compFile[1]*degree)
        else:
            compCoord=SkyCoord(ra=compFile[q][0]*degree, dec=compFile[q][1]*degree)
        idxcomp,d2dcomp,d3dcomp=compCoord.match_to_catalog_sky(catCoords)
        if d2dcomp < max_sep:
            if not isnan(coords.mag[idxcomp]):
                if compFile.shape[0] ==13 and compFile.size ==13:
                    calibStands.append([compFile[0],compFile[1],compFile[2],coords.mag[idxcomp],coords.emag[idxcomp]])
                else:
                    calibStands.append([compFile[q][0],compFile[q][1],compFile[q][2],coords.mag[idxcomp],coords.emag[idxcomp]])
    logger.info('Calibration Stars Identified below')
    logger.info(calibStands)



    # Get the set of least variable stars to use as a comparison to calibrate the files (to eventually get the *ACTUAL* standards
    #logger.debug(asarray(calibStands).shape[0])
    if asarray(calibStands).shape[0] == 0:
        logger.info("We could not find a suitable match between any of your stars and the calibration catalogue")
        logger.info("You might need to reduce the low value (usually 10000) to get some dimmer stars in script 1")
        logger.info("You might also try using one of --nosdss or --nopanstarrs option (not both!) to prevent comparisons to these catalogues")
        raise AstrosourceException("Stars are too dim to calibrate to.")

    varimin=(min(asarray(calibStands)[:,2])) * variabilityMultiplier


    loopbreaker=0
    while loopbreaker==0:
        calibStandsReject=[]
        for q in range(len(asarray(calibStands)[:,0])):
            if calibStands[q][2] > varimin:
                calibStandsReject.append(q)
            elif calibStands[q][4] == 0:
                calibStandsReject.append(q)
            elif np.isnan(calibStands[q][4]):
                calibStandsReject.append(q)
        if len(calibStands) > len(calibStandsReject):
            loopbreaker=1
        else:
            varimin=varimin+0.01
          

    calibStands=delete(calibStands, calibStandsReject, axis=0)

    calibStand=asarray(calibStands)

    savetxt(parentPath / "calibStands.csv", calibStands , delimiter=",", fmt='%0.8f')
    # Lets use this set to calibrate each datafile and pull out the calibrated compsused magnitudes
    compUsedFile = genfromtxt(parentPath / 'compsUsed.csv', dtype=float, delimiter=',')

    calibCompUsed=[]

    calibOverlord=[] # a huge array intended to create the calibration plot and data out of all the individual calibration files.

    logger.debug("CALIBRATING EACH FILE")
    for file in fileList:
        logger.debug(file)

        #Get the phot file into memory
        photFile = load(parentPath / file)
        photCoords=SkyCoord(ra=photFile[:,0]*degree, dec=photFile[:,1]*degree)

        #Convert the phot file into instrumental magnitudes
        for r in range(len(photFile[:,0])):
            photFile[r,5]=1.0857 * (photFile[r,5]/photFile[r,4])
            photFile[r,4]=-2.5*log10(photFile[r,4])

        #Pull out the CalibStands out of each file
        tempDiff=[]
        calibOut=[]
        for q in range(len(calibStands[:,0])):
            if calibStands.size == 13 and calibStands.shape[0]== 13:
                calibCoord=SkyCoord(ra=calibStand[0]*degree,dec=calibStand[1]*degree)
                idx,d2d,d3d=calibCoord.match_to_catalog_sky(photCoords)
                tempDiff.append(calibStand[3]-photFile[idx,4])
                calibOut.append([calibStand[3],calibStand[4],photFile[idx,4],photFile[idx,5],calibStand[3]-photFile[idx,4],0])
            else:
                calibCoord=SkyCoord(ra=calibStand[q][0]*degree,dec=calibStand[q][1]*degree)
                idx,d2d,d3d=calibCoord.match_to_catalog_sky(photCoords)
                tempDiff.append(calibStand[q,3]-photFile[idx,4])
                calibOut.append([calibStand[q,3],calibStand[q,4],photFile[idx,4],photFile[idx,5],calibStand[q,3]-photFile[idx,4],0])

        #logger.debug(tempDiff)
        tempZP= (median(tempDiff))
        #logger.debug(std(tempDiff))


        #Shift the magnitudes in the phot file by the zeropoint
        for r in range(len(photFile[:,0])):
            photFile[r,4]=photFile[r,4]+tempZP

        calibOut=asarray(calibOut)

        #Shift the magnitudes in the phot file by the zeropoint

        for r in range(len(calibOut[:,0])):
            calibOut[r,5]=calibOut[r,4]-tempZP        
            calibOverlord.append([calibOut[r,0],calibOut[r,1],calibOut[r,2],calibOut[r,3],calibOut[r,4],calibOut[r,5],float(file.split("_")[2].replace("d","."))])

        file = Path(file)
        #Save the calibrated photfiles to the calib directory
        #savetxt(calibPath / "{}.calibrated.{}".format(file.stem, file.suffix), photFile, delimiter=",", fmt='%0.8f')
        savetxt(calibPath / "{}.calibrated.{}".format(file.stem, 'csv'), photFile, delimiter=",", fmt='%0.8f')
        savetxt(calibPath / "{}.compared.{}".format(file.stem, 'csv'), calibOut, delimiter=",", fmt='%0.8f')

        


        #Look within photfile for ACTUAL usedcomps.csv and pull them out
        lineCompUsed=[]
        if compUsedFile.shape[0] ==3 and compUsedFile.size == 3:
            lenloop=1
        else:
            lenloop=len(compUsedFile[:,0])
        #logger.debug(compUsedFile.size)
        for r in range(lenloop):
            if compUsedFile.shape[0] ==3 and compUsedFile.size ==3:
                compUsedCoord=SkyCoord(ra=compUsedFile[0]*degree,dec=compUsedFile[1]*degree)
            else:

                compUsedCoord=SkyCoord(ra=compUsedFile[r][0]*degree,dec=compUsedFile[r][1]*degree)
            idx,d2d,d3d=compUsedCoord.match_to_catalog_sky(photCoords)
            lineCompUsed.append(photFile[idx,4])

        #logger.debug(lineCompUsed)
        calibCompUsed.append(lineCompUsed)
        sys.stdout.write('.')
        sys.stdout.flush()

    calibOverlord=asarray(calibOverlord)
    savetxt(parentPath / "CalibAll.csv", calibOverlord, delimiter=",", fmt='%0.8f')

    # Difference versus Magnitude calibration plot
    plt.cla()
    fig = plt.gcf()
    outplotx=calibOverlord[:,0]
    outploty=calibOverlord[:,5]
    sqsol = np.linalg.lstsq(np.vstack([calibOverlord[:,0],np.ones(len(calibOverlord[:,0]))]).T,calibOverlord[:,5], rcond=None)
    m, c = sqsol[0]
    x, residuals, rank, s = sqsol
    #logger.info(m)
    #logger.info(c)
    #logger.info(residuals[0])
    
    plt.xlabel(str(cat_used) + ' ' +str(filterCode) + ' Catalogue Magnitude')
    plt.ylabel('Calibrated - Catalogue Magnitude')
    plt.plot(outplotx,outploty,'bo')
    plt.plot(outplotx,m*outplotx+c,'r')
    #plt.plot(outplotxrepeat,outploty,'ro')
    #plt.plot(linex,liney)
    plt.ylim(min(outploty)-0.05,max(outploty)+0.05,'k-')
    plt.xlim(min(outplotx)-0.05,max(outplotx)+0.05)
    #plt.errorbar(outplotx, outploty, xerr=3*calibOverlord[:,1], fmt='-o', linestyle='None')
    #plt.errorbar(outplotxrepeat, outploty, yerr=3*calibFile[:,2], fmt='-o', linestyle='None')
    plt.grid(True)
    plt.subplots_adjust(left=0.15, right=0.98, top=0.98, bottom=0.17, wspace=0.3, hspace=0.4)
    fig.set_size_inches(6,3)
    plt.savefig(parentPath / str("CalibrationSanityPlot_" +str(filterCode)+"_Magnitude.png"))
    plt.savefig(parentPath / str("CalibrationSanityPlot_" +str(filterCode)+"_Magnitude.eps"))

    with open(parentPath / "CalibrationSanityPlotCoefficients.txt", "w") as f:
        f.write("Magnitude slope     : " + str(m)+"\n")
        f.write("Magnitude zeropoint : " + str(c) +"\n")
        if not residuals.size == 0:
            f.write("Magnitude residuals : " +str(residuals[0])+"\n")
        else:
            f.write("Magnitude residuals not calculated. \n")

    # Difference vs time calibration plot
    plt.cla()
    fig = plt.gcf()
    outplotx=calibOverlord[:,6]
    outploty=calibOverlord[:,5]
    sqsol = np.linalg.lstsq(np.vstack([calibOverlord[:,6],np.ones(len(calibOverlord[:,6]))]).T,calibOverlord[:,5], rcond=None)
    m, c = sqsol[0]
    x, residuals, rank, s = sqsol
    
    plt.xlabel('BJD')
    plt.ylabel('Calibrated - Catalogue Magnitude')
    plt.plot(outplotx,outploty,'bo')
    plt.plot(outplotx,m*outplotx+c,'r')
    #plt.plot(outplotxrepeat,outploty,'ro')
    #plt.plot(linex,liney)
    plt.ylim(min(outploty)-0.05,max(outploty)+0.05,'k-')
    plt.xlim(min(outplotx)-0.05,max(outplotx)+0.05)
    #plt.errorbar(outplotx, outploty, xerr=3*calibOverlord[:,1], fmt='-o', linestyle='None')
    #plt.errorbar(outplotxrepeat, outploty, yerr=3*calibFile[:,2], fmt='-o', linestyle='None')
    plt.grid(True)
    plt.subplots_adjust(left=0.15, right=0.98, top=0.98, bottom=0.17, wspace=0.3, hspace=0.4)
    fig.set_size_inches(6,3)
    plt.savefig(parentPath / str("CalibrationSanityPlot_" +str(filterCode)+"_Time.png"))
    plt.savefig(parentPath / str("CalibrationSanityPlot_" +str(filterCode)+"_Time.eps"))

    with open(parentPath / "CalibrationSanityPlotCoefficients.txt", "a") as f:
        f.write("Time slope     : " + str(m)+"\n")
        f.write("Time zeropoint : " + str(c) +"\n")        
        if not residuals.size == 0:
            f.write("Time residuals : " +str(residuals[0])+"\n")
        else:
            f.write("Time residuals not calculated. \n")
    # Finalise calibcompsusedfile
    #logger.debug(calibCompUsed)

    calibCompUsed=asarray(calibCompUsed)
    #logger.debug(calibCompUsed[0,:])

    finalCompUsedFile=[]
    sumStd=[]
    for r in range(len(calibCompUsed[0,:])):
        #Calculate magnitude and stdev
        sumStd.append(std(calibCompUsed[:,r]))

        if compUsedFile.shape[0] ==3  and compUsedFile.size ==3:
            finalCompUsedFile.append([compUsedFile[0],compUsedFile[1],compUsedFile[2],median(calibCompUsed[:,r]),asarray(calibStands[0])[4]])
        else:
            finalCompUsedFile.append([compUsedFile[r][0],compUsedFile[r][1],compUsedFile[r][2],median(calibCompUsed[:,r]),std(calibCompUsed[:,r])])

    #logger.debug(finalCompUsedFile)
    logger.debug(" ")
    sumStd=asarray(sumStd)

    errCalib = median(sumStd) / pow((len(calibCompUsed[0,:])), 0.5)

    logger.debug("Comparison Catalogue: " + str(cat_used))
    if len(calibCompUsed[0,:]) == 1:
        logger.debug("As you only have one comparison, the uncertainty in the calibration is unclear")
        logger.debug("But we can take the catalogue value, although we should say this is a lower uncertainty")
        logger.debug("Error/Uncertainty in Calibration: " +str(asarray(calibStands[0])[4]))
    else:
        logger.debug("Median Standard Deviation of any one star: " + str(median(sumStd)))
        logger.debug("Standard Error/Uncertainty in Calibration: " +str(errCalib))

    with open(parentPath / "calibrationErrors.txt", "w") as f:
        f.write("Comparison Catalogue: " + str(cat_used)+"\n")
        f.write("Median Standard Deviation of any one star: " + str(median(sumStd)) +"\n")
        f.write("Standard Error/Uncertainty in Calibration: " +str(errCalib))

    #logger.debug(finalCompUsedFile)
    compFile = asarray(finalCompUsedFile)
    savetxt(parentPath / "calibCompsUsed.csv", compFile, delimiter=",", fmt='%0.8f')
    sys.stdout.write('\n')
    return compFile
