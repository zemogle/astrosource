import glob
import sys
import os
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.vo_conesearch import conesearch
from astroquery.vo_conesearch import ConeSearch
from astroquery.vizier import Vizier

import logging

logger = logging.getLogger(__name__)


def find_comparisons(parentPath=None, stdMultiplier=3, thresholdCounts=1000000000, variabilityMultiplier=2.5, removeTargets=1, acceptDistance=1.0):
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

    # Get list of phot files
    if not parentPath:
        parentPath = Path(os.getcwd())
    if type(parentPath) == 'str':
        parentPath = Path(parentPath)

    compFile, photFileArray, fileList = read_data_files(parentPath)

    if removeTargets == 1:
        targetFile = remove_targets(parentPath, compFile, acceptDistance)

    while True:
        # First half of Loop: Add up all of the counts of all of the comparison stars
        # To create a gigantic comparison star.

        logger.debug("Please wait... calculating ensemble comparison star for each image")
        fileCount = ensemble_comparisons(photFileArray, compFile)

        # Second half of Loop: Calculate the variation in each candidate comparison star in brightness
        # compared to this gigantic comparison star.
        rejectStar=[]
        stdCompStar, sortStars = calculate_comparison_variation(compFile, photFileArray, fileCount)
        variabilityMax=(np.min(stdCompStar)*variabilityMultiplier)

        # Calculate and present the sample statistics
        stdCompMed=np.median(stdCompStar)
        stdCompStd=np.std(stdCompStar)

        logger.debug(fileCount)
        logger.debug(stdCompStar)
        logger.debug(np.median(stdCompStar))
        logger.debug(np.std(stdCompStar))

        # Delete comparisons that have too high a variability
        starRejecter=[]
        for j in range(len(stdCompStar)):
            logger.debug(stdCompStar[j])
            if ( stdCompStar[j] > (stdCompMed + (stdMultiplier*stdCompStd)) ):
                logger.debug("Star Rejected, Variability too high!")
                starRejecter.append(j)

            if ( np.isnan(stdCompStar[j]) ) :
                logger.debug("Star Rejected, Invalid Entry!")
                starRejecter.append(j)
        if starRejecter:
            logger.warning("Rejected {} stars".format(len(starRejecter)))


        compFile = np.delete(compFile, starRejecter, axis=0)
        sortStars = np.delete(sortStars, starRejecter, axis=0)

        # Calculate and present statistics of sample of candidate comparison stars.
        logger.info("Median variability {:.6f}".format(np.median(stdCompStar)))
        logger.info("Std variability {:.6f}".format(np.std(stdCompStar)))
        logger.info("Min variability {:.6f}".format(np.min(stdCompStar)))
        logger.info("Max variability {:.6f}".format(np.max(stdCompStar)))
        logger.info("Number of Stable Comparison Candidates {}".format(compFile.shape[0]))
        # Once we have stopped rejecting stars, this is our final candidate catalogue then we start to select the subset of this final catalogue that we actually use.
        if (starRejecter == []):
            break
        else:
            logger.warning("Trying again")

    logger.info('Statistical stability reached.')
    outfile, num_comparisons = final_candidate_catalogue(parentPath, photFileArray, sortStars, thresholdCounts, variabilityMax)
    return outfile, num_comparisons

def final_candidate_catalogue(parentPath, photFileArray, sortStars, thresholdCounts, variabilityMax):

    logger.info('List of stable comparison candidates output to stdComps.csv')

    np.savetxt(parentPath / "stdComps.csv", sortStars, delimiter=",", fmt='%0.8f')

    # The following process selects the subset of the candidates that we will use (the least variable comparisons that hopefully get the request countrate)

    # Sort through and find the largest file and use that as the reference file
    referenceFrame, fileRaDec = find_reference_frame(photFileArray)

    # SORT THE COMP CANDIDATE FILE such that least variable comparison is first
    sortStars=(sortStars[sortStars[:,2].argsort()])

    # PICK COMPS UNTIL OVER THE THRESHOLD OF COUNTS OR VRAIABILITY ACCORDING TO REFERENCE IMAGE
    logger.debug("PICK COMPARISONS UNTIL OVER THE THRESHOLD ACCORDING TO REFERENCE IMAGE")
    compFile=[]
    tempCountCounter=0.0
    for j in range(sortStars.shape[0]):
        matchCoord=SkyCoord(ra=sortStars[j][0]*u.degree, dec=sortStars[j][1]*u.degree)
        idx, d2d, d3d = matchCoord.match_to_catalog_sky(fileRaDec)

        if tempCountCounter < thresholdCounts:
            if sortStars[j][2] < variabilityMax:
                compFile.append([sortStars[j][0],sortStars[j][1]])
                tempCountCounter=np.add(tempCountCounter,referenceFrame[idx][4])
                logger.debug("Comp " + str(j+1) + " std: " + str(sortStars[j][2]))
                logger.debug("Cumulative Counts thus far: " + str(tempCountCounter))

    logger.debug("Selected stars listed below:")
    logger.debug(compFile)

    logger.info("Finale Ensemble Counts: " + str(tempCountCounter))
    compFile=np.asarray(compFile)

    logger.info(str(compFile.shape[0]) + " Stable Comparison Candidates below variability threshold output to compsUsed.csv")
    #logger.info(compFile.shape[0])

    outfile = parentPath / "compsUsed.csv"
    np.savetxt(outfile, compFile, delimiter=",", fmt='%0.8f')

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
    fileRaDec = SkyCoord(ra=referenceFrame[:,0]*u.degree, dec=referenceFrame[:,1]*u.degree)
    return referenceFrame, fileRaDec

def read_data_files(parentPath):
    fileList=[]
    used_file = parentPath / "usedImages.txt"
    with open(used_file, "r") as f:
        for line in f:
            fileList.append(line.strip())

    # LOAD Phot FILES INTO LIST

    photFileArray = []
    for file in fileList:
        photFileArray.append(np.genfromtxt(file, dtype=float, delimiter=','))
    photFileArray = np.asarray(photFileArray)


    #Grab the candidate comparison stars
    screened_file = parentPath / "screenedComps.csv"
    compFile = np.genfromtxt(screened_file, dtype=float, delimiter=',')
    return compFile, photFileArray, fileList

def ensemble_comparisons(photFileArray, compFile):
    fileCount=[]
    for photFile in photFileArray:
        allCounts=0.0
        fileRaDec = SkyCoord(ra=photFile[:,0]*u.degree, dec=photFile[:,1]*u.degree)
        for cf in compFile:
            matchCoord = SkyCoord(ra=cf[0]*u.degree, dec=cf[1]*u.degree)
            idx, d2d, d3d = matchCoord.match_to_catalog_sky(fileRaDec)
            allCounts = np.add(allCounts,photFile[idx][4])


        logger.debug("Total Counts in Image: {:.2f}".format(allCounts))
        fileCount=np.append(fileCount, allCounts)
    return fileCount

def calculate_comparison_variation(compFile, photFileArray, fileCount):
    stdCompStar=[]
    sortStars=[]

    for cf in compFile:
        compDiffMags = []
        q=0
        logger.debug("*************************")
        logger.debug("RA : " + str(cf[0]))
        logger.debug("DEC: " + str(cf[1]))
        for imgs in range(photFileArray.shape[0]):
            photFile = photFileArray[imgs]
            fileRaDec = SkyCoord(ra=photFile[:,0]*u.degree, dec=photFile[:,1]*u.degree)
            matchCoord = SkyCoord(ra=cf[0]*u.degree, dec=cf[1]*u.degree)
            idx, d2d, d3d = matchCoord.match_to_catalog_sky(fileRaDec)
            compDiffMags = np.append(compDiffMags,2.5 * np.log10(photFile[idx][4]/fileCount[q]))
            q = np.add(q,1)

        logger.debug("VAR: " +str(np.std(compDiffMags)))
        stdCompStar.append(np.std(compDiffMags))
        sortStars.append([cf[0],cf[1],np.std(compDiffMags),0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    return stdCompStar, sortStars

def remove_targets(parentPath, compFile, acceptDistance):
    max_sep=acceptDistance * u.arcsec
    logger.info("Removing Target Stars from potential Comparisons")
    targetFile = np.genfromtxt(parentPath / 'targetstars.csv', dtype=float, delimiter=',')
    fileRaDec = SkyCoord(ra=compFile[:,0]*u.degree, dec=compFile[:,1]*u.degree)
    # Remove any nan rows from targetFile
    targetRejecter=[]
    if not (targetFile.shape[0] == 4 and targetFile.size ==4):
        for z in range(targetFile.shape[0]):
          if np.isnan(targetFile[z][0]):
            targetRejecter.append(z)
        targetFile=np.delete(targetFile, targetRejecter, axis=0)

    # Remove targets from consideration
    if len(targetFile)== 4:
        loopLength=1
    else:
        loopLength=targetFile.shape[0]
    targetRejects=[]
    tg_file_len = len(targetFile)
    for tf in targetFile:
        if tg_file_len == 4:
            varCoord = SkyCoord(targetFile[0],(targetFile[1]), frame='icrs', unit=u.deg)
        else:
            varCoord = SkyCoord(tf[0],(tf[1]), frame='icrs', unit=u.deg) # Need to remove target stars from consideration
        idx, d2d, _ = varCoord.match_to_catalog_sky(fileRaDec)
        if d2d.arcsecond < acceptDistance:
            targetRejects.append(idx)
        if tg_file_len == 4:
            break
    compFile=np.delete(compFile, idx, axis=0)

    # Get Average RA and Dec from file
    if compFile.shape[0] == 13:
        logger.debug(compFile[0])
        logger.debug(compFile[1])
        avgCoord=SkyCoord(ra=(compFile[0])*u.degree, dec=(compFile[1]*u.degree))

    else:
        logger.debug(np.average(compFile[:,0]))
        logger.debug(np.average(compFile[:,1]))
        avgCoord=SkyCoord(ra=(np.average(compFile[:,0]))*u.degree, dec=(np.average(compFile[:,1]))*u.degree)


    # Check VSX for any known variable stars and remove them from the list
    variableResult=Vizier.query_region(avgCoord, '0.33 deg', catalog='VSX')['B/vsx/vsx']

    logger.debug(variableResult)

    variableResult=variableResult.to_pandas()

    logger.debug(variableResult.keys())

    variableSearchResult=variableResult[['RAJ2000','DEJ2000']].to_numpy()


    raCat=variableSearchResult[:,0]
    logger.debug(raCat)
    decCat=variableSearchResult[:,1]
    logger.debug(decCat)

    varStarReject=[]
    for t in range(raCat.size):
      logger.debug(raCat[t])
      compCoord=SkyCoord(ra=raCat[t]*u.degree, dec=decCat[t]*u.degree)
      logger.debug(compCoord)
      catCoords=SkyCoord(ra=compFile[:,0]*u.degree, dec=compFile[:,1]*u.degree)
      idxcomp,d2dcomp,d3dcomp=compCoord.match_to_catalog_sky(catCoords)
      logger.debug(d2dcomp)
      if d2dcomp *u.arcsecond < max_sep*u.arcsecond:
        logger.debug("match!")
        varStarReject.append(t)
      else:
        logger.debug("no match!")


    logger.debug("Number of stars prior to VSX reject")
    logger.debug(compFile.shape[0])
    compFile=np.delete(compFile, varStarReject, axis=0)
    logger.debug("Number of stars post to VSX reject")
    logger.debug(compFile.shape[0])


    if (compFile.shape[0] ==1):
        compFile=[[compFile[0][0],compFile[0][1],0.01]]
        compFile=np.asarray(compFile)
        np.savetxt(parentPath / "compsUsed.csv", compFile, delimiter=",", fmt='%0.8f')
        sortStars=[[compFile[0][0],compFile[0][1],0.01,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]]
        sortStars=np.asarray(sortStars)
        np.savetxt("stdComps.csv", sortStars, delimiter=",", fmt='%0.8f')
        raise Exception("Looks like you have a single comparison star!")
    return compFile
