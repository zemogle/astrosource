from numpy import genfromtxt, savetxt, load, delete, asarray, multiply, log10, divide, \
    less, append, add, std, average, median, inf, nan, isnan, nanstd, nanmean
from astropy.units import degree
from astropy.coordinates import SkyCoord
import glob
import sys
import matplotlib
from pathlib import Path
matplotlib.use("Agg") # must be before pyplot

import matplotlib.pyplot as plt
import math
import os

import logging

from astrosource.utils import photometry_files_to_array, AstrosourceException

logger = logging.getLogger('astrosource')


def calculate_curves(targets, acceptDistance=10.0, errorReject=0.05, parentPath = None):
    '''
    Find stable comparison stars for the target photometry

    Parameters
    ----------
    targetStars : list
            List of target tuples in the formal (ra, dec, 0, 0). ra and dec must be in decimal
    acceptDistance : float
        acceptible distance between stars in different images
    errorReject : float
        reject measurements with instrumental errors larger than this (this is not total error, just the estimated error in the single measurement of the variable)
    acceptDistance : float
        Furthest distance in arcseconds for matches

    Returns
    -------
    outfile : str
    '''
    minimumVariableCounts=10000  # Do not try to detect variables dimmer than this.
    minimumNoOfObs=10 # Minimum number of observations to count as a potential variable.


    # Load in list of used files
    fileList=[]
    with open(parentPath / "usedImages.txt", "r") as f:
      for line in f:
        fileList.append(line.strip())

    # LOAD Phot FILES INTO LIST
    photFileArray=[]
    for file in fileList:
        photFileArray.append(load(file))
    photFileArray=asarray(photFileArray)

    # LOAD IN COMPARISON FILE
    preFile = genfromtxt(parentPath / 'stdComps.csv', dtype=float, delimiter=',')

    logger.debug(preFile.shape)
    if preFile.shape[0] !=13:
        preFile=(preFile[preFile[:,2].argsort()])

    # GET REFERENCE IMAGE
    # Sort through and find the largest file and use that as the reference file
    fileSizer=0
    logger.debug("Finding image with most stars detected")
    for imgs in range(photFileArray.shape[0]):
        photFile = photFileArray[imgs]
        if photFile.size > fileSizer:
            referenceFrame=photFile
            logger.debug(photFile.size)
            fileSizer=photFile.size

    compFile=genfromtxt(parentPath / "compsUsed.csv", dtype=float, delimiter=',')
    logger.debug("Stable Comparison Candidates below variability threshold")
    outputPhot=[]

    # Get total counts for each file

    fileCount=[]
    compArray=[]
    allCountsArray=[]

    for imgs in range(photFileArray.shape[0]):
        allCounts=0.0
        allCountsErr=0.0
        photFile = photFileArray[imgs]
        fileRaDec = SkyCoord(ra=photFile[:,0]*degree, dec=photFile[:,1]*degree)
        #Array of comp measurements
        logger.debug("***************************************")
        logger.debug("Calculating total Comparison counts for")
        logger.debug(fileList[imgs])

        logger.debug(compFile.shape)
        #logger.debug"Calculating total counts"
        for j in range(compFile.shape[0]):
            if compFile.size == 2 or (compFile.shape[0]== 3 and compFile.size ==3) or (compFile.shape[0]== 5 and compFile.size ==5):
                matchCoord=SkyCoord(ra=compFile[0]*degree, dec=compFile[1]*degree)
            else:
                matchCoord=SkyCoord(ra=compFile[j][0]*degree, dec=compFile[j][1]*degree)
            idx, d2d, d3d = matchCoord.match_to_catalog_sky(fileRaDec)
            allCounts=allCounts+photFile[idx][4]
            allCountsErr=allCountsErr+photFile[idx][5]
            if (compFile.shape[0]== 5 and compFile.size ==5) or (compFile.shape[0]== 3 and compFile.size ==3):
                break

        allCountsArray.append([allCounts,allCountsErr])

    logger.debug(allCountsArray)

    # Define targetlist as every star in referenceImage above a count threshold
    logger.debug("Setting up Variable Search List")
    targetFile=referenceFrame
    # Although remove stars that are below the variable countrate
    starReject=[]
    for q in range(targetFile.shape[0]):
        if targetFile[q][4] < minimumVariableCounts:
            starReject.append(q)
    logger.debug("Total number of stars in reference Frame: {}".format(targetFile.shape[0]))
    targetFile=delete(targetFile, starReject, axis=0)
    logger.debug("Total number of stars with sufficient counts: {}".format(targetFile.shape[0]))

    ## NEED TO REMOVE COMPARISON STARS FROM TARGETLIST

    allcountscount=0
    # For each variable calculate the variability
    outputVariableHolder=[]
    for q in range(targetFile.shape[0]):
        logger.debug("*********************")
        logger.debug("Processing Target {}".format(str(q+1)))
        logger.debug("RA {}".format(targetFile[q][0]))
        logger.debug("DEC {}".format(targetFile[q][1]))
        varCoord = SkyCoord(targetFile[q][0],(targetFile[q][1]), frame='icrs', unit=degree) # Need to remove target stars from consideration
        outputPhot=[]
        compArray=[]
        compList=[]

        diffMagHolder=[]

        allcountscount=0

        for imgs in range(photFileArray.shape[0]):
            compList=[]
            photFile = photFileArray[imgs]

            fileRaDec = SkyCoord(ra=photFile[:,0]*degree, dec=photFile[:,1]*degree)

            idx, d2d, d3d = varCoord.match_to_catalog_sky(fileRaDec)
            if (less(d2d.arcsecond, acceptDistance) and ((multiply(-2.5,log10(divide(photFile[idx][4],allCountsArray[allcountscount][0])))) != inf )):

                diffMagHolder=append(diffMagHolder,(multiply(-2.5,log10(divide(photFile[idx][4],allCountsArray[allcountscount][0])))))
            allcountscount=add(allcountscount,1)


        ## REMOVE MAJOR OUTLIERS FROM CONSIDERATION
        while True:
            stdVar=std(diffMagHolder)
            avgVar=average(diffMagHolder)
            starReject=[]
            z=0
            for j in range(asarray(diffMagHolder).shape[0]):
                if diffMagHolder[j] > avgVar+(4*stdVar) or diffMagHolder[j] < avgVar-(4*stdVar) :
                    starReject.append(j)
                    logger.debug("REJECT {}".format(diffMagHolder[j]))
                    z=1
            diffMagHolder=delete(diffMagHolder, starReject, axis=0)
            if z==0:
                break


        logger.debug("Standard Deviation in mag: {}".format(std(diffMagHolder)))
        logger.debug("Median Magnitude: {}".format(median(diffMagHolder)))
        logger.debug("Number of Observations: {}".format(asarray(diffMagHolder).shape[0]))

        if (  asarray(diffMagHolder).shape[0] > minimumNoOfObs):
            outputVariableHolder.append( [targetFile[q][0],targetFile[q][1],median(diffMagHolder), std(diffMagHolder), asarray(diffMagHolder).shape[0]])

    savetxt(parentPath / "starVariability.csv", outputVariableHolder, delimiter=",", fmt='%0.8f')

    return outputVariableHolder

def photometric_calculations(targets, paths, acceptDistance=10.0, errorReject=0.5):
    sys.stdout.write('🖥 Starting photometric calculations\n')

    photFileArray,fileList = photometry_files_to_array(paths['parent'])

    if (paths['parent'] / 'calibCompsUsed.csv').exists():
        logger.debug("Calibrated")
        compFile=genfromtxt(paths['parent'] / 'calibCompsUsed.csv', dtype=float, delimiter=',')
        calibFlag=1
    else:
        logger.debug("Differential")
        compFile=genfromtxt(paths['parent'] / 'compsUsed.csv', dtype=float, delimiter=',')
        calibFlag=0

    # Get total counts for each file
    fileCount=[]
    compArray=[]
    allCountsArray=[]
    for ifile, photFile in enumerate(photFileArray):
        allCounts=0.0
        allCountsErr=0.0
        fileRaDec = SkyCoord(ra=photFile[:,0]*degree, dec=photFile[:,1]*degree)
        logger.debug(f"Calculating total Comparison counts for : {ifile}")
        #logger.debug(compFile.shape[0])

        if compFile.shape[0]== 5 and compFile.size ==5:
            loopLength=1
        else:
            loopLength=compFile.shape[0]
        #logger.debug(compFile.size)
                    #sys.exit()
        for j in range(loopLength):
            if compFile.size == 2 or (compFile.shape[0]== 5 and compFile.size ==5):
    ##
    ##    for j in range(compFile.shape[0]):
    ##        if compFile.size == 2 or compFile.shape[0] == 5:
                matchCoord=SkyCoord(ra=compFile[0]*degree, dec=compFile[1]*degree)
            else:
                matchCoord=SkyCoord(ra=compFile[j][0]*degree, dec=compFile[j][1]*degree)
            idx, d2d, d3d = matchCoord.match_to_catalog_sky(fileRaDec)
            allCounts=add(allCounts,photFile[idx][4])
            allCountsErr=add(allCountsErr,photFile[idx][5])

        allCountsArray.append([allCounts,allCountsErr])

    logger.debug(allCountsArray)

    allcountscount=0

    if len(targets)== 4:
        loopLength=1
    else:
        loopLength=targets.shape[0]
    # For each variable calculate all the things
    for q in range(loopLength):
        starErrorRejCount=0
        starDistanceRejCount=0
        logger.debug("****************************")
        logger.debug("Processing Variable {}".format(q+1))
        if int(len(targets)) == 4:
            logger.debug("RA {}".format(targets[0]))
        else:
            logger.debug("RA {}".format(targets[q][0]))
        if int(len(targets)) == 4:
            logger.debug("Dec {}".format(targets[1]))
        else:
            logger.debug("Dec {}".format(targets[q][1]))
        if int(len(targets)) == 4:
            varCoord = SkyCoord(targets[0],(targets[1]), frame='icrs', unit=degree) # Need to remove target stars from consideration
        else:
            varCoord = SkyCoord(targets[q][0],(targets[q][1]), frame='icrs', unit=degree) # Need to remove target stars from consideration

        # Grabbing variable rows
        logger.debug("Extracting and Measuring Differential Magnitude in each Photometry File")
        outputPhot=[] # new
        compArray=[]
        compList=[]
        allcountscount=0
        for imgs, photFile in enumerate(photFileArray):
            sys.stdout.write('.')
            compList=[]
            fileRaDec = SkyCoord(ra=photFile[:,0]*degree, dec=photFile[:,1]*degree)
            idx, d2d, _ = varCoord.match_to_catalog_sky(fileRaDec)
            starRejected=0
            if (less(d2d.arcsecond, acceptDistance)):
                magErrVar = 1.0857 * (photFile[idx][5]/photFile[idx][4])
                if magErrVar < errorReject:

                    magErrEns = 1.0857 * (allCountsErr/allCounts)
                    magErrTotal = pow( pow(magErrVar,2) + pow(magErrEns,2),0.5)

                    #templist is a temporary holder of the resulting file.
                    tempList=photFile[idx,:]
                    googFile = Path(fileList[imgs]).name
                    tempList=append(tempList, float(googFile.split("_")[2].replace("d",".")))
                    tempList=append(tempList, float(googFile.split("_")[4].replace("a",".")))
                    tempList=append(tempList, allCountsArray[allcountscount][0])
                    tempList=append(tempList, allCountsArray[allcountscount][1])

                    #Differential Magnitude
                    tempList=append(tempList, 2.5 * log10(allCountsArray[allcountscount][0]/photFile[idx][4]))
                    tempList=append(tempList, magErrTotal)
                    tempList=append(tempList, photFile[idx][4])
                    tempList=append(tempList, photFile[idx][5])



                    if (compFile.shape[0]== 5 and compFile.size ==5) or (compFile.shape[0]== 3 and compFile.size ==3):
                        loopLength=1
                    else:
                        loopLength=compFile.shape[0]
                    #logger.debug(compFile.size)
                    #sys.exit()
                    for j in range(loopLength):
                        if compFile.size == 2 or (compFile.shape[0]== 3 and compFile.size ==3) or (compFile.shape[0]== 5 and compFile.size ==5):
                            matchCoord=SkyCoord(ra=compFile[0]*degree, dec=compFile[1]*degree)
                        else:
                            matchCoord=SkyCoord(ra=compFile[j][0]*degree, dec=compFile[j][1]*degree)
                        idx, d2d, d3d = matchCoord.match_to_catalog_sky(fileRaDec)
                        tempList=append(tempList, photFile[idx][4])

                    outputPhot.append(tempList)
                    fileCount.append(allCounts)
                    allcountscount=allcountscount+1

                else:
                    starErrorRejCount=starErrorRejCount+1
                    starRejected=1
            else:
                starDistanceRejCount=starDistanceRejCount+1
                starRejected=1

            if ( starRejected == 1):

                    #templist is a temporary holder of the resulting file.
                    tempList=photFileArray[imgs][idx,:]
                    googFile = Path(fileList[imgs]).name
                    tempList=append(tempList, float(googFile.split("_")[5].replace("d",".")))
                    tempList=append(tempList, float(googFile.split("_")[4].replace("a",".")))
                    tempList=append(tempList, allCountsArray[allcountscount][0])
                    tempList=append(tempList, allCountsArray[allcountscount][1])

                    #Differential Magnitude
                    tempList=append(tempList,nan)
                    tempList=append(tempList,nan)
                    tempList=append(tempList, photFileArray[imgs][idx][4])
                    tempList=append(tempList, photFileArray[imgs][idx][5])


                    if (compFile.shape[0]== 5 and compFile.size ==5) or (compFile.shape[0]== 3 and compFile.size ==3):
                        loopLength=1
                    else:
                        loopLength=compFile.shape[0]
                    #logger.debug(compFile.shape[0])
                    #sys.exit()
                    for j in range(loopLength):
                        if compFile.size == 2 or (compFile.shape[0]== 3 and compFile.size ==3) or (compFile.shape[0]== 5 and compFile.size ==5):
                            matchCoord=SkyCoord(ra=compFile[0]*degree, dec=compFile[1]*degree)
                        else:
                            matchCoord=SkyCoord(ra=compFile[j][0]*degree, dec=compFile[j][1]*degree)
                        idx, d2d, d3d = matchCoord.match_to_catalog_sky(fileRaDec)
                        tempList=append(tempList, photFileArray[imgs][idx][4])
                    outputPhot.append(tempList)
                    fileCount.append(allCounts)
                    allcountscount=allcountscount+1

        # Check for dud images
        imageReject=[]
        for j in range(asarray(outputPhot).shape[0]):
            if isnan(outputPhot[j][11]):
                imageReject.append(j)
        outputPhot=delete(outputPhot, imageReject, axis=0)

        ## REMOVE MAJOR OUTLIERS FROM CONSIDERATION
        stdVar=nanstd(asarray(outputPhot)[:,10])
        avgVar=nanmean(asarray(outputPhot)[:,10])
        starReject=[]
        stdevReject=0
        for j in range(asarray(outputPhot).shape[0]):
            if outputPhot[j][10] > avgVar+(4*stdVar) or outputPhot[j][10] < avgVar-(4*stdVar) :
                starReject.append(j)
                stdevReject=stdevReject+1
        sys.stdout.write('\n')
        logger.info("Rejected Stdev Measurements: : {}".format(stdevReject))
        logger.info("Rejected Error Measurements: : {}".format(starErrorRejCount))
        logger.info("Rejected Distance Measurements: : {}".format(starDistanceRejCount))
        logger.info("Variability of Comparisons")
        logger.info("Average : {}".format(avgVar))
        logger.info("Stdev   : {}".format(stdVar))

        outputPhot=delete(outputPhot, starReject, axis=0)
        if outputPhot.shape[0] > 2:
            savetxt(os.path.join(paths['outcatPath'],"doerPhot_V" +str(q+1) +".csv"), outputPhot, delimiter=",", fmt='%0.8f')
            logger.debug('Saved doerPhot_V')
        else:
            raise AstrosourceException("Photometry not possible")
        return outputPhot
