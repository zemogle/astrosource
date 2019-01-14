import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
import glob
import sys
import os

import logging

logger = logging.getLogger(__name__)

def find_comparisons(parentPath, stdMultiplier=3, thresholdCounts=1000000, variabilityMax=0.025, removeTargets=1, acceptDistance=1.0):
    '''
    :stdMultiplier param:  This is how many standard deviations above the mean to cut off the top. The cycle will continue until there are no stars this many std.dev above the mean
    :thresholdCounts param:  This is the target countrate for the ensemble comparison... the lowest variability stars will be added until this countrate is reached.
    :variabilityMax param: This will stop adding ensemble comparisons if it starts using stars higher than this variability
    :removeTargets param: Set this to 1 to remove targets from consideration for comparison stars
    :acceptDistance param: Furtherest distance in arcseconds for matches

    '''

    # Get list of phot files
    if not parentPath:
        parentPath = os.getcwd()

    fileList=[]
    used_file = os.path.join(parentPath, "usedImages.txt")
    with open(used_file, "r") as f:
      for line in f:
        fileList.append(line.strip())

    # LOAD Phot FILES INTO LIST

    photFileArray=[]
    for file in fileList:
        photFileArray.append(numpy.genfromtxt(file, dtype=float, delimiter=','))
    photFileArray=numpy.asarray(photFileArray)


    #Grab the candidate comparison stars
    screened_file = os.path.join(parentPath, "screenedComps.csv")
    compFile = numpy.genfromtxt(screened_file, dtype=float, delimiter=',')

    if removeTargets == 1:
<<<<<<< HEAD
        logger.info("Removing Target Stars from potential Comparisons")
        targetFile = numpy.genfromtxt(os.path.join(parentPath, 'targetstars.csv'), dtype=float, delimiter=',')
=======
        logger.info("Removing Target Stars from potential comparisons")
        targetFile = numpy.genfromtxt('targetstars.csv', dtype=float, delimiter=',')
>>>>>>> e0abfd11dccb295ec14dd29d64430f333ecc495b
        fileRaDec = SkyCoord(ra=compFile[:,0]*u.degree, dec=compFile[:,1]*u.degree)
        for q in range(targetFile.shape[0]):
            varCoord = SkyCoord(targetFile[q][0],(targetFile[q][1]), frame='icrs', unit=u.deg) # Need to remove target stars from consideration
            idx, d2d, _ = varCoord.match_to_catalog_sky(fileRaDec)
            if d2d.arcsecond < acceptDistance:
              targetFile=numpy.delete(compFile, idx, axis=0)

    while True:
        # First half of Loop: Add up all of the counts of all of the comparison stars
        # To create a gigantic comparison star.
        fileCount=[]

        logger.debug("Please wait... calculating ensemble comparison star for each image")
        for imgs in range(photFileArray.shape[0]):
            allCounts=0.0
            photFile = photFileArray[imgs]
            fileRaDec = SkyCoord(ra=photFile[:,0]*u.degree, dec=photFile[:,1]*u.degree)
            for j in range(compFile.shape[0]):
                matchCoord=SkyCoord(ra=compFile[j][0]*u.degree, dec=compFile[j][1]*u.degree)
                idx, d2d, d3d = matchCoord.match_to_catalog_sky(fileRaDec)
                allCounts=numpy.add(allCounts,photFile[idx][4])


            logger.info(fileList[imgs] + "\nTotal Counts in Image: " + str(allCounts) +"\n*")
            fileCount=numpy.append(fileCount, allCounts)

        # Second half of Loop: Calculate the variation in each candidate comparison star in brightness
        # compared to this gigantic comparison star.
        rejectStar=[]
        stdCompStar=[]
        sortStars=[]
        for j in range(compFile.shape[0]):
            compDiffMags=[]
            q=0
            logger.info("*************************")
            logger.info("RA : " + str(compFile[j][0]))
            logger.info("DEC: " + str(compFile[j][1]))
            for imgs in range(photFileArray.shape[0]):
                #print file
                photFile = photFileArray[imgs]
                fileRaDec = SkyCoord(ra=photFile[:,0]*u.degree, dec=photFile[:,1]*u.degree)
                matchCoord=SkyCoord(ra=compFile[j][0]*u.degree, dec=compFile[j][1]*u.degree)
                idx, d2d, d3d = matchCoord.match_to_catalog_sky(fileRaDec)
                compDiffMags=numpy.append(compDiffMags,2.5 * numpy.log10(photFile[idx][4]/fileCount[q]))
                q=numpy.add(q,1)

            logger.info("VAR: " +str(numpy.std(compDiffMags)))
            stdCompStar.append(numpy.std(compDiffMags))
            sortStars.append([compFile[j][0],compFile[j][1],numpy.std(compDiffMags),0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

        # Calculate and present the sample statistics
        stdCompMed=numpy.median(stdCompStar)
        stdCompStd=numpy.std(stdCompStar)

        logger.info(fileCount)
        logger.info(stdCompStar)
        logger.info(numpy.median(stdCompStar))
        logger.info(numpy.std(stdCompStar))

        # Delete comparisons that have too high a variability
        starRejecter=[]
        for j in range(len(stdCompStar)):
            logger.info(stdCompStar[j])
            if ( stdCompStar[j] > (stdCompMed + (stdMultiplier*stdCompStd)) ):
                logger.error("Star Rejected, Variability too high!")
                starRejecter.append(j)
                
            if ( numpy.isnan(stdCompStar[j]) ) :
                logger.error("Star Rejected, Invalid Entry!")
                starRejecter.append(j)


        compFile=numpy.delete(compFile, starRejecter, axis=0)
        sortStars=numpy.delete(sortStars, starRejecter, axis=0)

        # Calculate and present statistics of sample of candidate comparison stars.
        logger.info("Median variability")
        logger.info(numpy.median(stdCompStar))
        logger.info("Std variability")
        logger.info(numpy.std(stdCompStar))
        logger.info("Min variability")
        logger.info(numpy.min(stdCompStar))
        logger.info("Max variability")
        logger.info(numpy.max(stdCompStar))
        logger.info("Number of Stable Comparison Candidates")
        logger.info((compFile.shape[0]))



        # Once we have stopped rejecting stars, this is our final candidate catalogue then we start to select the subset of this final catalogue that we actually use.
        if (starRejecter == []):

            logger.info('Statistical stability reached.')
            logger.info('List of stable comparison candidates output to stdComps.csv')

            numpy.savetxt(os.path.join(parentPath, "stdComps.csv"), sortStars, delimiter=",", fmt='%0.8f')

            # The following process selects the subset of the candidates that we will use (the least variable comparisons that hopefully get the request countrate)
            
            # Sort through and find the largest file and use that as the reference file
            fileSizer=0
            logger.info("Finding image with most stars detected")
            for imgs in range(photFileArray.shape[0]):
                photFile = photFileArray[imgs]
                if photFile.size > fileSizer:
                    referenceFrame=photFile
                    logger.info(photFile.size)
                    fileSizer=photFile.size
            logger.info("Setting up reference Frame")
            fileRaDec = SkyCoord(ra=referenceFrame[:,0]*u.degree, dec=referenceFrame[:,1]*u.degree)

            # SORT THE COMP CANDIDATE FILE such that least variable comparison is first
            sortStars=(sortStars[sortStars[:,2].argsort()])

            # PICK COMPS UNTIL OVER THE THRESHOLD OF COUNTS OR VRAIABILITY ACCORDING TO REFERENCE IMAGE
            logger.info("PICK COMPS UNTIL OVER THE THRESHOLD ACCORDING TO REFERENCE IMAGE")
            compFile=[]
            tempCountCounter=0.0
            for j in range(sortStars.shape[0]):
                matchCoord=SkyCoord(ra=sortStars[j][0]*u.degree, dec=sortStars[j][1]*u.degree)
                idx, d2d, d3d = matchCoord.match_to_catalog_sky(fileRaDec)                

                if tempCountCounter < thresholdCounts:
                    if sortStars[j][2] < variabilityMax:
                        compFile.append([sortStars[j][0],sortStars[j][1]])
                        tempCountCounter=numpy.add(tempCountCounter,referenceFrame[idx][4])
                        logger.info("Comp " + str(j+1) + " std: " + str(sortStars[j][2]))
                        logger.info("Cumulative Counts thus far: " + str(tempCountCounter))
                        
            logger.info("Selected stars listed below:")
            logger.info(compFile)

            logger.info("Finale Ensemble Counts: " + str(tempCountCounter))
            compFile=numpy.asarray(compFile)

            logger.info(str(compFile.shape[0]) + " Stable Comparison Candidates below variability threshold output to compsUsed.csv")
            #logger.info(compFile.shape[0])

            numpy.savetxt(os.path.join(parentPath,"compsUsed.csv"), compFile, delimiter=",", fmt='%0.8f')

            return
