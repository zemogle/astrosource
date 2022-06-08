import glob
import sys
import os
from pathlib import Path
from collections import namedtuple
import numpy as np
import time
from datetime import datetime
import shutil

import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from numpy import min, max, median, std, isnan, delete, genfromtxt, savetxt, load, \
    asarray, add, append, log10, average, array, where
from astropy.units import degree, arcsecond
from astropy.coordinates import SkyCoord
from astroquery.vo_conesearch.exceptions import VOSError
from astroquery.vizier import Vizier


from astrosource.utils import AstrosourceException

import logging

logger = logging.getLogger('astrosource')



def check_comparisons_files(parentPath=None, fileList=None, matchRadius=1.45):
    '''
    Find stable comparison stars for the target photometry
    '''

    sys.stdout.write("⭐️ Checking Provided stars are in every image\n")
    sys.stdout.flush()
    
    # Get list of phot files
    if not parentPath:
        parentPath = Path(os.getcwd())
    if type(parentPath) == 'str':
        parentPath = Path(parentPath)
    
    
    
    photFileArray = []
    for file in fileList:
        photFileArray.append(load(parentPath / file))
    photFileArray = asarray(photFileArray)
    
    compFile = genfromtxt(parentPath / 'compsUsed.csv', dtype=float, delimiter=',')
    
    #print ("Constructing Sky Coords for photometry files....")
    photSkyCoord=[]
    for q, photFile in enumerate(photFileArray):
        #print (q)
        photSkyCoord.append(SkyCoord(ra=photFile[:,0]*degree, dec=photFile[:,1]*degree))
    #file1=open(parentPath / "photSkyCoord","wb")
    #pickle.dump(photSkyCoord, file1)
    #file1.close
    q=0
    
    photFileHolder=[]
    photSkyCoord=[]
    for file in fileList: 
        photFile = load(parentPath / file)
        photFileHolder.append(photFile)
        photSkyCoord.append(SkyCoord(ra=photFile[:,0]*degree, dec=photFile[:,1]*degree))
    
    usedImages=[]
    q=0
    imageRemove=[]
    for file in fileList:
        #logger.debug(file)
        photRAandDec = photSkyCoord[q]
        # Check that each star is in the image
        rejectImage=0
        for j in range(compFile.shape[0]):            
            testStar = SkyCoord(ra = compFile[j][0]*degree, dec = compFile[j][1]*degree)
            # This is the function in the whole package which requires scipy
            idx, d2d, _ = testStar.match_to_catalog_sky(photRAandDec)
            if (d2d.arcsecond > matchRadius):
                #"No Match! Nothing within range."
                rejectImage=1
            
        if rejectImage==1:
            logger.debug('**********************')
            logger.debug('The Following file does not contain one or more of the provided comparison stars. It has been removed')
            logger.debug(file)
            logger.debug('**********************')
            #fileList.remove(file)
            imageRemove.append(q)
        else:   
            
            usedImages.append(file)
            
        q=q+1
    
    fileList=delete(fileList, imageRemove, axis=0)
    
    used_file =parentPath / "usedImages.txt"
    with open(used_file, "w") as f:
        for s in usedImages:
            filename = Path(s).name
            f.write(str(filename) +"\n")
    
    logger.debug('Checking Completed.')
    
    return usedImages


def find_comparisons(targets,  parentPath=None, fileList=None, photFileArray=None, photSkyCoord=None, matchRadius=1.45, stdMultiplier=2.5, thresholdCounts=10000000, variabilityMultiplier=2.5, removeTargets=True):
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
    matchRadius: float
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

    #compFile, photFileArray = read_data_files(parentPath, fileList)

    screened_file = parentPath / "screenedComps.csv"
    compFile = genfromtxt(screened_file, dtype=float, delimiter=',')

    compFile = remove_stars_targets(parentPath, compFile, matchRadius, targets, removeTargets)

    # Removes odd duplicate entries from comparison list
    compFile=np.unique(compFile, axis=0)

    # Create or load in skycoord array
    # if os.path.exists(parentPath / "photSkyCoord"):
    #     print ("Loading Sky coords for photometry files")
    #     with open(parentPath / "photSkyCoord", 'rb') as f:
    #         photSkyCoord=pickle.load(f)
    # else:
    # print ("Constructing Sky Coords for photometry files....")
    # photSkyCoord=[]
    # for q, photFile in enumerate(photFileArray):
    #     #print (q)
    #     photSkyCoord.append(SkyCoord(ra=photFile[:,0]*degree, dec=photFile[:,1]*degree))
    #file1=open(parentPath / "photSkyCoord","wb")
    #pickle.dump(photSkyCoord, file1)
    #file1.close

    while True:
        # First half of Loop: Add up all of the counts of all of the comparison stars
        # To create a gigantic comparison star.

        logger.debug("Please wait... calculating ensemble comparison star for each image")
        fileCount = ensemble_comparisons(photFileArray, compFile, parentPath, photSkyCoord)

        # Second half of Loop: Calculate the variation in each candidate comparison star in brightness
        # compared to this gigantic comparison star.

        stdCompStar, sortStars = calculate_comparison_variation(compFile, photFileArray, fileCount, parentPath, photSkyCoord)

        variabilityMax=(min(stdCompStar)*variabilityMultiplier)

        # Calculate and present the sample statistics
        stdCompMed=median(stdCompStar)
        stdCompStd=std(stdCompStar)

        #logger.debug(fileCount)
        #logger.debug(stdCompStar)
        logger.debug(f"Median of comparisons = {stdCompMed}")
        logger.debug(f"STD of comparisons = {stdCompStd}")

        # Delete comparisons that have too high a variability

        starRejecter=[]
        if min(stdCompStar) > 0.0009:
            for j in range(len(stdCompStar)):

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

    savetxt(parentPath / "referenceFrame.csv", referenceFrame, delimiter=",", fmt='%0.8f')

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
                with open(parentPath / "EnsembleStats.txt", "w") as f:
                    f.write("Comp " + str(j+1) + " std: " + str(sortStars[0][2]))
                    f.write("Cumulative Counts thus far: " + str(tempCountCounter))
                finalCountCounter=add(finalCountCounter,referenceFrame[idx][4])

            elif sortStars[j][2] < variabilityMax:
                compFile.append([sortStars[j][0],sortStars[j][1],sortStars[j][2]])
                logger.debug("Comp " + str(j+1) + " std: " + str(sortStars[j][2]))
                logger.debug("Cumulative Counts thus far: " + str(tempCountCounter))
                with open(parentPath / "EnsembleStats.txt", "w") as f:
                    f.write("Comp " + str(j+1) + " std: " + str(sortStars[j][2]))
                    f.write("Cumulative Counts thus far: " + str(tempCountCounter))
                finalCountCounter=add(finalCountCounter,referenceFrame[idx][4])

        tempCountCounter=add(tempCountCounter,referenceFrame[idx][4])

    logger.debug("Selected stars listed below:")
    logger.debug(compFile)

    logger.info("Finale Ensemble Counts: " + str(finalCountCounter))
    
    
        
    compFile=asarray(compFile)

    logger.info(str(compFile.shape[0]) + " Stable Comparison Candidates below variability threshold output to compsUsed.csv")

    with open(parentPath / "EnsembleStats.txt", "w") as f:
        f.write('Number of counts at mag zero: ' + str(finalCountCounter) +"\n")
        f.write('Number of stars used: ' + str(j) +"\n")

    outfile = parentPath / "compsUsed.csv"
    savetxt(outfile, compFile, delimiter=",", fmt='%0.8f')

    return outfile, compFile.shape[0]

def find_reference_frame(photFileArray):
    fileSizer = 0
    logger.info("Finding image with most stars detected")
    for photFile in photFileArray:
        if photFile.size > fileSizer:
            referenceFrame = photFile
            #logger.debug(photFile.size)
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

def ensemble_comparisons(photFileArray, compFile, parentPath, photSkyCoord):
        
    # fileCount = []
    # q=0
    # for photFile in photFileArray:
    #     allCounts = 0.0
    #     #fileRaDec = SkyCoord(ra=photFile[:,0]*degree, dec=photFile[:,1]*degree)
    #     if compFile.size ==2 and compFile.shape[0]==2:
    #         matchCoord = SkyCoord(ra=compFile[0]*degree, dec=compFile[1]*degree)
    #         idx, d2d, _ = matchCoord.match_to_catalog_sky(photSkyCoord[q])
    #         allCounts = add(allCounts,photFile[idx][4])
    #     else:
    #         for cf in compFile:
    #             matchCoord = SkyCoord(ra=cf[0]*degree, dec=cf[1]*degree)
    #             idx, d2d, _ = matchCoord.match_to_catalog_sky(photSkyCoord[q])
    #             allCounts = add(allCounts,photFile[idx][4])
    #     logger.debug("Total Counts in Image: {:.2f}".format(allCounts))
    #     fileCount.append(allCounts)
    #     q=q+1
    
    # get rid of dumb for loop
    fileCount = []
    q=0
    for photFile in photFileArray:
        allCounts = 0.0
        #fileRaDec = SkyCoord(ra=photFile[:,0]*degree, dec=photFile[:,1]*degree)
        # if compFile.size ==2 and compFile.shape[0]==2:
        #     matchCoord = SkyCoord(ra=compFile[0]*degree, dec=compFile[1]*degree)
        #     idx, d2d, _ = matchCoord.match_to_catalog_sky(photSkyCoord[q])
        #     allCounts = add(allCounts,photFile[idx][4])
        # else:
        #     for cf in compFile:
        #         matchCoord = SkyCoord(ra=cf[0]*degree, dec=cf[1]*degree)
        #         idx, d2d, _ = matchCoord.match_to_catalog_sky(photSkyCoord[q])
        #         allCounts = add(allCounts,photFile[idx][4])
                
        if compFile.size ==2 and compFile.shape[0]==2:                
            matchCoord = SkyCoord(ra=compFile[0]*degree, dec=compFile[1]*degree)
        else:    
            matchCoord = SkyCoord(ra=compFile[:,0]*degree, dec=compFile[:,1]*degree)
        #print (compFile[:,0])
        #testStars=SkyCoord(ra = referenceFrame[:,0]*u.degree, dec = referenceFrame[:,1]*u.degree)

        idx, d2d, _ = matchCoord.match_to_catalog_sky(photSkyCoord[q])
        #rejectStars=where(d2d.arcsecond > acceptDistance)[0] 
        #print (photFile[idx,4])
        allCounts = add(allCounts,sum(photFile[idx,4]))
        
        #logger.debug("Total Counts in Image: {:.2f}".format(allCounts))
        fileCount.append(allCounts)
        q=q+1    
    
    # fileCount = []
    # for photFile in photFileArray:
    #     allCounts = 0.0
    #     fileRaDec = SkyCoord(ra=photFile[:,0]*degree, dec=photFile[:,1]*degree)
    #     if compFile.size ==2 and compFile.shape[0]==2:
    #         matchCoord = SkyCoord(ra=compFile[0]*degree, dec=compFile[1]*degree)
    #         idx, d2d, d3d = matchCoord.match_to_catalog_sky(fileRaDec)
    #         allCounts = add(allCounts,photFile[idx][4])
    #     else:
    #         for cf in compFile:
    #             matchCoord = SkyCoord(ra=cf[0]*degree, dec=cf[1]*degree)
    #             idx, d2d, d3d = matchCoord.match_to_catalog_sky(fileRaDec)
    #             allCounts = add(allCounts,photFile[idx][4])
    #     #logger.debug("Total Counts in Image: {:.2f}".format(allCounts))
    #     fileCount.append(allCounts)
    logger.debug("Total Ensemble Star Counts in Reference Frame {}".format(np.sum(np.array(fileCount))))
    return fileCount

def calculate_comparison_variation(compFile, photFileArray, fileCount, parentPath, photSkyCoord):
    stdCompStar=[]
    sortStars=[]
    logger.debug("Calculating Variation in Individual Comparisons")
        
    if compFile.size ==2 and compFile.shape[0]==2:
        compDiffMags = []
        #logger.debug("*************************")
        #logger.debug("RA : " + str(compFile[0]))
        #logger.debug("DEC: " + str(compFile[1]))
        matchCoord = SkyCoord(ra=compFile[0]*degree, dec=compFile[1]*degree)
        for q, photFile in enumerate(photFileArray):
            #fileRaDec = SkyCoord(ra=photFile[:,0]*degree, dec=photFile[:,1]*degree)
            
            idx, d2d, _ = matchCoord.match_to_catalog_sky(photSkyCoord[q])
            compDiffMags = append(compDiffMags,2.5 * log10(photFile[idx][4]/fileCount[q]))
            instrMags = -2.5 * log10(photFile[idx][4])

        
        stdCompDiffMags=std(compDiffMags)
        medCompDiffMags=np.nanmedian(compDiffMags)
        medInstrMags= np.nanmedian(instrMags)

        #logger.debug("VAR: " +str(stdCompDiffMags))
        if np.isnan(stdCompDiffMags) :
            logger.error("Star Variability non rejected")
            stdCompDiffMags=99
        stdCompStar.append(stdCompDiffMags)
        sortStars.append([compFile[0],compFile[1],stdCompDiffMags,medCompDiffMags,medInstrMags,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])



    else:
        # for cf in compFile:
        #     compDiffMags = []
        #     instrMags=[]
        #     #logger.debug("*************************")
        #     #logger.debug("RA : " + str(cf[0]))
        #     #logger.debug("DEC: " + str(cf[1]))
        #     matchCoord = SkyCoord(ra=cf[0]*degree, dec=cf[1]*degree)
        #     for q, photFile in enumerate(photFileArray):
        #         #fileRaDec = SkyCoord(ra=photFile[:,0]*degree, dec=photFile[:,1]*degree)
                
        #         idx, d2d, _ = matchCoord.match_to_catalog_sky(photSkyCoord[q])
        #         compDiffMags = append(compDiffMags,2.5 * log10(photFile[idx][4]/fileCount[q]))
        #         instrMags = -2.5 * log10(photFile[idx][4])

        #     stdCompDiffMags=std(compDiffMags)
        #     medCompDiffMags=np.nanmedian(compDiffMags)
        #     medInstrMags=np.nanmedian(instrMags)
            
        #     #print(stdCompDiffMags)
        #     #print(medCompDiffMags)
        #     #print(medInstrMags)

        #     #logger.debug("VAR: " +str(stdCompDiffMags))

        #     if np.isnan(stdCompDiffMags) :
        #         logger.error("Star Variability non rejected")
        #         stdCompDiffMags=99
        #     stdCompStar.append(stdCompDiffMags)

        #     sortStars.append([cf[0],cf[1],stdCompDiffMags,medCompDiffMags,medInstrMags,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

        # print (sortStars)


        sortStars=[]
        compDiffMags = []
        instrMags=[]
        matchCoord = SkyCoord(ra=compFile[:,0]*degree, dec=compFile[:,1]*degree)
        for q, photFile in enumerate(photFileArray):
            idx, d2d, _ = matchCoord.match_to_catalog_sky(photSkyCoord[q])
            compDiffMags.append(2.5 * log10(photFile[idx,4]/fileCount[q]))
            instrMags.append(-2.5 * log10(photFile[idx,4]))
                
        compDiffMags=array(compDiffMags)
        instrMags=array(instrMags)
        
        
        #print (len(compDiffMags[0]))
        
        #z=0
        sortStars=[]
        for z in range(len(compDiffMags[0])):
            #print (z)
            #print (compDiffMags[:,z])
            stdCompDiffMags=std(compDiffMags[:,z])
            medCompDiffMags=np.nanmedian(compDiffMags[:,z])
            medInstrMags=np.nanmedian(instrMags[:,z])
            
            if np.isnan(stdCompDiffMags) :
                logger.error("Star Variability non rejected")
                stdCompDiffMags=99
            stdCompStar.append(stdCompDiffMags)
            
            sortStars.append([compFile[z,0],compFile[z,1],stdCompDiffMags,medCompDiffMags,medInstrMags,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            
            #z=z+1
        #print (sortStars)
        # sys.exit()
        
        #     #compDiffMags = append(compDiffMags,2.5 * log10(photFile[idx,4]/fileCount[q]))
        # #compDiffMags = 2.5 * log10(photFile[idx,4]/fileCount[q])
        # #instrMags = -2.5 * log10(photFile[idx,4])
        
        # print (fileCount)
        
        # #print (compDiffMags)
        # #print (instrMags)
        
        

        # stdCompDiffMags=std(compDiffMags)
        # medCompDiffMags=np.nanmedian(compDiffMags)
        # medInstrMags=np.nanmedian(instrMags)
        
        # #print(stdCompDiffMags)
        # #print(medCompDiffMags)
        # #print(medInstrMags)

        # if np.isnan(stdCompDiffMags) :
        #     logger.error("Star Variability non rejected")
        #     stdCompDiffMags=99
        # stdCompStar.append(stdCompDiffMags)
        
        # sortStars.append([cf[0],cf[1],stdCompDiffMags,medCompDiffMags,medInstrMags,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        
        # print (sortStars)
        
        # sys.exit()
        #testStars=SkyCoord(ra = referenceFrame[:,0]*u.degree, dec = referenceFrame[:,1]*u.degree)
        #idx, d2d, _ = testStars.match_to_catalog_sky(photRAandDec)
        #rejectStars=where(d2d.arcsecond > acceptDistance)[0]



    # if compFile.size ==2 and compFile.shape[0]==2:
    #     compDiffMags = []
    #     #logger.debug("*************************")
    #     #logger.debug("RA : " + str(compFile[0]))
    #     #logger.debug("DEC: " + str(compFile[1]))
    #     matchCoord = SkyCoord(ra=compFile[0]*degree, dec=compFile[1]*degree)
    #     for q, photFile in enumerate(photFileArray):
    #         fileRaDec = SkyCoord(ra=photFile[:,0]*degree, dec=photFile[:,1]*degree)
            
    #         idx, d2d, d3d = matchCoord.match_to_catalog_sky(fileRaDec)
    #         compDiffMags = append(compDiffMags,2.5 * log10(photFile[idx][4]/fileCount[q]))
    #         instrMags = -2.5 * log10(photFile[idx][4])


    #     stdCompDiffMags=std(compDiffMags)
    #     medCompDiffMags=np.nanmedian(compDiffMags)
    #     medInstrMags= np.nanmedian(instrMags)

    #     #logger.debug("VAR: " +str(stdCompDiffMags))
    #     if np.isnan(stdCompDiffMags) :
    #         logger.error("Star Variability non rejected")
    #         stdCompDiffMags=99
    #     stdCompStar.append(stdCompDiffMags)
    #     sortStars.append([compFile[0],compFile[1],stdCompDiffMags,medCompDiffMags,medInstrMags,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])



    # else:
    #     for cf in compFile:
    #         compDiffMags = []
    #         instrMags=[]
    #         #logger.debug("*************************")
    #         #logger.debug("RA : " + str(cf[0]))
    #         #logger.debug("DEC: " + str(cf[1]))
    #         matchCoord = SkyCoord(ra=cf[0]*degree, dec=cf[1]*degree)
    #         for q, photFile in enumerate(photFileArray):
    #             fileRaDec = SkyCoord(ra=photFile[:,0]*degree, dec=photFile[:,1]*degree)
                
    #             idx, d2d, d3d = matchCoord.match_to_catalog_sky(fileRaDec)
    #             compDiffMags = append(compDiffMags,2.5 * log10(photFile[idx][4]/fileCount[q]))
    #             instrMags = -2.5 * log10(photFile[idx][4])

    #         stdCompDiffMags=std(compDiffMags)
    #         medCompDiffMags=np.nanmedian(compDiffMags)
    #         medInstrMags=np.nanmedian(instrMags)

    #         #logger.debug("VAR: " +str(stdCompDiffMags))

    #         if np.isnan(stdCompDiffMags) :
    #             logger.error("Star Variability non rejected")
    #             stdCompDiffMags=99
    #         stdCompStar.append(stdCompDiffMags)

    #         sortStars.append([cf[0],cf[1],stdCompDiffMags,medCompDiffMags,medInstrMags,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])

    return stdCompStar, sortStars

def remove_stars_targets(parentPath, compFile, acceptDistance, targetFile, removeTargets):
    max_sep=acceptDistance * arcsecond
    logger.info("Removing Target Stars from potential Comparisons")

    if not (compFile.shape[0] == 2 and compFile.size ==2):
        fileRaDec = SkyCoord(ra=compFile[:,0]*degree, dec=compFile[:,1]*degree)
    else:
        fileRaDec = SkyCoord(ra=compFile[0]*degree, dec=compFile[1]*degree)

    # Remove any nan rows from targetFile
    if targetFile is not None:
        targetRejecter=[]
        if not (targetFile.shape[0] == 4 and targetFile.size ==4):
            for z in range(targetFile.shape[0]):
              if isnan(targetFile[z][0]):
                targetRejecter.append(z)
            targetFile=delete(targetFile, targetRejecter, axis=0)

    # Get Average RA and Dec from file
    if compFile.shape[0] == 2 and compFile.size == 2:
        #logger.debug(compFile[0])
        #logger.debug(compFile[1])
        avgCoord=SkyCoord(ra=(compFile[0])*degree, dec=(compFile[1]*degree))

    else:
        #logger.debug(average(compFile[:,0]))
        #logger.debug(average(compFile[:,1]))

        #remarkably dumb way of averaging around zero RA and just normal if not
        resbelow= any(ele >350.0 and ele<360.0 for ele in compFile[:,0].tolist())
        resabove= any(ele >0.0 and ele<10.0 for ele in compFile[:,0].tolist())
        if resbelow and resabove:
            avgRAFile=[]
            for q in range(len(compFile[:,0])):
                if compFile[q,0] > 350:
                    avgRAFile.append(compFile[q,0]-360)
                else:
                    avgRAFile.append(compFile[q,0])
            avgRA=average(avgRAFile)
            if avgRA <0:
                avgRA=avgRA+360
            avgCoord=SkyCoord(ra=(avgRA*degree), dec=((average(compFile[:,1])*degree)))
        else:
            avgCoord=SkyCoord(ra=(average(compFile[:,0])*degree), dec=((average(compFile[:,1])*degree)))

        #logger.info(avgCoord)


    # Check VSX for any known variable stars and remove them from the list
    logger.info("Searching for known variable stars in VSX......")
    try:
        v=Vizier(columns=['all']) # Skymapper by default does not report the error columns
        v.ROW_LIMIT=-1
        #logger.info(avgCoord)
        variableResult=v.query_region(avgCoord, '0.33 deg', catalog='VSX')
        #logger.info(variableResult)
        if str(variableResult)=="Empty TableList":
            logger.info("VSX Returned an Empty Table.")
            varTable=0
        else:
            variableResult=variableResult['B/vsx/vsx']
            varTable=1
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

    if varTable==1:
        #logger.debug(variableResult)

        #logger.debug(variableResult.keys())

        raCat=array(variableResult['RAJ2000'].data)
        #logger.debug(raCat)
        decCat=array(variableResult['DEJ2000'].data)
        #logger.debug(decCat)
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

            if d2dcomp != 9999:
                if d2dcomp.arcsecond[0] < max_sep.value:
                    #logger.debug("match!")
                    varStarReject.append(idxcomp)

        logger.debug("Number of stars prior to VSX reject")
        logger.debug(compFile.shape[0])
        compFile=delete(compFile, varStarReject, axis=0)
        logger.debug("Number of stars post VSX reject")
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
    data = namedtuple(typename='data',field_names=['ra','dec','mag','emag','cat_name', 'colmatch', 'colerr'])

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
    if targets is not None:
        if targets.shape == (4,):
            targets = [targets]
        for tg in targets:
            resp = resp[where(np.abs(resp[radecname['ra']]-tg[0]) > 0.0014) and where(np.abs(resp[radecname['dec']]-tg[1]) > 0.0014)]
    
        logger.info("Number of calibration sources after removal of sources near targets: "+str(len(resp)))

    if len(resp) == 0:
        logger.info("Looks like your catalogue has too many sources next to nearby sources. ABORTING.")

    # Remove any star that has invalid values for mag or magerror
    if len(resp) != 0:
        catReject=[]
        
        #print (resp[opt['filter']])
        
        for q in range(len(resp)):
            if np.asarray(resp[opt['filter']][q]) == 0.0 or np.asarray(resp[opt['error']][q]) == 0.0 or np.isnan(resp[opt['filter']][q]) or np.isnan(resp[opt['error']][q]):
                catReject.append(q)
        del resp[catReject]
        logger.info(f"Stars rejected that are have invalid mag or magerror entries: {len(catReject)}")
    
        if len(resp) == 0:
            logger.info("Looks like your catalogue doesn't have any suitable comparison magnitudes. ABORTING.")
            

    # Remove any star from calibration catalogue that has another star in the catalogue within closerejectd arcseconds of it.
    if len(resp) != 0:
        catReject=[]
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
    data.colmatch = array(resp[opt['colmatch']].data)
    data.colerr = array(resp[opt['colerr']].data)

    return data

def find_comparisons_calibrated(targets, paths, filterCode, nopanstarrs=False, nosdss=False, colourdetect=False, linearise=False, closerejectd=5.0, max_magerr=0.05, stdMultiplier=2, variabilityMultiplier=2, colourTerm=0.0, colourError=0.0, restrictmagbrightest=-99.9, restrictmagdimmest=99.9, photCoordsFile=None, photFileHolder=None, calibSave=False):
    

    sys.stdout.write("⭐️ Find comparison stars in catalogues for calibrated photometry\n")

    FILTERS = {
                'B' : {'APASS' : {'filter' : 'Bmag', 'error' : 'e_Bmag', 'colmatch' : 'Vmag', 'colerr' : 'e_Vmag', 'colname' : 'B-V', 'colrev' : '0'}},
                'V' : {'APASS' : {'filter' : 'Vmag', 'error' : 'e_Vmag', 'colmatch' : 'Bmag', 'colerr' : 'e_Bmag', 'colname' : 'B-V', 'colrev' : '1'}},
                'CV' : {'APASS' : {'filter' : 'Vmag', 'error' : 'e_Vmag', 'colmatch' : 'Bmag', 'colerr' : 'e_Bmag', 'colname' : 'B-V', 'colrev' : '1'}},
                'up' : {'SDSS' : {'filter' : 'umag', 'error' : 'e_umag', 'colmatch' : 'gmag', 'colerr' : 'e_gmag', 'colname' : 'u-g', 'colrev' : '0'},
                        'SkyMapper' : {'filter' : 'uPSF', 'error' : 'e_uPSF', 'colmatch' : 'gPSF', 'colerr' : 'e_gPSF', 'colname' : 'u-g', 'colrev' : '0'}},
                'gp' : {'SDSS' : {'filter' : 'gmag', 'error' : 'e_gmag', 'colmatch' : 'rmag', 'colerr' : 'e_rmag', 'colname' : 'g-r', 'colrev' : '0'},
                        'SkyMapper' : {'filter' : 'gPSF', 'error' : 'e_gPSF', 'colmatch' : 'rPSF', 'colerr' : 'e_rPSF', 'colname' : 'g-r', 'colrev' : '0'},
                        'PanSTARRS': {'filter' : 'gmag', 'error' : 'e_gmag', 'colmatch' : 'rmag', 'colerr' : 'e_rmag', 'colname' : 'g-r', 'colrev' : '0'},
                        'APASS' : {'filter' : 'g_mag', 'error' : 'e_g_mag', 'colmatch' : 'r_mag', 'colerr' : 'e_r_mag', 'colname' : 'g-r', 'colrev' : '0'}},
                'rp' : {'SDSS' : {'filter' : 'rmag', 'error' : 'e_rmag', 'colmatch' : 'imag', 'colerr' : 'e_imag', 'colname' : 'r-i', 'colrev' : '0'},
                        'SkyMapper' : {'filter' : 'rPSF', 'error' : 'e_rPSF', 'colmatch' : 'iPSF', 'colerr' : 'e_iPSF', 'colname' : 'r-i', 'colrev' : '0'},
                        'PanSTARRS': {'filter' : 'rmag', 'error' : 'e_rmag', 'colmatch' : 'imag', 'colerr' : 'e_imag', 'colname' : 'r-i', 'colrev' : '0'},
                        'APASS' : {'filter' : 'r_mag', 'error' : 'e_r_mag', 'colmatch' : 'i_mag', 'colerr' : 'e_i_mag', 'colname' : 'r-i', 'colrev' : '0'}},
                'ip' : {'SDSS' : {'filter' : 'imag', 'error' : 'e_imag', 'colmatch' : 'rmag', 'colerr' : 'e_rmag', 'colname' : 'r-i', 'colrev' : '1'},
                        'SkyMapper' : {'filter' : 'iPSF', 'error' : 'e_iPSF', 'colmatch' : 'rPSF', 'colerr' : 'e_rPSF', 'colname' : 'r-i', 'colrev' : '1'},
                        'PanSTARRS': {'filter' : 'imag', 'error' : 'e_imag', 'colmatch' : 'rmag', 'colerr' : 'e_rmag', 'colname' : 'r-i', 'colrev' : '1'},
                        'APASS' : {'filter' : 'i_mag', 'error' : 'e_i_mag', 'colmatch' : 'r_mag', 'colerr' : 'e_r_mag', 'colname' : 'r-i', 'colrev' : '1'}},
                'zs' : {'PanSTARRS': {'filter' : 'zmag', 'error' : 'e_zmag', 'colmatch' : 'rmag', 'colerr' : 'e_rmag', 'colname' : 'r-zs', 'colrev' : '1'},
                        'SkyMapper' : {'filter' : 'zPSF', 'error' : 'e_zPSF', 'colmatch' : 'rPSF', 'colerr' : 'e_rPSF', 'colname' : 'r-zs', 'colrev' : '1'},
                        'SDSS' : {'filter' : 'zmag', 'error' : 'e_zmag', 'colmatch' : 'rmag', 'colerr' : 'e_rmag', 'colname' : 'r-zs', 'colrev' : '1'}},
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

    if filterCode == 'clear':
        filterCode = 'CV'

    logger.debug("Filter Set: " + filterCode)

    # Load compsused
    compFile = genfromtxt(parentPath / 'stdComps.csv', dtype=float, delimiter=',')
    #logger.debug(compFile.shape[0])

    if compFile.shape[0] == 13 and compFile.size == 13:
        compCoords=SkyCoord(ra=compFile[0]*degree, dec=compFile[1]*degree)
    else:
        compCoords=SkyCoord(ra=compFile[:,0]*degree, dec=compFile[:,1]*degree)

    # Get Average RA and Dec from file
    if compFile.shape[0] == 13 and compFile.size == 13:
        #logger.debug(compFile[0])
        #logger.debug(compFile[1])
        avgCoord=SkyCoord(ra=(compFile[0])*degree, dec=(compFile[1]*degree))

    else:
        #logger.debug(average(compFile[:,0]))
        #logger.debug(average(compFile[:,1]))


        #remarkably dumb way of averaging around zero RA and just normal if not
        resbelow= any(ele >350.0 and ele<360.0 for ele in compFile[:,0].tolist())
        resabove= any(ele >0.0 and ele<10.0 for ele in compFile[:,0].tolist())
        if resbelow and resabove:
            avgRAFile=[]
            for q in range(len(compFile[:,0])):
                if compFile[q,0] > 350:
                    avgRAFile.append(compFile[q,0]-360)
                else:
                    avgRAFile.append(compFile[q,0])
            avgRA=average(avgRAFile)
            if avgRA <0:
                avgRA=avgRA+360
            avgCoord=SkyCoord(ra=(avgRA*degree), dec=((average(compFile[:,1])*degree)))
        else:
            avgCoord=SkyCoord(ra=(average(compFile[:,0])*degree), dec=((average(compFile[:,1])*degree)))

        logger.debug(f"Average: RA {avgCoord.ra}, Dec {avgCoord.dec}")

        #avgCoord=SkyCoord(ra=(average(compFile[:,0]))*degree, dec=(average(compFile[:,1]))*degree)

    try:
        catalogues = FILTERS[filterCode]
    except IndexError:
        raise AstrosourceException(f"{filterCode} is not accepted at present")

    # Look up in online catalogues and make sure there are sufficient comparison stars

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
                    # If no results try next catalogue
                    #print (len(coords.ra))
                    if len(coords.ra) == 0:
                        coords=[]
                        raise AstrosourceException("Empty catalogue produced from catalogue call")
                        
                    
                    if coords.cat_name == 'PanSTARRS' or coords.cat_name == 'APASS':
                        max_sep=2.5 * arcsecond
                    else:
                        max_sep=1.5 * arcsecond

                    # The following commented out code is temporary code to manually access Skymapper DR3 and APASS10 prior to them becoming publically available through vizier.
                    # It will be removed in later versions.

                    # SKYMAPPER OVERRIDE
                    # max_sep=1.5 * arcsecond
                    # smapdr3=genfromtxt(parentPath / 'SkymapperDR3.csv', dtype=float, delimiter=',')
                    # coords = namedtuple(typename='data',field_names=['ra','dec','mag','emag','cat_name', 'colmatch', 'colerr'])
                    # coords.ra=smapdr3[:,0]
                    # coords.dec=smapdr3[:,1]
                    # print (opt['filter'])
                    # if opt['filter'] == 'uPSF':
                    #     coords.mag = smapdr3[:,2]
                    #     coords.emag = smapdr3[:,3]
                    #     coords.colmatch = smapdr3[:,4]
                    #     coords.colerr = smapdr3[:,5]
                    #     coords.colrev = 0

                    # if opt['filter'] == 'gPSF':
                    #     coords.mag = smapdr3[:,4]
                    #     coords.emag = smapdr3[:,5]
                    #     coords.colmatch = smapdr3[:,6]
                    #     coords.colerr = smapdr3[:,7]
                    #     coords.colrev = 0

                    # if opt['filter'] == 'rPSF':
                    #     coords.mag = smapdr3[:,6]
                    #     coords.emag = smapdr3[:,7]
                    #     coords.colmatch = smapdr3[:,8]
                    #     coords.colerr = smapdr3[:,9]
                    #     coords.colrev = 0

                    # if opt['filter'] == 'iPSF':
                    #     coords.mag = smapdr3[:,8]
                    #     coords.emag = smapdr3[:,9]
                    #     coords.colmatch = smapdr3[:,6]
                    #     coords.colerr = smapdr3[:,7]
                    #     coords.colrev = 1

                    # if opt['filter'] == 'zPSF':
                    #     coords.mag = smapdr3[:,10]
                    #     coords.emag = smapdr3[:,11]
                    #     coords.colmatch = smapdr3[:,8]
                    #     coords.colerr = smapdr3[:,9]
                    #     coords.colrev = 1

                    # APASS OVERRIDE
                    # max_sep=2.5 * arcsecond
                    # smapdr3=genfromtxt(parentPath / 'ap10.csv', dtype=float, delimiter=',')
                    # coords = namedtuple(typename='data',field_names=['ra','dec','mag','emag','cat_name', 'colmatch', 'colerr'])
                    # coords.ra=smapdr3[:,0]
                    # coords.dec=smapdr3[:,1]
                    # print (opt['filter'])

                    # if opt['filter'] == 'Bmag':
                    #     coords.mag = smapdr3[:,2]
                    #     coords.emag = smapdr3[:,3]
                    #     coords.colmatch = smapdr3[:,4]
                    #     coords.colerr = smapdr3[:,5]
                    #     coords.colrev = 0

                    # if opt['filter'] == 'Vmag':
                    #     coords.mag = smapdr3[:,4]
                    #     coords.emag = smapdr3[:,5]
                    #     coords.colmatch = smapdr3[:,2]
                    #     coords.colerr = smapdr3[:,3]
                    #     coords.colrev = 1

                    # if opt['filter'] == 'umag':
                    #     coords.mag = smapdr3[:,6]
                    #     coords.emag = smapdr3[:,7]
                    #     coords.colmatch = smapdr3[:,8]
                    #     coords.colerr = smapdr3[:,9]
                    #     coords.colrev = 0

                    # if opt['filter'] == 'gmag':
                    #     coords.mag = smapdr3[:,8]
                    #     coords.emag = smapdr3[:,9]
                    #     coords.colmatch = smapdr3[:,10]
                    #     coords.colerr = smapdr3[:,11]
                    #     coords.colrev = 0

                    # if opt['filter'] == 'rmag':
                    #     coords.mag = smapdr3[:,10]
                    #     coords.emag = smapdr3[:,11]
                    #     coords.colmatch = smapdr3[:,12]
                    #     coords.colerr = smapdr3[:,13]
                    #     coords.colrev = 0

                    # if opt['filter'] == 'imag':
                    #     coords.mag = smapdr3[:,12]
                    #     coords.emag = smapdr3[:,13]
                    #     coords.colmatch = smapdr3[:,10]
                    #     coords.colerr = smapdr3[:,11]
                    #     coords.colrev = 1

                    # if opt['filter'] == 'zmag':
                    #     coords.mag = smapdr3[:,14]
                    #     coords.emag = smapdr3[:,15]
                    #     coords.colmatch = smapdr3[:,12]
                    #     coords.colerr = smapdr3[:,13]
                    #     coords.colrev = 1

                    # Save catalogue search
                    catalogueOut=np.hstack(np.array([[coords.ra],[coords.dec],[coords.mag],[coords.emag],[coords.colmatch],[coords.colerr]]))
                    savetxt(parentPath / "catalogueSearch.csv", np.asarray(catalogueOut) , delimiter=",", fmt='%0.8f')
                    del catalogueOut

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
                            if not isnan(coords.mag[idxcomp]) and not isnan(coords.emag[idxcomp]):
                                if compFile.shape[0] ==13 and compFile.size ==13:
                                    calibStands.append([compFile[0],compFile[1],compFile[2],coords.mag[idxcomp],coords.emag[idxcomp],compFile[3],coords.colmatch[idxcomp],coords.colerr[idxcomp],compFile[4]])
                                else:
                                    calibStands.append([compFile[q][0],compFile[q][1],compFile[q][2],coords.mag[idxcomp],coords.emag[idxcomp],compFile[q][3],coords.colmatch[idxcomp],coords.colerr[idxcomp],compFile[q][4]])


                    ### remove stars that that brighter (--restrictmagbrighter) or dimmer (--restrictmagdimmer) than requested.
                    calibStandsReject=[]
                    if (asarray(calibStands).shape[0] != 9 and asarray(calibStands).size !=9) and calibStands != []:
                        for q in range(len(asarray(calibStands)[:,0])):

                            if (calibStands[q][3] > restrictmagdimmest) or (calibStands[q][3] < restrictmagbrightest):
                                calibStandsReject.append(q)
                                #logger.info(calibStands[q][3])

                        if len(calibStandsReject) != len(asarray(calibStands)[:,0]):
                            calibStands=delete(calibStands, calibStandsReject, axis=0)

                    logger.info('Removed ' + str(len(calibStandsReject)) + ' Calibration Stars for being too bright or too dim')

                    ### If looking for colour, remove those without matching colour information

                    calibStandsReject=[]

                    if (asarray(calibStands).shape[0] != 9 and asarray(calibStands).size !=9) and calibStands != []:
                        for q in range(len(asarray(calibStands)[:,0])):
                            reject=0
                            if colourdetect == True:
                                if np.isnan(calibStands[q][6]): # if no matching colour
                                    reject=1
                                elif calibStands[q][6] == 0:
                                    reject=1
                                elif np.isnan(calibStands[q][7]):
                                    reject=1
                                elif calibStands[q][7] == 0:
                                    reject=1
                            if np.isnan(calibStands[q][3]): # If no magnitude info
                                reject=1
                            elif calibStands[q][3] == 0:
                                reject=1
                            elif np.isnan(calibStands[q][4]):
                                reject=1
                            elif calibStands[q][4] == 0:
                                reject=1

                            if reject==1:
                                calibStandsReject.append(q)

                        if len(calibStandsReject) != len(asarray(calibStands)[:,0]):
                            calibStands=delete(calibStands, calibStandsReject, axis=0)

                    if asarray(calibStands).shape[0] != 0:
                        logger.info('Calibration Stars Identified below')
                        logger.info(asarray(calibStands))

                    # Get the set of least variable stars to use as a comparison to calibrate the files (to eventually get the *ACTUAL* standards
                    #logger.debug(asarray(calibStands).shape[0])
                    if asarray(calibStands).shape[0] == 0:
                        logger.info("We could not find a suitable match between any of your stars and the calibration catalogue")
                        coords=[]
                        raise AstrosourceException("There is no adequate match between this catalogue and your comparisons.")

                    if coords !=[] and calibStands !=[]:
                        cat_used=cat_name

        except AstrosourceException as e:
            logger.debug(e)

    if not coords or len(coords.ra)==0:
        raise AstrosourceException(f"Could not find coordinate match in any catalogues for {filterCode}")

    savetxt(parentPath / "calibStandsAll.csv", calibStands , delimiter=",", fmt='%0.8f')

    # Colour Term Calculations
    colname=(opt['colname'])
    colrev=int((opt['colrev']))
    colourPath = paths['parent'] / 'colourplots'
    if not colourPath.exists():
        os.makedirs(colourPath)

    #Colour Term Estimation routine
    if colourdetect == True and colourTerm == 0.0:

        # use a temporary calibStands array for colour terms
        arrayCalibStands=np.asarray(calibStands)

        # MAKE REFERENCE PRE-COLOUR PLOT AND COLOUR TERM ESTIMATE
        referenceFrame = genfromtxt(parentPath / 'referenceFrame.csv', dtype=float, delimiter=',')
        referenceFrame[:,5] = 1.0857 * (referenceFrame[:,5]/referenceFrame[:,4])
        referenceFrame[:,4]=-2.5 * np.log10(referenceFrame[:,4])

        photCoords=SkyCoord(ra=referenceFrame[:,0]*degree, dec=referenceFrame[:,1]*degree)
        colTemp=[]
        for q in range(len(arrayCalibStands[:,0])):
            if arrayCalibStands.size == 13 and arrayCalibStands.shape[0]== 13:
                calibCoord=SkyCoord(ra=arrayCalibStands[0]*degree,dec=arrayCalibStands[1]*degree)
                idx,d2d,_=calibCoord.match_to_catalog_sky(photCoords)
                if colrev == 1:
                    colTemp.append([arrayCalibStands[3],arrayCalibStands[4],referenceFrame[idx,4],referenceFrame[idx,5],arrayCalibStands[6]-arrayCalibStands[3],0])
                else:
                    colTemp.append([arrayCalibStands[3],arrayCalibStands[4],referenceFrame[idx,4],referenceFrame[idx,5],arrayCalibStands[3]-arrayCalibStands[6],0])
            else:
                calibCoord=SkyCoord(ra=arrayCalibStands[q][0]*degree,dec=arrayCalibStands[q][1]*degree)
                idx,d2d,_=calibCoord.match_to_catalog_sky(photCoords)
                if colrev == 1:
                    colTemp.append([arrayCalibStands[q,3],arrayCalibStands[q,4],referenceFrame[idx,4],referenceFrame[idx,5],arrayCalibStands[q,6]-arrayCalibStands[q,3],0])
                else:
                    colTemp.append([arrayCalibStands[q,3],arrayCalibStands[q,4],referenceFrame[idx,4],referenceFrame[idx,5],arrayCalibStands[q,3]-arrayCalibStands[q,6],0])
        colTemp=np.asarray(colTemp)


        # Outlier reject, simple sigma
        while True:
            calibStandsReject=[]
            tempmed = (np.median(colTemp[:,2]-colTemp[:,0]))
            tempstd = (np.std(colTemp[:,2]-colTemp[:,0]))
            for q in range(len(asarray(colTemp)[:,0])):
                if colTemp[q,2]-colTemp[q,0] > (tempmed + 2.5*tempstd):
                    calibStandsReject.append(q)
                if colTemp[q,2]-colTemp[q,0] < (tempmed - 2.5*tempstd):
                    calibStandsReject.append(q)
            if calibStandsReject == []:
                break
            colTemp=delete(colTemp, calibStandsReject, axis=0)

        # Pre-colour Reference plot
        plt.cla()
        fig = plt.gcf()
        outplotx=colTemp[:,4]
        outploty=colTemp[:,2]-colTemp[:,0]
        weights=1/(colTemp[:,3])
        linA = np.vstack([outplotx,np.ones(len(outplotx))]).T * np.sqrt(weights[:,np.newaxis])
        linB = outploty * np.sqrt(weights)
        sqsol = np.linalg.lstsq(linA,linB, rcond=None)
        m, c = sqsol[0]
        x, residuals, rank, s = sqsol

        plt.xlabel(colname + ' Catalogue Colour')
        plt.ylabel('Instrumental - Calibrated ' + str(filterCode) + ' Mag')
        plt.plot(outplotx,outploty,'bo')
        plt.plot(outplotx,m*outplotx+c,'r')
        plt.ylim(max(outploty)+0.05,min(outploty)-0.05,'k-')
        plt.xlim(min(outplotx)-0.05,max(outplotx)+0.05)
        plt.errorbar(outplotx, outploty, yerr=colTemp[:,3], fmt='-o', linestyle='None')
        plt.grid(True)
        plt.subplots_adjust(left=0.15, right=0.98, top=0.98, bottom=0.17, wspace=0.3, hspace=0.4)
        fig.set_size_inches(6,3)
        plt.savefig(parentPath / str("CalibrationSanityPlot_PreColour_Reference.png"))
        plt.savefig(parentPath / str("CalibrationSanityPlot_PreColour_Reference.eps"))

        logger.info('Estimated Colour Slope in Reference Frame: ' + str(m))

        # MAKE ALL PRE-COLOUR PLOTS

        logger.info('Estimating Colour Slope from all Frames...')


        fileList=[]
        for line in (parentPath / "usedImages.txt").read_text().strip().split('\n'):
            fileList.append(line.strip())

        z=0
        slopeHolder=[]
        zeroHolder=[]
        colourPath = paths['parent'] / 'colourplots'
        if not colourPath.exists():
            os.makedirs(colourPath)

        for filen in fileList:

            z=z+1
            photFrame = load(parentPath / filen)

            photFrame[:,5] = 1.0857 * (photFrame[:,5]/photFrame[:,4])
            photFrame[:,4]=-2.5 * np.log10(photFrame[:,4])

            photCoords=SkyCoord(ra=photFrame[:,0]*degree, dec=photFrame[:,1]*degree)
            colTemp=[]
            for q in range(len(arrayCalibStands[:,0])):
                if arrayCalibStands.size == 13 and arrayCalibStands.shape[0]== 13:
                    calibCoord=SkyCoord(ra=arrayCalibStands[0]*degree,dec=arrayCalibStands[1]*degree)
                    idx,d2d,_=calibCoord.match_to_catalog_sky(photCoords)
                    if colrev == 1:
                        colTemp.append([arrayCalibStands[3],arrayCalibStands[4],photFrame[idx,4],photFrame[idx,5],arrayCalibStands[6]-arrayCalibStands[3],0])
                    else:
                        colTemp.append([arrayCalibStands[3],arrayCalibStands[4],photFrame[idx,4],photFrame[idx,5],arrayCalibStands[3]-arrayCalibStands[6],0])
                else:
                    calibCoord=SkyCoord(ra=arrayCalibStands[q][0]*degree,dec=arrayCalibStands[q][1]*degree)
                    idx,d2d,_=calibCoord.match_to_catalog_sky(photCoords)
                    if colrev == 1:
                        colTemp.append([arrayCalibStands[q,3],arrayCalibStands[q,4],photFrame[idx,4],photFrame[idx,5],arrayCalibStands[q,6]-arrayCalibStands[q,3],0])
                    else:
                        colTemp.append([arrayCalibStands[q,3],arrayCalibStands[q,4],photFrame[idx,4],photFrame[idx,5],arrayCalibStands[q,3]-arrayCalibStands[q,6],0])
            colTemp=np.asarray(colTemp)

            # Outlier reject, simple sigma
            while True:
                calibStandsReject=[]
                tempmed = (np.median(colTemp[:,2]-colTemp[:,0]))
                tempstd = (np.std(colTemp[:,2]-colTemp[:,0]))
                for q in range(len(asarray(colTemp)[:,0])):
                    if colTemp[q,2]-colTemp[q,0] > (tempmed + 2.5*tempstd):
                        calibStandsReject.append(q)
                    if colTemp[q,2]-colTemp[q,0] < (tempmed - 2.5*tempstd):
                        calibStandsReject.append(q)
                if calibStandsReject == []:
                    break
                colTemp=delete(colTemp, calibStandsReject, axis=0)


            # Colour Term Reference plot
            plt.cla()
            fig = plt.gcf()
            outplotx=colTemp[:,4]
            outploty=colTemp[:,2]-colTemp[:,0]
            #Weighted fit, weighted by error in this dataset
            weights=1/(colTemp[:,3])
            linA = np.vstack([outplotx,np.ones(len(outplotx))]).T * np.sqrt(weights[:,np.newaxis])
            linB = outploty * np.sqrt(weights)
            sqsol = np.linalg.lstsq(linA,linB, rcond=None)
            m, c = sqsol[0]
            x, residuals, rank, s = sqsol

            plt.xlabel(colname + ' Catalogue Colour')
            plt.ylabel('Instrumental - Calibrated ' + str(filterCode) + ' Mag')
            plt.plot(outplotx,outploty,'bo')
            plt.plot(outplotx,m*outplotx+c,'r')
            plt.ylim(max(outploty)+0.05,min(outploty)-0.05,'k-')
            plt.xlim(min(outplotx)-0.05,max(outplotx)+0.05)
            plt.errorbar(outplotx, outploty, yerr=colTemp[:,3], fmt='-o', linestyle='None')
            plt.grid(True)
            plt.subplots_adjust(left=0.15, right=0.98, top=0.98, bottom=0.17, wspace=0.3, hspace=0.4)
            fig.set_size_inches(6,3)
            plt.savefig(colourPath / str("CalibrationSanityPlot_Colour_" + str(z) + "_Pre.png"))
            plt.savefig(colourPath  / str("CalibrationSanityPlot_Colour_" + str(z) + "_Pre.eps"))

            slopeHolder.append(m)
            zeroHolder.append(c)

        # Reject outliers in colour slope
        outReject=[]
        while True:
            outMed=np.median(slopeHolder)
            outStd=np.std(slopeHolder)
            for q in range(len(slopeHolder)):

                if (slopeHolder[q] >= (outMed + 3*outStd)) :
                    outReject.append(q)
                if (slopeHolder[q] <= (outMed - 3*outStd)) :
                    outReject.append(q)
            slopeHolder=delete(slopeHolder, outReject, axis=0)
            zeroHolder=delete(zeroHolder, outReject, axis=0)

            if outReject == []:
                break
            outReject = []

        logger.info('Median Estimated Colour Slope from all frames: ' + str(outMed) )
        logger.info('Estimated Colour Slope Standard Deviation from all frames: ' + str(outStd))
        logger.info('Number of frames used : ' + str(len(slopeHolder)))
        logger.info('Standard Error : ' + str(outStd / pow (len(slopeHolder),0.5)))

        with open(parentPath / "ColourCoefficientsESTIMATED.txt", "w") as f:
            f.write('Median Estimated Colour Slope from all frames: ' + str(outMed) +"\n")
            f.write('Estimated Colour Slope Standard Deviation from all frames: ' + str(outStd) +"\n")
            f.write('Number of frames used : ' + str(len(slopeHolder)) +"\n")
            f.write('Standard Error : ' + str(outStd / pow (len(slopeHolder),0.5)) +"\n")

        # Plot histogram of colour terms
        plt.cla()
        fig = plt.gcf()
        plt.hist(slopeHolder, bins=16)
        fig.set_size_inches(6,3)
        plt.xlabel(str(filterCode) + ' Colour Term')
        plt.ylabel('Number of images')
        plt.savefig(parentPath / str("CalibrationSanityPlot_ColourTermHistogram.png"))
        plt.savefig(parentPath / str("CalibrationSanityPlot_ColourTermHistogram.eps"))

        colourTerm=outMed
        colourError=outStd / pow (len(slopeHolder),0.5)

    else:
        logger.info("Skipping Colour Correction Estimation")



    # Get rid of higher variability stars from calibration list
    varimin=(min(asarray(calibStands)[:,2])) * variabilityMultiplier
    calibStandsReject=[]
    for q in range(len(asarray(calibStands)[:,0])):
        if calibStands[q][2] > varimin:
            calibStandsReject.append(q)
    calibStands=delete(calibStands, calibStandsReject, axis=0)

    #Get rid of outlier stars that typically represent systematically faulty calibration catalogue magnitudes
    while True:
        calibStandsReject=[]
        variavg=(np.average(asarray(calibStands)[:,3]+asarray(calibStands)[:,5]))
        varimin=(np.average((abs(asarray(calibStands)[:,3]+asarray(calibStands)[:,5]-variavg)))) * 2 # 3 stdevs to be a bit more conservative in the rejection
        for q in range(len(asarray(calibStands)[:,0])):
            if abs(asarray(abs((calibStands)[q,3]+asarray(calibStands)[q,5])-variavg)) > varimin:
                calibStandsReject.append(q)
        if calibStandsReject==[]:
            break
        else:
            calibStands=delete(calibStands, calibStandsReject, axis=0)

    savetxt(parentPath / "calibStands.csv", calibStands , delimiter=",", fmt='%0.8f')

    # Lets use this set to calibrate each datafile and pull out the calibrated compsused magnitudes
    compUsedFile = genfromtxt(parentPath / 'compsUsed.csv', dtype=float, delimiter=',')

    calibCompUsed=[]

    calibOverlord=[] # a huge array intended to create the calibration plot and data out of all the individual calibration files.

    calibStands=asarray(calibStands)

    z=0
    logger.debug("Calibrating each photometry file......")
    
    # clear calibcats directory and regenerate
    folders = ['calibcats']
    for fd in folders:
        if (paths['parent'] / fd).exists():
            shutil.rmtree(paths['parent'] / fd, ignore_errors=True )
            try:
                os.mkdir(paths['parent'] / fd)
            except OSError:
                print ("Creation of the directory %s failed" % paths['parent'])


    #print(datetime.now().strftime("%H:%M:%S"))
    slopeHolder=[]
    
    # REMEMBER photCoordsFile
    counter=0
    for file in fileList:
        
        #print ("**********************************")
        #print(datetime.now().strftime("%H:%M:%S"))
        #logger.debug(file)

        #Get the phot file into memory
        photFile = photFileHolder[counter]
        #photFile = load(parentPath / file)
        
        #photCoords=SkyCoord(ra=photFile[:,0]*degree, dec=photFile[:,1]*degree)
        photCoords=photCoordsFile[counter]
        
        counter=counter+1
        # Get colour information into photFile
        # adding in colour columns to photfile
        photFile=np.c_[photFile,np.zeros(len(photFile[:,0])),np.zeros(len(photFile[:,0])),np.zeros(len(photFile[:,0]))]

        #print ("Loading")
        #print(datetime.now().strftime("%H:%M:%S"))

        #print(catCoords.shape)
        #print(photCoords.shape)
        
        # for q in range(len(photFile[:,0])):
        #     #photCoord=SkyCoord(ra=photFile[q][0]*degree, dec=photFile[q][1]*degree)
        #     photCoord=photCoords[q]
        #     idx,d2d,_=photCoord.match_to_catalog_sky(catCoords)

        #     if d2d < max_sep:
        #         if colrev == 1:
        #             photFile[q,8]=coords.colmatch[idx]-coords.mag[idx]
        #         else:
        #             photFile[q,8]=coords.mag[idx]-coords.colmatch[idx]
        #         photFile[q,9]=pow(pow(coords.colerr[idx],2)+pow(coords.emag[idx],2),0.5)
        #         photFile[q,10]=1
        #     else:
        #         photFile[q,8]=np.nan
        #         photFile[q,9]=np.nan
        #         photFile[q,10]=0
        
        #for q in range(len(catCoords[:,0])):
        idx,d2d,_=catCoords.match_to_catalog_sky(photCoords)
        #print (idx)
        #print (d2d)
        
        #print (photFile[idx,8])
        
        photFile[:,8]=np.nan
        photFile[:,9]=np.nan
        photFile[:,10]=0
        
        rejectStars=where(d2d > max_sep)[0]
        #print (d2d)
        #print (max_sep)
        idx=delete(idx, rejectStars, axis=0)
                
        for q in range(len(idx)):
            #if d2d[q] < max_sep:
            if colrev == 1:
                photFile[idx[q],8]=coords.colmatch[q]-coords.mag[q]
            else:
                photFile[idx[q],8]=coords.mag[q]-coords.colmatch[q]
            photFile[idx[q],9]=pow(pow(coords.colerr[q],2)+pow(coords.emag[q],2),0.5)
            photFile[idx[q],10]=1                
        
        
        
        
        
        #print ("First For Loop")
        #print(datetime.now().strftime("%H:%M:%S"))

        #Replace undetected colours with average colour of the field
        photFile[:,8]=np.nan_to_num(photFile[:,8],nan=np.nanmedian(photFile[:,8]))
        photFile[:,9]=np.nan_to_num(photFile[:,9],nan=np.nanmedian(photFile[:,9]))


        #Convert the phot file into instrumental magnitudes with colour correction
        # for r in range(len(photFile[:,0])):
        #     photFile[r,5]=1.0857 * (photFile[r,5]/photFile[r,4])
        #     if not np.isnan(photFile[r,8]):
        #         photFile[r,4]=-2.5*log10(photFile[r,4]) -(colourTerm*photFile[r,8])
        #     else:
        #         photFile[r,4]=-2.5*log10(photFile[r,4])
        #         photFile[r,10]=2 # 2 means that there was no colour to use to embed the colour.

        photFile[:,5]=1.0857 * (photFile[:,5]/photFile[:,4])
        
        #notNanTemp=photFile[:,:][~np.isnan(photFile[:,8]).any(axis=1)]
        #NanTemp=photFile[:,:][np.isnan(photFile[:,8]).any(axis=1)]
        
        notNanTemp=photFile[np.isnan(photFile).any(axis=1)] #rows where any value is nan
        notNanTemp[:,4]=-2.5*log10(notNanTemp[:,4]) -(colourTerm*notNanTemp[:,8])
        
        
        NanTemp=photFile[~np.isnan(photFile).any(axis=1)] #rows where no value is nan
        NanTemp[:,4]=-2.5*log10(photFile[:,4])
        NanTemp[:,10]=2 # 2 means that there was no colour to use to embed the colour.
        
        photFile=np.concatenate([notNanTemp,NanTemp])
        del notNanTemp
        del NanTemp
           
        
        #print ("Nantemp")
        #print(datetime.now().strftime("%H:%M:%S"))


        #Pull out the CalibStands out of each file

        # tempDiff=[]
        # calibOut=[]
        # for q in range(len(calibStands[:,0])):
        #     if calibStands.size == 13 and calibStands.shape[0]== 13:
        #         calibCoord=SkyCoord(ra=calibStands[0]*degree,dec=calibStands[1]*degree)
        #         idx,d2d,_=calibCoord.match_to_catalog_sky(photCoords)
        #         if photFile[idx,10] != 0:
        #             tempDiff.append(calibStands[3]-(photFile[idx,4]))
        #             calibOut.append([calibStands[3],calibStands[4],photFile[idx,4],photFile[idx,5],calibStands[3]-(photFile[idx,4]),0,photFile[idx,8],photFile[idx,0],photFile[idx,1]])
        #     else:
        #         calibCoord=SkyCoord(ra=calibStands[q][0]*degree,dec=calibStands[q][1]*degree)
        #         idx,d2d,_=calibCoord.match_to_catalog_sky(photCoords)
        #         if photFile[idx,10] != 0:
        #             tempDiff.append(calibStands[q,3]-(photFile[idx,4]))
        #             calibOut.append([calibStands[q,3],calibStands[q,4],photFile[idx,4],photFile[idx,5],calibStands[q,3]-(photFile[idx,4]),0,photFile[idx,8],photFile[idx,0],photFile[idx,1]])
        # tempZP= (median(tempDiff))
        
        
        
        
        tempDiff=[]
        calibOut=[]
        if calibStands.size == 13 and calibStands.shape[0]== 13:
            calibCoord=SkyCoord(ra=calibStands[0]*degree,dec=calibStands[1]*degree)
        else:
            calibCoord=SkyCoord(ra=calibStands[:,0]*degree,dec=calibStands[:,1]*degree)         
        
        
        
        
        if calibStands.size == 13 and calibStands.shape[0]== 13:
            for q in range(len(calibStands[:,0])):        
                idx,d2d,_=calibCoord.match_to_catalog_sky(photCoords)
                if photFile[idx,10] != 0:
                    tempDiff.append(calibStands[3]-(photFile[idx,4]))
                    calibOut.append([calibStands[3],calibStands[4],photFile[idx,4],photFile[idx,5],calibStands[3]-(photFile[idx,4]),0,photFile[idx,8],photFile[idx,0],photFile[idx,1]])
        else:
        
            # for q in range(len(calibStands[:,0])):        
            #     idx,d2d,_=calibCoord[q].match_to_catalog_sky(photCoords)
            #     if photFile[idx,10] != 0:
            #         tempDiff.append(calibStands[q,3]-(photFile[idx,4]))
            #         calibOut.append([calibStands[q,3],calibStands[q,4],photFile[idx,4],photFile[idx,5],calibStands[q,3]-(photFile[idx,4]),0,photFile[idx,8],photFile[idx,0],photFile[idx,1]])
        
        
            idx,d2d,_=calibCoord.match_to_catalog_sky(photCoords)
            rejectStars=where(d2d > max_sep)[0]
            #print (d2d)
            #print (max_sep)
            idx=delete(idx, rejectStars, axis=0)
            #referenceFrame=delete(referenceFrame, rejectStars, axis=0)
            #print (idx)
            #print (rejectStars)
            #for r in range(len(idx)):
            for q in range(len(idx)):  
                if photFile[idx[q],10] != 0:
                    tempDiff.append(calibStands[q,3]-(photFile[idx[q],4]))
                    calibOut.append([calibStands[q,3],calibStands[q,4],photFile[idx[q],4],photFile[idx[q],5],calibStands[q,3]-(photFile[idx[q],4]),0,photFile[idx[q],8],photFile[idx[q],0],photFile[idx[q],1]])
        
                
                
                
            
            #sys.exit()
        
            #rejectStars=where((referenceFrame[:,4] < lowcounts) | (referenceFrame[:,4] > hicounts))[0]
            #referenceFrame=delete(referenceFrame, rejectStars, axis=0)
        
            # for q in range(len(calibStands[:,0])):        
            #     idx,d2d,_=calibCoord[q].match_to_catalog_sky(photCoords)
            #     if photFile[idx,10] != 0:
            #         tempDiff.append(calibStands[q,3]-(photFile[idx,4]))
            #         calibOut.append([calibStands[q,3],calibStands[q,4],photFile[idx,4],photFile[idx,5],calibStands[q,3]-(photFile[idx,4]),0,photFile[idx,8],photFile[idx,0],photFile[idx,1]])
        
        
        tempZP= (median(tempDiff))








        
        #print ("Second For Loop")
        #print(datetime.now().strftime("%H:%M:%S"))

        #Shift the magnitudes in the phot file by the zeropoint
        photFile[:,4]=photFile[:,4]+tempZP # Speedup


        calibOut=asarray(calibOut)

        #Shift the magnitudes in the phot file by the zeropoint
        #for r in range(len(calibOut[:,0])):
        #    calibOut[r,5]=calibOut[r,4]-tempZP
        #    calibOverlord.append([calibOut[r,0],calibOut[r,1],calibOut[r,2],calibOut[r,3],calibOut[r,4],calibOut[r,5],float(file.split("_")[2].replace("d",".")),tempZP,calibOut[r,6],calibOut[r,7],calibOut[r,8]])
        calibOut[:,5]=calibOut[:,4]-tempZP # Speedup
        for r in range(len(calibOut[:,0])):
            
            calibOverlord.append([calibOut[r,0],calibOut[r,1],calibOut[r,2],calibOut[r,3],calibOut[r,4],calibOut[r,5],float(file.split("_")[2].replace("d",".")),tempZP,calibOut[r,6],calibOut[r,7],calibOut[r,8]])

        
        #print ("Last For Loop")
        #print(datetime.now().strftime("%H:%M:%S"))

        calibOut=asarray(calibOut)
        #print (calibOut)

        


        
        #Save the calibrated photfiles to the calib directory
        if calibSave == True:
            #print ("SPLAY")
            file = Path(file)
            savetxt(calibPath / "{}.calibrated.{}".format(file.stem, 'csv'), photFile, delimiter=",", fmt='%0.8f')
            savetxt(calibPath / "{}.compared.{}".format(file.stem, 'csv'), calibOut, delimiter=",", fmt='%0.8f')


        #PRINT POSTCOLOUR CORRECTION PLOTS
        if colourdetect == True and (not np.all(np.isnan(photFile[:,8]))):
            # Colour Term Reference plot
            plt.cla()
            fig = plt.gcf()
            outplotx=calibOut[:,6]
            outploty=calibOut[:,2]-calibOut[:,0]
            #Weighted fit, weighted by error in this dataset
            weights=1/(calibOut[:,1])
            linA = np.vstack([outplotx,np.ones(len(outplotx))]).T * np.sqrt(weights[:,np.newaxis])
            linB = outploty * np.sqrt(weights)
            sqsol = np.linalg.lstsq(linA,linB, rcond=None)
            m, c = sqsol[0]
            x, residuals, rank, s = sqsol

            plt.xlabel(colname + ' Catalogue Colour')
            plt.ylabel('Instrumental - Calibrated ' + str(filterCode) + ' Mag')
            plt.plot(outplotx,outploty,'bo')
            plt.plot(outplotx,m*outplotx+c,'r')
            plt.ylim(max(outploty)+0.05,min(outploty)-0.05,'k-')
            plt.xlim(min(outplotx)-0.05,max(outplotx)+0.05)
            plt.errorbar(outplotx, outploty, yerr=calibOut[:,1], fmt='-o', linestyle='None')
            plt.grid(True)
            plt.subplots_adjust(left=0.15, right=0.98, top=0.98, bottom=0.17, wspace=0.3, hspace=0.4)
            fig.set_size_inches(6,3)
            plt.savefig(colourPath / str("CalibrationSanityPlot_Colour_" + str(z) + "_Post.png"))
            plt.savefig(colourPath  / str("CalibrationSanityPlot_Colour_" + str(z) + "_Post.eps"))
            z=z+1
            slopeHolder.append(m)

        #Look within photfile for ACTUAL usedcomps.csv and pull them out
        lineCompUsed=[]
        if compUsedFile.shape[0] ==3 and compUsedFile.size == 3:
            lenloop=1
        else:
            lenloop=len(compUsedFile[:,0])

        for r in range(lenloop):
            if compUsedFile.shape[0] ==3 and compUsedFile.size ==3:
                compUsedCoord=SkyCoord(ra=compUsedFile[0]*degree,dec=compUsedFile[1]*degree)
            else:
                compUsedCoord=SkyCoord(ra=compUsedFile[r][0]*degree,dec=compUsedFile[r][1]*degree)
            idx,d2d,_=compUsedCoord.match_to_catalog_sky(photCoords)
            lineCompUsed.append(photFile[idx,4])

        calibCompUsed.append(lineCompUsed)
        sys.stdout.write('.')
        sys.stdout.flush()
    
    #print(datetime.now().strftime("%H:%M:%S"))

    # Reject outliers in colour slope
    if colourdetect == True:
        outReject=[]
        while True:
            outMed=np.median(slopeHolder)
            outStd=np.std(slopeHolder)
            for q in range(len(slopeHolder)):
                if (slopeHolder[q] >= (outMed + 3*outStd)) :
                    outReject.append(q)
                if (slopeHolder[q] <= (outMed - 3*outStd)) :
                    outReject.append(q)
            slopeHolder=delete(slopeHolder, outReject, axis=0)

            if outReject == []:
                break
            outReject = []

        logger.info('Median Estimated Colour CORRECTED Slope from all frames: ' + str(outMed) )
        logger.info('Estimated Colour CORRECTED Slope Standard Deviation from all frames: ' + str(outStd))
        logger.info('Number of frames used : ' + str(len(slopeHolder)))
        logger.info('Standard Error : ' + str(outStd / pow (len(slopeHolder),0.5)))

        with open(parentPath / "ColourCoefficientsFINAL.txt", "w") as f:
            f.write('Median Estimated Colour CORRECTED Slope from all frames: ' + str(outMed) +"\n")
            f.write('Estimated Colour CORRECTED Slope Standard Deviation from all frames: ' + str(outStd) +"\n")
            f.write('Number of frames used : ' + str(len(slopeHolder)) +"\n")
            f.write('Standard Error : ' + str(outStd / pow (len(slopeHolder),0.5)) +"\n")

        # Update colour error
        colourError=outStd / pow (len(slopeHolder),0.5)

        # Plot histogram of colour terms
        plt.cla()
        fig = plt.gcf()
        plt.hist(slopeHolder, bins=16)
        fig.set_size_inches(6,3)
        plt.xlabel(str(filterCode) + ' Colour Term')
        plt.ylabel('Number of images')
        plt.savefig(parentPath / str("CalibrationSanityPlot_CORRECTEDColourTermHistogram.png"))
        plt.savefig(parentPath / str("CalibrationSanityPlot_CORRECTEDColourTermHistogram.eps"))

    calibOverlord=asarray(calibOverlord)
    savetxt(parentPath / "CalibAll.csv", calibOverlord, delimiter=",", fmt='%0.8f')


    logger.info("*********************")
    logger.info("Calculating Linearity")

    try:
        # Difference versus Magnitude calibration plot
        plt.cla()
        fig = plt.gcf()
        outplotx=calibOverlord[:,0]
        outploty=calibOverlord[:,5]
        sqsol = np.linalg.lstsq(np.vstack([calibOverlord[:,0],np.ones(len(calibOverlord[:,0]))]).T,calibOverlord[:,5], rcond=None)
        m, c = sqsol[0]
        x, residuals, rank, s = sqsol

        plt.xlabel(str(cat_used) + ' ' +str(filterCode) + ' Catalogue Magnitude')
        plt.ylabel('Calibrated - Catalogue Magnitude')
        plt.plot(outplotx,outploty,'bo')
        plt.plot(outplotx,m*outplotx+c,'r')

        plt.ylim(min(outploty)-0.05,max(outploty)+0.05,'k-')
        plt.xlim(min(outplotx)-0.05,max(outplotx)+0.05)
        plt.grid(True)
        plt.subplots_adjust(left=0.15, right=0.98, top=0.98, bottom=0.17, wspace=0.3, hspace=0.4)
        fig.set_size_inches(6,3)
        plt.savefig(parentPath / str("CalibrationSanityPlot_Magnitude.png"))
        plt.savefig(parentPath / str("CalibrationSanityPlot_Magnitude.eps"))

        with open(parentPath / "CalibrationSanityPlotCoefficients.txt", "w+") as f:
            f.write("Magnitude slope     : " + str(m)+"\n")
            f.write("Magnitude zeropoint : " + str(c) +"\n")
            if not residuals.size == 0:
                f.write("Magnitude residuals : " +str(residuals[0])+"\n")
            else:
                f.write("Magnitude residuals not calculated. \n")

        catalogueNonLinearSlope=m
        catalogueNonLinearZero=c
        catalogueNonLinearError=residuals[0]
        logger.info("Magnitude slope     : " + str(m)+"\n")
        logger.info("Magnitude zeropoint : " + str(c) +"\n")
        if not residuals.size == 0:
            logger.info("Magnitude residuals : " +str(residuals[0])+"\n")
        else:
            logger.info("Magnitude residuals not calculated. \n")
    except:
        logger.info("Could not create Difference versus Magnitude calibration plot")


    # Non-linearity correction routine
    if linearise == False:
        logger.info("Skipping Non-linearity correction routine")
    if linearise == True:

        calibOverlordNonLinear=[]
        for q in range (len(calibOverlord[:,0])):
            calibOverlordNonLinear.append([calibOverlord[q,0],calibOverlord[q,1],calibOverlord[q,2],calibOverlord[q,3],calibOverlord[q,4],calibOverlord[q,5],calibOverlord[q,6],calibOverlord[q,7],calibOverlord[q,2]+calibOverlord[q,7],(calibOverlord[q,2]+calibOverlord[q,7])-calibOverlord[q,0],0,0,0])

        calibOverlordNonLinear=asarray(calibOverlordNonLinear)

        # Calculate correction for unknown calibration values
        plt.cla()
        fig = plt.gcf()
        sqsol = np.linalg.lstsq(np.vstack([calibOverlordNonLinear[:,8],np.ones(len(calibOverlordNonLinear[:,0]))]).T,(calibOverlordNonLinear[:,9]), rcond=None)
        m, c = sqsol[0]
        x, residuals, rank, s = sqsol
        nonlinearSlope=m
        nonlinearZero=c
        nonlinearError=residuals[0] / pow(len(fileList), 0.5)


        with open(parentPath / "CalibrationSanityPlotCoefficients.txt", "a+") as f:
            f.write("Corrected Magnitude slope     : " + str(m)+"\n")
            f.write("Corrected Magnitude zeropoint : " + str(c) +"\n")
            if not residuals.size == 0:
                f.write("Corrected Magnitude residuals : " +str(residuals[0])+"\n")
            else:
                f.write("Magnitude residuals not calculated. \n")

        logger.info("Corrected Magnitude slope     : " + str(m))
        logger.info("Corrected Magnitude zeropoint : " + str(c))

        #Correct Non-linear Overlord for nonlinearity
        for q in range (len(calibOverlordNonLinear[:,0])):
            calibOverlordNonLinear[q,10]=nonlinearSlope*calibOverlordNonLinear[q,8]+nonlinearZero
            calibOverlordNonLinear[q,11]=calibOverlordNonLinear[q,8]-calibOverlordNonLinear[q,10]
            calibOverlordNonLinear[q,12]=calibOverlordNonLinear[q,11]-calibOverlordNonLinear[q,0]


        for q in range(len(calibOverlord[:,0])):
            # Fix value
            calibOverlord[q,4]=calibOverlord[q,4]-(catalogueNonLinearSlope*calibOverlord[q,0]+catalogueNonLinearZero)

            calibOverlord[q,5]=calibOverlord[q,5]-(catalogueNonLinearSlope*calibOverlord[q,0]+catalogueNonLinearZero)

        # Nonlinear CORRECTED plot
        plt.cla()
        fig = plt.gcf()
        outplotx=calibOverlord[:,0]
        outploty=calibOverlord[:,5]
        sqsol = np.linalg.lstsq(np.vstack([calibOverlord[:,0],np.ones(len(calibOverlord[:,0]))]).T,calibOverlord[:,5], rcond=None)
        m, c = sqsol[0]
        x, residuals, rank, s = sqsol

        plt.xlabel(str(cat_used) + ' ' +str(filterCode) + ' Catalogue Magnitude')
        plt.ylabel('Calculated - Catalogue Magnitude')
        plt.plot(outplotx,outploty,'bo')
        plt.plot(outplotx,m*outplotx+c,'r')

        plt.ylim(min(outploty)-0.05,max(outploty)+0.05,'k-')
        plt.xlim(min(outplotx)-0.05,max(outplotx)+0.05)
        plt.grid(True)
        plt.subplots_adjust(left=0.15, right=0.98, top=0.98, bottom=0.17, wspace=0.3, hspace=0.4)
        fig.set_size_inches(6,3)
        plt.savefig(parentPath / str("CalibrationSanityPlotLinearityCorrected_Magnitude.png"))
        plt.savefig(parentPath / str("CalibrationSanityPlotLinearityCorrected_Magnitude.eps"))

        # Add correction into calibrated files
        z=0
        logger.debug("CORRECTING EACH FILE FOR NONLINEARITY")
        correctFileList = calibPath.glob("*.calibrated.csv")
        for file in correctFileList:
            #logger.debug(file)

            # NEED TO FIX UP COMPARED!

            photFile = genfromtxt(file, dtype=float, delimiter=',')

            for q in range(len(photFile[:,0])):
                photFile[q,4]=photFile[q,4] - nonlinearSlope*photFile[q,4]+nonlinearZero

            savetxt(file, photFile, delimiter=",", fmt='%0.8f')

    # Difference vs time calibration plot
    try:
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
        plt.ylim(min(outploty)-0.05,max(outploty)+0.05,'k-')
        plt.xlim(min(outplotx)-0.05,max(outplotx)+0.05)
        plt.grid(True)
        plt.subplots_adjust(left=0.15, right=0.98, top=0.98, bottom=0.17, wspace=0.3, hspace=0.4)
        fig.set_size_inches(6,3)
        plt.savefig(parentPath / str("CalibrationSanityPlot_Time.png"))
        plt.savefig(parentPath / str("CalibrationSanityPlot_Time.eps"))

        with open(parentPath / "CalibrationSanityPlotCoefficients.txt", "a+") as f:
            f.write("Time slope     : " + str(m)+"\n")
            f.write("Time zeropoint : " + str(c) +"\n")
            if not residuals.size == 0:
                f.write("Time residuals : " +str(residuals[0])+"\n")
            else:
                f.write("Time residuals not calculated. \n")
    except:
        logger.info("Could not create Difference versus BJD calibration plot")

    # Finalise calibcompsusedfile
    calibCompUsed=asarray(calibCompUsed)

    finalCompUsedFile=[]
    sumStd=[]
    for r in range(len(calibCompUsed[0,:])):
        #Calculate magnitude and stdev
        sumStd.append(std(calibCompUsed[:,r]))

        if compUsedFile.shape[0] ==3  and compUsedFile.size ==3:
            finalCompUsedFile.append([compUsedFile[0],compUsedFile[1],compUsedFile[2],median(calibCompUsed[:,r]),asarray(calibStands[0])[4]])
        else:
            finalCompUsedFile.append([compUsedFile[r][0],compUsedFile[r][1],compUsedFile[r][2],median(calibCompUsed[:,r]),std(calibCompUsed[:,r])])

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

    compFile = asarray(finalCompUsedFile)
    savetxt(parentPath / "calibCompsUsed.csv", compFile, delimiter=",", fmt='%0.8f')
    sys.stdout.write('\n')

    return colourTerm, colourError, compFile
