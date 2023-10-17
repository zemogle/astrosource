from numpy import genfromtxt, savetxt, load, delete, asarray, multiply, log10, divide, \
    less, append, add, std, average, median, inf, nan, isnan, nanstd, nanmean, array
import numpy as np
from astropy.units import degree
from astropy.coordinates import SkyCoord
import glob
import sys
from pathlib import Path
import shutil
import time
import math
import os
from tqdm import tqdm

import logging

#from astrosource.utils import photometry_files_to_array, AstrosourceException
from astrosource.utils import AstrosourceException
from astrosource.plots import plot_variability

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors

logger = logging.getLogger('astrosource')


def get_total_counts(photFileArray, compFile, loopLength, photCoords):

    compArray = []
    allCountsArray = []
    logger.debug("***************************************")
    logger.debug("Calculating total counts")
    counter=0
    for photFile in photFileArray:
        allCounts = 0.0
        allCountsErr = 0.0
        #fileRaDec = SkyCoord(ra=photFile[:, 0]*degree, dec=photFile[:, 1]*degree)
        fileRaDec = photCoords[counter]
        counter=counter+1
        #Array of comp measurements
        for j in range(loopLength):
            if compFile.size == 2 or (compFile.shape[0]== 3 and compFile.size ==3) or (compFile.shape[0]== 5 and compFile.size ==5):
                matchCoord = SkyCoord(ra=compFile[0]*degree, dec=compFile[1]*degree)
            else:
                matchCoord = SkyCoord(ra=compFile[j][0]*degree, dec=compFile[j][1]*degree)

            idx, d2d, d3d = matchCoord.match_to_catalog_sky(fileRaDec)
            allCounts = add(allCounts, photFile[idx][11])
            allCountsErr = add(allCountsErr, photFile[idx][5])
            if (compFile.shape[0] == 5 and compFile.size == 5) or (compFile.shape[0] == 3 and compFile.size == 3):
                break
        allCountsArray.append([allCounts, allCountsErr])
    #logger.debug(allCountsArray)
    return allCountsArray

def find_variable_stars(targets, matchRadius, errorReject=0.05, parentPath=None, varsearchglobalstdev=-99.9, varsearchthresh=10000, varsearchstdev=2.0, varsearchmagwidth=0.25, varsearchminimages=0.3, photCoords=None, photFileHolder=None, fileList=None):
    '''
    Find stable comparison stars for the target photometry and remove variables

    Parameters
    ----------
    targetStars : list
            List of target tuples in the formal (ra, dec, 0, 0). ra and dec must be in decimal
    errorReject : float
        reject measurements with instrumental errors larger than this (this is not total error, just the estimated error in the single measurement of the variable)
    matchRadius : float
        Furthest distance in arcseconds for matches

    Returns
    -------
    outfile : str
    '''


    photFileArray=photFileHolder
    photFileCoords=photCoords

    # allocate minimum images to detect
    minimumNoOfObs=int(varsearchminimages*len(fileList))
    logger.info("Minimum number of observations to detect: " + str(minimumNoOfObs))

    # LOAD IN COMPARISON FILE
    preFile = genfromtxt(parentPath / 'results/stdComps.csv', dtype=float, delimiter=',')

    if preFile.shape[0] != 13:
        preFile=(preFile[preFile[:, 2].argsort()])

    # GET REFERENCE IMAGE
    # Sort through and find the largest file and use that as the reference file
    fileSizer = 0
    logger.debug("Finding image with most stars detected")
    for photFile in photFileArray:
        if photFile.size > fileSizer:
            referenceFrame = photFile
            fileSizer = photFile.size

    compFile = genfromtxt(parentPath / "results/compsUsed.csv", dtype=float, delimiter=',')
    logger.debug("Stable Comparison Candidates below variability threshold")
    outputPhot = []

    # Get total counts for each file
    allCountsArray = get_total_counts(photFileArray, compFile, loopLength=compFile.shape[0], photCoords=photCoords)

    # Define targetlist as every star in referenceImage above a count threshold
    logger.info("Setting up Variable Search List")
    targetFile = referenceFrame
    # Although remove stars that are below the variable countrate
    starReject=[]
    for q in range(targetFile.shape[0]):
        if targetFile[q][11] < varsearchthresh:
            starReject.append(q)
    logger.info("Total number of stars in reference Frame: {}".format(targetFile.shape[0]))
    targetFile = delete(targetFile, starReject, axis=0)
    logger.info("Total number of stars with sufficient counts: {}".format(targetFile.shape[0]))

    ## NEED TO REMOVE COMPARISON STARS FROM TARGETLIST

    allcountscount=0
    # For each variable calculate the variability
    outputVariableHolder=[]

    logger.info("Measuring variability of stars...... ")
    taketime=time.time()

    for target in targetFile:
        diffMagHolder=[]
        allcountscount=0

        
        for photFile in photFileArray:
            # A bit rougher than using SkyCoord, but way faster
            # The amount of calculations is too slow for SkyCoord
            idx=(np.abs(photFile[:,0] - target[0]) + np.abs(photFile[:,1] - target[1])).argmin()
            d2d=pow(pow(photFile[idx,0] - target[0],2) + pow(photFile[idx,0] - target[0],2),0.5) * 3600
            multTemp=(multiply(-2.5,log10(divide(photFile[idx][11],allCountsArray[allcountscount][0]))))
            if less(d2d, matchRadius) and (multTemp != inf) :
                diffMagHolder=append(diffMagHolder,multTemp)
            allcountscount=add(allcountscount,1)
            
        ## REMOVE MAJOR OUTLIERS FROM CONSIDERATION
        diffMagHolder=np.array(diffMagHolder)
        while True:
            stdVar=std(diffMagHolder)
            avgVar=average(diffMagHolder)

            sizeBefore=diffMagHolder.shape[0]
            #print (sizeBefore)

            diffMagHolder[diffMagHolder > avgVar+(4*stdVar) ] = np.nan
            diffMagHolder[diffMagHolder < avgVar-(4*stdVar) ] = np.nan
            diffMagHolder=diffMagHolder[~np.isnan(diffMagHolder)]

            if diffMagHolder.shape[0] == sizeBefore:
                break

        if (diffMagHolder.shape[0] > minimumNoOfObs):
            outputVariableHolder.append( [target[0],target[1],median(diffMagHolder), std(diffMagHolder), diffMagHolder.shape[0]])

    print ("Star Variability done in " + str(time.time()-taketime))

    savetxt(parentPath / "results/starVariability.csv", outputVariableHolder, delimiter=",", fmt='%0.8f')

    ## Routine that actually pops out potential variables.
    starVar = np.asarray(outputVariableHolder)

    
    meanMags = starVar[:,2]
    variations = starVar[:,3]

    xStepSize= varsearchmagwidth
    yStepSize=0.02
    xbins = np.arange(np.min(meanMags), np.max(meanMags), xStepSize)
    ybins = np.arange(np.min(variations), np.max(variations), yStepSize)

    #split it into one array and identify variables in bins with centre
    variationsByMag=[]
    potentialVariables=[]
    for xbinner in range(len (xbins)):

        starsWithin=[]
        for q in range(len(meanMags)):
            if meanMags[q] >= xbins[xbinner] and meanMags[q] < xbins[xbinner]+xStepSize:
                starsWithin.append(variations[q])

        meanStarsWithin= (np.mean(starsWithin))
        stdStarsWithin= (np.std(starsWithin))
        variationsByMag.append([xbins[xbinner]+0.5*xStepSize,meanStarsWithin,stdStarsWithin])

        # At this point extract RA and Dec of stars that may be variable
        for q in range(len(starVar[:,2])):
            if starVar[q,2] >= xbins[xbinner] and starVar[q,2] < xbins[xbinner]+xStepSize:
                if varsearchglobalstdev != -99.9:
                    if starVar[q,3] > varsearchglobalstdev :
                        potentialVariables.append([starVar[q,0],starVar[q,1],starVar[q,2],starVar[q,3]])
                elif starVar[q,3] > (meanStarsWithin + varsearchstdev*stdStarsWithin):
                    #print (starVar[q,3])
                    potentialVariables.append([starVar[q,0],starVar[q,1],starVar[q,2],starVar[q,3]])

    potentialVariables=np.array(potentialVariables)
    logger.debug("Potential Variables Identified: " + str(potentialVariables.shape[0]))

    if potentialVariables.shape[0] == 0:
        logger.info("No Potential Variables identified in this set of data using the parameters requested.")
    else:
        savetxt(parentPath / "results/potentialVariables.csv", potentialVariables , delimiter=",", fmt='%0.8f')

        plot_variability(outputVariableHolder, potentialVariables, parentPath, compFile)

        plt.cla()
        fig, ax = plt.subplots(figsize =(10, 7))
        plt.hist2d(meanMags, variations,  bins =[xbins, ybins], norm=colors.LogNorm(), cmap = plt.cm.YlOrRd)
        plt.colorbar()
        plt.title("Variation Histogram")
        ax.set_xlabel('Mean Differential Magnitude')
        ax.set_ylabel('Variation (Standard Deviation)')
        plt.plot(potentialVariables[:,2],potentialVariables[:,3],'bo')
        plt.tight_layout()

        plt.savefig(parentPath / "results/Variation2DHistogram.png")

    return outputVariableHolder

def photometric_calculations(targets, paths, targetRadius, errorReject=0.1, filesave=True, outliererror=4, outlierstdev=4,photCoordsFile=None, photFileArray=None, fileList=None):

    fileCount=[]
    photometrydata = []
    sys.stdout.write('🖥 Starting photometric calculations\n')

    if (paths['parent'] / 'results/calibCompsUsed.csv').exists():
        logger.debug("Calibrated")
        compFile=genfromtxt(paths['parent'] / 'results/calibCompsUsed.csv', dtype=float, delimiter=',')
        calibFlag=1
    else:
        logger.debug("Differential")
        compFile=genfromtxt(paths['parent'] / 'results/compsUsed.csv', dtype=float, delimiter=',')
        calibFlag=0

    # clear output plots, cats and period fodlers and regenerate
    folders = ['periods', 'checkplots', 'eelbs', 'outputcats','outputplots','trimcats']
    for fd in folders:
        if (paths['parent'] / fd).exists():
            shutil.rmtree(paths['parent'] / fd, ignore_errors=True )
            time.sleep(0.1)
            try:
                os.mkdir(paths['parent'] / fd)
            except OSError:
                print ("Creation of the directory %s failed" % paths['parent'])

    # Get total counts for each file
    if compFile.shape[0]== 5 and compFile.size ==5:
        loopLength=1
    else:
        loopLength=compFile.shape[0]
    allCountsArray = get_total_counts(photFileArray, compFile, loopLength, photCoords=photCoordsFile)

    allcountscount=0

    if len(targets)== 4 and targets.size == 4:
        loopLength=1
    else:
        loopLength=targets.shape[0]

    # For each variable calculate all the things
    for q in range(loopLength):
        starErrorRejCount=0
        starDistanceRejCount=0
        logger.debug("****************************")
        logger.debug("Processing Variable {}".format(q+1))
        if int(len(targets)) == 4 and targets.size==4:
            logger.debug("RA {}".format(targets[0]))
        else:
            logger.debug("RA {}".format(targets[q][0]))
        if int(len(targets)) == 4 and targets.size==4:
            logger.debug("Dec {}".format(targets[1]))
        else:
            logger.debug("Dec {}".format(targets[q][1]))
        if int(len(targets)) == 4 and targets.size==4:
            varCoord = SkyCoord(targets[0],(targets[1]), frame='icrs', unit=degree) # Need to remove target stars from consideration
        else:
            varCoord = SkyCoord(targets[q][0],(targets[q][1]), frame='icrs', unit=degree) # Need to remove target stars from consideration

        # Grabbing variable rows
        logger.debug("Extracting and Measuring Differential Magnitude in each Photometry File")
        outputPhot=[] # new
        allcountscount=0

        for imgs, photFile in enumerate(tqdm(photFileArray)):
            fileRaDec = photCoordsFile[imgs]
            idx, d2d, _ = varCoord.match_to_catalog_sky(fileRaDec)
            starRejected=0
            if (less(d2d.arcsecond, targetRadius)):
                #magErrVar = 1.0857 * (photFile[idx][5]/photFile[idx][4])
                magErrVar = photFile[idx][5]
                if magErrVar < errorReject:

                    magErrEns = 1.0857 * (allCountsArray[allcountscount][1]/allCountsArray[allcountscount][0])
                    magErrTotal = pow( pow(magErrVar,2) + pow(magErrEns,2),0.5)

                    #templist is a temporary holder of the resulting file.
                    tempList=photFile[idx,0:6]
                    googFile = Path(fileList[imgs]).name
                    tempList = append(tempList, float(googFile.split("_")[2].replace("d",".")))
                    tempList = append(tempList, float(googFile.split("_")[4].replace("a",".")))
                    tempList = append(tempList, allCountsArray[allcountscount][0])
                    tempList = append(tempList, allCountsArray[allcountscount][1])

                    #Differential Magnitude
                    tempList = append(tempList, 2.5 * log10(allCountsArray[allcountscount][0]/photFile[idx][11]))
                    tempList = append(tempList, magErrTotal)
                    tempList = append(tempList, photFile[idx][11])
                    tempList = append(tempList, photFile[idx][5])

                    if (compFile.shape[0]== 5 and compFile.size ==5) or (compFile.shape[0]== 3 and compFile.size ==3):
                        loopLength=1
                    else:
                        loopLength=compFile.shape[0]
                    for j in range(loopLength):
                        if compFile.size == 2 or (compFile.shape[0]== 3 and compFile.size ==3) or (compFile.shape[0]== 5 and compFile.size ==5):
                            matchCoord=SkyCoord(ra=compFile[0]*degree, dec=compFile[1]*degree)
                        else:
                            matchCoord=SkyCoord(ra=compFile[j][0]*degree, dec=compFile[j][1]*degree)
                        idx, d2d, d3d = matchCoord.match_to_catalog_sky(fileRaDec)
                        tempList=append(tempList, photFileArray[imgs][idx][11])
                    outputPhot.append(tempList)

                    fileCount.append(allCountsArray[allcountscount][0])
                    allcountscount=allcountscount+1

                else:
                    starErrorRejCount=starErrorRejCount+1
                    starRejected=1

            else:
                starDistanceRejCount=starDistanceRejCount+1
                starRejected=1

            if ( starRejected == 1):

                    #templist is a temporary holder of the resulting file.
                    tempList=photFileArray[imgs][idx,0:6]
                    googFile = Path(fileList[imgs]).name
                    tempList=append(tempList, float(googFile.split("_")[2].replace("d",".")))
                    tempList=append(tempList, float(googFile.split("_")[4].replace("a",".")))
                    tempList=append(tempList, allCountsArray[allcountscount][0])
                    tempList=append(tempList, allCountsArray[allcountscount][1])

                    #Differential Magnitude
                    tempList=append(tempList,nan)
                    tempList=append(tempList,nan)
                    tempList=append(tempList, photFileArray[imgs][idx][11])
                    tempList=append(tempList, photFileArray[imgs][idx][5])

                    if (compFile.shape[0]== 5 and compFile.size ==5) or (compFile.shape[0]== 3 and compFile.size ==3):
                        loopLength=1
                    else:
                        loopLength=compFile.shape[0]

                    for j in range(loopLength):
                        if compFile.size == 2 or (compFile.shape[0]== 3 and compFile.size ==3) or (compFile.shape[0]== 5 and compFile.size ==5):
                            matchCoord=SkyCoord(ra=compFile[0]*degree, dec=compFile[1]*degree)
                        else:
                            matchCoord=SkyCoord(ra=compFile[j][0]*degree, dec=compFile[j][1]*degree)
                        idx, d2d, d3d = matchCoord.match_to_catalog_sky(fileRaDec)
                        tempList=append(tempList, photFileArray[imgs][idx][11])
                    outputPhot.append(tempList)
                    fileCount.append(allCountsArray[allcountscount][0])
                    allcountscount=allcountscount+1

        # Check for dud images
        imageReject=[]
        for j in range(asarray(outputPhot).shape[0]):
            if isnan(outputPhot[j][11]):
                imageReject.append(j)
        outputPhot=delete(outputPhot, imageReject, axis=0)

        try:
            outputPhot=np.vstack(asarray(outputPhot))

            ## REMOVE MAJOR OUTLIERS FROM CONSIDERATION
            stdVar=nanstd((outputPhot)[:,10])
            avgVar=nanmean((outputPhot)[:,10])
            starReject=[]
            stdevReject=0
            # Reject by simple major stdev elimination
            while True:
                starReject=[]
                for j in range(asarray(outputPhot).shape[0]):

                    if outputPhot[j][10] > avgVar+(4*stdVar) or outputPhot[j][10] < avgVar-(4*stdVar) :
                        starReject.append(j)
                        stdevReject=stdevReject+1

                if starReject != []:
                    outputPhot=delete(outputPhot, starReject, axis=0)
                else:
                    break

            # Reject by outsized error elimination

            while True:
                errorsArray=[]
                for j in range(asarray(outputPhot).shape[0]):
                    errorsArray.append(outputPhot[j][11])
                errorsArray=np.asarray(errorsArray)
                stdErrors=nanstd(errorsArray)
                avgErrors=nanmean(errorsArray)
                starReject=[]
                for j in range(asarray(outputPhot).shape[0]):
                    if outputPhot[j][11] > avgErrors+(4*stdErrors):
                        starReject.append(j)
                        starErrorRejCount=starErrorRejCount+1

                if starReject != []:
                    outputPhot=delete(outputPhot, starReject, axis=0)
                else:
                    break

            sys.stdout.write('\n')
            logger.info("Rejected Stdev Measurements: : {}".format(stdevReject))
            logger.info("Rejected Error Measurements: : {}".format(starErrorRejCount))
            logger.info("Rejected Distance Measurements: : {}".format(starDistanceRejCount))
            logger.info("Variability of Comparisons")
            logger.info("Average : {}".format(avgVar))
            logger.info("Stdev   : {}".format(stdVar))

            outputPhot=delete(outputPhot, starReject, axis=0)

        except ValueError:
            #raise AstrosourceException("No target stars were detected in your dataset. Check your input target(s) RA/Dec")
            logger.error("This target star was not detected in your dataset. Check your input target(s) RA/Dec")
            logger.info("Rejected Stdev Measurements: : {}".format(stdevReject))
            logger.error("Rejected Error Measurements: : {}".format(starErrorRejCount))
            logger.error("Rejected Distance Measurements: : {}".format(starDistanceRejCount))

        # Add calibration columns
        outputPhot= np.c_[outputPhot, np.ones(outputPhot.shape[0]),np.ones(outputPhot.shape[0])]

        if outputPhot.shape[0] > 2:
            savetxt(paths['outcatPath'] / f"doerPhot_V{str(q+1)}.csv", outputPhot, delimiter=",", fmt='%0.8f')
            logger.debug('Saved doerPhot_V')
        else:
            logger.info("Could not make photometry file, not enough observations.")


        photometrydata.append(outputPhot)

    return photometrydata

def calibrated_photometry(paths, photometrydata, colourterm, colourerror, colourdetect, linearise, targetcolour, rejectmagbrightest, rejectmagdimmest, calibCompFile):
    pdata = []

    for j, outputPhot in enumerate(photometrydata):

        logger.info("Calibrating Photometry")

        # Load in calibrated magnitudes and add them
        single_value = True if calibCompFile.shape[0] == 5 and calibCompFile.size != 25 else False

        if single_value:
            ensembleMag=calibCompFile[3]
        else:
            ensembleMag=calibCompFile[:,3]
        ensMag=0

        if single_value:
            ensMag=pow(10,-ensembleMag*0.4)
        else:
            for q in range(len(ensembleMag)):
                ensMag=ensMag+(pow(10,-ensembleMag[q]*0.4))

        ensembleMag=-2.5*math.log10(ensMag)
        logger.info(f"Ensemble Magnitude: {ensembleMag}")

        #calculate error
        if single_value:
            ensembleMagError=calibCompFile[4]
        else:
            ensembleMagError=0.0
            for t in range(len(calibCompFile[:,4])):
                ensembleMagError=ensembleMagError+pow(calibCompFile[t,4],2)
            ensembleMagError=ensembleMagError/pow(len(calibCompFile[:,4]),0.5)

        calibIndex=np.asarray(outputPhot).shape[1]-1

        if (targetcolour == -99.0):
            for i in range(outputPhot.shape[0]):
                if (ensembleMag+outputPhot[i][10]) > rejectmagbrightest and (ensembleMag+outputPhot[i][10]) < rejectmagdimmest:
                    outputPhot[i][calibIndex-1]=ensembleMag+outputPhot[i][10] # Calibrated Magnitude SKIPPING colour term
                    #outputPhot[i][calibIndex]=pow(pow(outputPhot[i][11],2)+pow(errCalib,2),0.5) # Calibrated Magnitude Error. NEEDS ADDING in calibration error. NEEDS ADDING IN COLOUR ERROR
                    outputPhot[i][calibIndex]=pow(pow(outputPhot[i][11],2)+pow(ensembleMagError,2),0.5) # Calibrated Magnitude Error. NEEDS ADDING in calibration error. NEEDS ADDING IN COLOUR ERROR
                else:
                    outputPhot[i][calibIndex-1]=np.nan
                    outputPhot[i][calibIndex]=np.nan
        else:
            for i in range(outputPhot.shape[0]):
                if (ensembleMag+outputPhot[i][10]-(colourterm * targetcolour)) > rejectmagbrightest and (ensembleMag+outputPhot[i][10]-(colourterm * targetcolour)) < rejectmagdimmest:
                    outputPhot[i][calibIndex-1]=ensembleMag+outputPhot[i][10] - (colourterm * targetcolour) # Calibrated Magnitude incorporating colour term
                    #outputPhot[i][calibIndex]=pow(pow(outputPhot[i][11],2)+pow(errCalib,2),0.5) # Calibrated Magnitude Error. NEEDS ADDING in calibration error. NEEDS ADDING IN COLOUR ERROR
                    outputPhot[i][calibIndex]=pow(pow(outputPhot[i][11],2)+pow(ensembleMagError,2),0.5) # Calibrated Magnitude Error. NEEDS ADDING in calibration error. NEEDS ADDING IN COLOUR ERROR
                else:
                    outputPhot[i][calibIndex-1]=np.nan
                    outputPhot[i][calibIndex]=np.nan

        # Write back to photometry data
        pdata.append(outputPhot)

        #update doerphot on disk
        savetxt(paths['outcatPath'] / f"doerPhot_V{str(j+1)}.csv", outputPhot, delimiter=",", fmt='%0.8f')

    if (targetcolour == -99.0):
        logger.info("Target colour not provided. Target magnitude does not incorporate a colour correction.")
        logger.info("This is likely ok if your colour term is low (<<0.05). If your colour term is high (>0.05), ")
        logger.info("then consider providing an appropriate colour for this filter using the --targetcolour option")
        logger.info("as well as an appropriate colour term for this filter (using --colourdetect or --colourterm).")

    return pdata
