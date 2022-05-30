from numpy import genfromtxt, savetxt, load, delete, asarray, multiply, log10, divide, \
    less, append, add, std, average, median, inf, nan, isnan, nanstd, nanmean, array
import numpy as np
from astropy.units import degree
from astropy.coordinates import SkyCoord
import glob
import sys
from pathlib import Path
import shutil

import math
import os

import logging

from astrosource.utils import photometry_files_to_array, AstrosourceException
from astrosource.plots import plot_variability

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger('astrosource')


def get_total_counts(photFileArray, compFile, loopLength):

    compArray = []
    allCountsArray = []
    logger.debug("***************************************")
    logger.debug("Calculating total counts")
    for photFile in photFileArray:
        allCounts = 0.0
        allCountsErr = 0.0
        fileRaDec = SkyCoord(ra=photFile[:, 0]*degree, dec=photFile[:, 1]*degree)
        #Array of comp measurements
        for j in range(loopLength):
            if compFile.size == 2 or (compFile.shape[0]== 3 and compFile.size ==3) or (compFile.shape[0]== 5 and compFile.size ==5):
                matchCoord = SkyCoord(ra=compFile[0]*degree, dec=compFile[1]*degree)
            else:
                matchCoord = SkyCoord(ra=compFile[j][0]*degree, dec=compFile[j][1]*degree)

            idx, d2d, d3d = matchCoord.match_to_catalog_sky(fileRaDec)
            allCounts = add(allCounts, photFile[idx][4])
            allCountsErr = add(allCountsErr, photFile[idx][5])
            if (compFile.shape[0] == 5 and compFile.size == 5) or (compFile.shape[0] == 3 and compFile.size == 3):
                break
        allCountsArray.append([allCounts, allCountsErr])
    #logger.debug(allCountsArray)
    return allCountsArray

def find_variable_stars(targets, matchRadius, errorReject=0.05, parentPath=None, varsearchthresh=10000, varsearchstdev=2.0, varsearchmagwidth=0.25, varsearchminimages=0.3):
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

    print (varsearchstdev)
    print (varsearchmagwidth)
    print (varsearchminimages)
    #sys.exit()


    #minimumNoOfObs = 10 # Minimum number of observations to count as a potential variable.

    # Load in list of used files
    fileList = []
    with open(parentPath / "usedImages.txt", "r") as f:
        for line in f:
            fileList.append(line.strip())

    # allocate minimum images to detect
    minimumNoOfObs=int(varsearchminimages*len(fileList))
    logger.debug("Minimum number of observations to detect: " + str(minimumNoOfObs))
    
    # LOAD Phot FILES INTO LIST
    photFileArray = []
    for file in fileList:
        photFileArray.append(load(parentPath / file))

    if not photFileArray:
        raise AstrosourceException("No input files")

    # LOAD IN COMPARISON FILE
    preFile = genfromtxt(parentPath / 'stdComps.csv', dtype=float, delimiter=',')

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

    compFile = genfromtxt(parentPath / "compsUsed.csv", dtype=float, delimiter=',')
    logger.debug("Stable Comparison Candidates below variability threshold")
    outputPhot = []

    # Get total counts for each file
    allCountsArray = get_total_counts(photFileArray, compFile, loopLength=compFile.shape[0])

    # Define targetlist as every star in referenceImage above a count threshold
    logger.debug("Setting up Variable Search List")
    targetFile = referenceFrame
    # Although remove stars that are below the variable countrate
    starReject=[]
    for q in range(targetFile.shape[0]):
        if targetFile[q][4] < varsearchthresh:
            starReject.append(q)
    logger.debug("Total number of stars in reference Frame: {}".format(targetFile.shape[0]))
    targetFile = delete(targetFile, starReject, axis=0)
    logger.debug("Total number of stars with sufficient counts: {}".format(targetFile.shape[0]))

    ## NEED TO REMOVE COMPARISON STARS FROM TARGETLIST

    allcountscount=0
    # For each variable calculate the variability
    outputVariableHolder=[]
    
    # Prep photfile coordinates
    photFileCoords=[]

    for photFile in photFileArray:
        photFileCoords.append(SkyCoord(ra=photFile[:,0]*degree, dec=photFile[:,1]*degree))
        
    logger.debug("Measuring variability of stars...... ")
        
    q=0
    for target in targetFile:
        q=q+1
        #logger.debug("*********************")
        #logger.debug("Processing Target {}".format(str(q)))
        #logger.debug("RA {}".format(target[0]))
        #logger.debug("DEC {}".format(target[1]))
        varCoord = SkyCoord(target[0],(target[1]), frame='icrs', unit=degree) # Need to remove target stars from consideration
        #outputPhot=[]
        #compArray=[]
        #compList=[]

        diffMagHolder=[]

        allcountscount=0

        r=0
        for photFile in photFileArray:
            #compList=[]
            fileRaDec = photFileCoords[r]
            r=r+1
            idx, d2d, _ = varCoord.match_to_catalog_sky(fileRaDec)
            multTemp=(multiply(-2.5,log10(divide(photFile[idx][4],allCountsArray[allcountscount][0]))))
            if less(d2d.arcsecond, matchRadius) and (multTemp != inf) :
                diffMagHolder=append(diffMagHolder,multTemp)
            allcountscount=add(allcountscount,1)

        # ## REMOVE MAJOR OUTLIERS FROM CONSIDERATION
        # while True:
        #     stdVar=std(diffMagHolder)
        #     avgVar=average(diffMagHolder)
        #     starReject=[]
        #     z=0
        #     for j in range(asarray(diffMagHolder).shape[0]):
        #         if diffMagHolder[j] > avgVar+(4*stdVar) or diffMagHolder[j] < avgVar-(4*stdVar) :
        #             starReject.append(j)
        #             logger.debug("REJECT {}".format(diffMagHolder[j]))
        #             z=1
        #     diffMagHolder=delete(diffMagHolder, starReject, axis=0)
        #     if z==0:
        #         break
    
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
   

        #diffmag = asarray(diffMagHolder)
        #logger.debug("Standard Deviation in mag: {}".format(std(diffMagHolder)))
        #logger.debug("Median Magnitude: {}".format(median(diffMagHolder)))
        #logger.debug("Number of Observations: {}".format(diffMagHolder.shape[0]))

        if (diffMagHolder.shape[0] > minimumNoOfObs):
            outputVariableHolder.append( [target[0],target[1],median(diffMagHolder), std(diffMagHolder), diffMagHolder.shape[0]])

    

    savetxt(parentPath / "starVariability.csv", outputVariableHolder, delimiter=",", fmt='%0.8f')
    
    
    
    
    ## Routine that actually pops out potential variables.
    starVar = np.asarray(outputVariableHolder)
    
    meanMags = starVar[:,2]
    variations = starVar[:,3]

    #print (meanMags)
    #print (variations)

    xStepSize= varsearchmagwidth
    yStepSize=0.02
    #print (np.min(meanMags))
    #print (np.max(meanMags))
    xbins = np.arange(np.min(meanMags), np.max(meanMags), xStepSize)
    #print (xbins)
    #ybins = np.linspace(np.min(variations), np.max(variations), num=10)
    ybins = np.arange(np.min(variations), np.max(variations), yStepSize)
    #print (ybins)


    #print (np.digitize(meanMags, bins))
    #binStarVar = np.histogram()

    #H, xedges, yedges = np.histogram2d(meanMags, variations, bins=(xbins, ybins))


    #print (H.T)

    #H=H.T


    #split it into one array and identify variables in bins with centre
    variationsByMag=[]
    potentialVariables=[]
    for xbinner in range(len (xbins)):
        #print (xbinner)
        #print (xbins[xbinner])
        starsWithin=[]
        for q in range(len(meanMags)):
            if meanMags[q] >= xbins[xbinner] and meanMags[q] < xbins[xbinner]+xStepSize:
                starsWithin.append(variations[q])
        #print (np.mean(starsWithin))
        #print (np.std(starsWithin))
        meanStarsWithin= (np.mean(starsWithin))
        stdStarsWithin= (np.std(starsWithin))
        variationsByMag.append([xbins[xbinner]+0.5*xStepSize,meanStarsWithin,stdStarsWithin])
        
        # At this point extract RA and Dec of stars that may be variable
        for q in range(len(starVar[:,2])):
            if starVar[q,2] >= xbins[xbinner] and starVar[q,2] < xbins[xbinner]+xStepSize:
                if starVar[q,3] > (meanStarsWithin + varsearchstdev*stdStarsWithin):
                    #print (starVar[q,3])
                    potentialVariables.append([starVar[q,0],starVar[q,1],starVar[q,2],starVar[q,3]])
                    
        
    #print (variationsByMag)
    #print (potentialVariables)
        

    potentialVariables=np.array(potentialVariables)
    savetxt(parentPath / "potentialVariables.csv", potentialVariables , delimiter=",", fmt='%0.8f')
    #savetxt("potentialVariables.csv", potentialVariables , delimiter=",", fmt='%0.8f')
    fig, ax = plt.subplots(figsize =(10, 7))
    plt.hist2d(meanMags, variations, bins =[xbins, ybins], cmap = plt.cm.nipy_spectral)
    plt.colorbar()
    plt.title("Variation Histogram")
    ax.set_xlabel('Mean Differential Magnitude') 
    ax.set_ylabel('Variation (Standard Deviation)') 
    plt.plot(potentialVariables[:,2],potentialVariables[:,3],'ro')
    plt.tight_layout()
    # fig=plt.figure()
    # #fig.set_size_inches(17,3)
    # #fig = plt.figure(figsize=(7, 3))
    # #ax = fig.add_subplot(131, title='imshow: square bins')
    # #plt.imshow(H, interpolation='nearest', origin='lower')#,
    # plt.imshow(H, interpolation='nearest', origin='lower')#,
    #         #extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]])
    # plt.subplots_adjust(left=0.15, right=0.98, top=0.98, bottom=0.17, wspace=0.3, hspace=0.4)

    # #ax = fig.add_subplot(132, title='pcolormesh: actual edges',
    # #        aspect='equal')
    # #X, Y = np.meshgrid(xbins, ybins)
    # #ax.pcolormesh(X, Y, H)

    # # ax = fig.add_subplot(133, title='NonUniformImage: interpolated',
    # #         aspect='equal', xlim=xbins[[0, -1]], ylim=ybins[[0, -1]])
    # # im = NonUniformImage(ax, interpolation='bilinear')
    # # xcenters = (xedges[:-1] + xedges[1:]) / 2
    # # ycenters = (yedges[:-1] + yedges[1:]) / 2
    # # im.set_data(xcenters, ycenters, H)
    # #ax.images.append(im)

    plt.savefig(parentPath / "Variation2DHistogram.png")
    
    
    
    plot_variability(outputVariableHolder, potentialVariables, parentPath)

    return outputVariableHolder

def photometric_calculations(targets, paths, targetRadius, errorReject=0.5, filesave=True):
    
    fileCount=[]
    photometrydata = []
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

    # clear output plots, cats and period fodlers and regenerate
    folders = ['periods', 'checkplots', 'eelbs', 'outputcats','outputplots','trimcats']
    for fd in folders:
        if (paths['parent'] / fd).exists():
            shutil.rmtree(paths['parent'] / fd, ignore_errors=True )
            try:
                os.mkdir(paths['parent'] / fd)
            except OSError:
                print ("Creation of the directory %s failed" % paths['parent'])
            else:
                print ("Successfully created the directory %s " % paths['parent'])


    # Get total counts for each file
    if compFile.shape[0]== 5 and compFile.size ==5:
        loopLength=1
    else:
        loopLength=compFile.shape[0]
    allCountsArray = get_total_counts(photFileArray, compFile, loopLength)

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
        compArray=[]
        compList=[]
        allcountscount=0
        for imgs, photFile in enumerate(photFileArray):
            sys.stdout.write('.')
            compList=[]
            fileRaDec = SkyCoord(ra=photFile[:,0]*degree, dec=photFile[:,1]*degree)
            idx, d2d, _ = varCoord.match_to_catalog_sky(fileRaDec)
            starRejected=0
            if (less(d2d.arcsecond, targetRadius)):
                magErrVar = 1.0857 * (photFile[idx][5]/photFile[idx][4])
                if magErrVar < errorReject:

                    magErrEns = 1.0857 * (allCountsArray[allcountscount][1]/allCountsArray[allcountscount][0])
                    magErrTotal = pow( pow(magErrVar,2) + pow(magErrEns,2),0.5)

                    #templist is a temporary holder of the resulting file.
                    tempList=photFile[idx,0:6]
                    # logger.debug(f"{tempList}")
                    googFile = Path(fileList[imgs]).name
                    tempList = append(tempList, float(googFile.split("_")[2].replace("d",".")))
                    tempList = append(tempList, float(googFile.split("_")[4].replace("a",".")))
                    tempList = append(tempList, allCountsArray[allcountscount][0])
                    tempList = append(tempList, allCountsArray[allcountscount][1])

                    #Differential Magnitude
                    tempList = append(tempList, 2.5 * log10(allCountsArray[allcountscount][0]/photFile[idx][4]))
                    tempList = append(tempList, magErrTotal)
                    tempList = append(tempList, photFile[idx][4])
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
                        tempList=append(tempList, photFileArray[imgs][idx][4])
                    # logger.debug(f"{tempList}")
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
                    tempList=photFileArray[imgs][idx,:]
                    googFile = Path(fileList[imgs]).name
                    tempList=append(tempList, float(googFile.split("_")[2].replace("d",".")))
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

                    for j in range(loopLength):
                        if compFile.size == 2 or (compFile.shape[0]== 3 and compFile.size ==3) or (compFile.shape[0]== 5 and compFile.size ==5):
                            matchCoord=SkyCoord(ra=compFile[0]*degree, dec=compFile[1]*degree)
                        else:
                            matchCoord=SkyCoord(ra=compFile[j][0]*degree, dec=compFile[j][1]*degree)
                        idx, d2d, d3d = matchCoord.match_to_catalog_sky(fileRaDec)
                        tempList=append(tempList, photFileArray[imgs][idx][4])
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
        except ValueError:
            #raise AstrosourceException("No target stars were detected in your dataset. Check your input target(s) RA/Dec")
            logger.info("This target star was not detected in your dataset. Check your input target(s) RA/Dec")

        ## REMOVE MAJOR OUTLIERS FROM CONSIDERATION
        stdVar=nanstd((outputPhot)[:,10])
        avgVar=nanmean((outputPhot)[:,10])
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

        # Add calibration columns
        outputPhot= np.c_[outputPhot, np.ones(outputPhot.shape[0]),np.ones(outputPhot.shape[0])]

        if outputPhot.shape[0] > 2:
            savetxt(paths['outcatPath'] / f"doerPhot_V{str(q+1)}.csv", outputPhot, delimiter=",", fmt='%0.8f')
            logger.debug('Saved doerPhot_V')
        else:
            #raise AstrosourceException("Photometry not possible")
            logger.info("Could not make photometry file, not enough observations.")
        #logger.debug(array(outputPhot).shape)

        photometrydata.append(outputPhot)
    # photometrydata = trim_catalogue(photometrydata)
    return photometrydata

def calibrated_photometry(paths, photometrydata, colourterm, colourerror, colourdetect, linearise, targetcolour, rejectmagbrightest, rejectmagdimmest):
    pdata = []

    for j, outputPhot in enumerate(photometrydata):
        calibCompFile = genfromtxt(paths['parent'] / 'calibCompsUsed.csv', dtype=float, delimiter=',')
        compFile = genfromtxt(paths['parent'] / 'stdComps.csv', dtype=float, delimiter=',')
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
                    
            logger.info("No provided target colour was provided. Target magnitude does not incorporate a colour correction.")
            logger.info("This is likely ok if your colour term is low (<<0.05). If your colour term is high (>0.05), ")
            logger.info("then consider providing an appropriate colour for this filter using the --targetcolour option")
            logger.info("as well as an appropriate colour term for this filter (using --colourdetect or --colourterm).")
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

    return pdata
