import numpy
from astropy import units as u
import scipy
from astropy.coordinates import SkyCoord
import glob
import sys
import matplotlib
matplotlib.use("TkAgg") # must be before pyplot

import matplotlib.pyplot as plt
import math
import os
import platform

import logging

logger = logging.getLogger(__name__)

def calculate_curves(parentPath = None):
    errorReject=0.05 # reject measurements with instrumental errors larger than this (this is not total error, just the estimated error in the single measurement of the variable)
    acceptDistance=2.0 # Furtherest distance in arcseconds for matches

    # Get list of phot files
    if not parentPath:
        parentPath = os.getcwd()
    outputPath = os.path.join(parentPath,"outputplots")
    outcatPath = os.path.join(parentPath,"outputcats")
    checkPath = os.path.join(parentPath,"checkplots")

    #create directory structure
    if not os.path.exists(outputPath):
        os.makedirs(outputPath)

    if not os.path.exists(outcatPath):
        os.makedirs(outcatPath)

    if not os.path.exists(checkPath):
        os.makedirs(checkPath)


    # Load in list of used files
    fileList=[]
    with open("usedImages.txt", "r") as f:
      for line in f:
        fileList.append(line.strip())

    # LOAD Phot FILES INTO LIST

    photFileArray=[]
    for file in fileList:
        photFileArray.append(numpy.genfromtxt(file, dtype=float, delimiter=','))

    photFileArray=numpy.asarray(photFileArray)

    targetFile = numpy.genfromtxt('targetstars.csv', dtype=float, delimiter=',')

    compFile=numpy.genfromtxt('compsUsed.csv', dtype=float, delimiter=',')
    #compFile=numpy.asarray(compFile)

    logger.info("Stable Comparison Candidates below variability threshold")
    logger.info(compFile.shape[0])
    logger.debug(compFile)



    outputPhot=[]

    # Get total counts for each file

    fileCount=[]

    compArray=[]

    allCountsArray=[]

    for imgs in range(photFileArray.shape[0]):
        allCounts=0.0
        allCountsErr=0.0
        photFile = photFileArray[imgs]
        fileRaDec = SkyCoord(ra=photFile[:,0]*u.degree, dec=photFile[:,1]*u.degree)
        #Array of comp measurements
        #compList=[]
        #logger.info("***************************************")
        logger.info("Calculating total Comparison counts for" + str(fileList[imgs]))


        for j in range(compFile.shape[0]):

            matchCoord=SkyCoord(ra=compFile[j][0]*u.degree, dec=compFile[j][1]*u.degree)
            idx, d2d, d3d = matchCoord.match_to_catalog_sky(fileRaDec)

            allCounts=numpy.add(allCounts,photFile[idx][4])
            allCountsErr=numpy.add(allCountsErr,photFile[idx][5])
            # Make array of comp measurements for this file

        allCountsArray.append([allCounts,allCountsErr])

    logger.debug(allCountsArray)

    allcountscount=0
    # For each variable calculate all the things
    for q in range(targetFile.shape[0]):
        starErrorRejCount=0
        starDistanceRejCount=0
        logger.info("***********************************************************************")
        logger.info("Processing Variable " +str(q+1))
        logger.info("RA")
        logger.info(targetFile[q][0])
        logger.info("DEC")
        logger.info(targetFile[q][1])
        varCoord = SkyCoord(targetFile[q][0],(targetFile[q][1]), frame='icrs', unit=u.deg) # Need to remove target stars from consideration
        # Check that this is connecting to the correct star... is there a maximum limit set? 2 arcseconds?
        #idx, d2d, d3d = varCoord.match_to_catalog_sky(fileRaDec)
        #referenceFrame=numpy.delete(referenceFrame, idx, axis=0)
        # Grabbing variable rows
        logger.info("Extracting and Measuring Differential Magnitude in each Photometry File")
        #countCount=0
        outputPhot=[] # new
        compArray=[]
        compList=[]
        allcountscount=0
        for imgs in range(photFileArray.shape[0]):
            compList=[]

            fileRaDec = SkyCoord(ra=photFileArray[imgs][:,0]*u.degree, dec=photFileArray[imgs][:,1]*u.degree)

            idx, d2d, _ = varCoord.match_to_catalog_sky(fileRaDec)

            starRejected=0
            if (numpy.less(d2d.arcsecond, acceptDistance)):
                magErrVar = 1.0857 * (photFileArray[imgs][idx][5]/photFileArray[imgs][idx][4])
                #logger.info("Distance ok!")
                #logger.info(magErrVar)
                if magErrVar < errorReject:
                    #logger.info("MagError ok!")
                    magErrEns = 1.0857 * (allCountsErr/allCounts)
                    magErrTotal = pow( pow(magErrVar,2) + pow(magErrEns,2),0.5)

                    #templist is a temporary holder of the resulting file.
                    tempList=photFileArray[imgs][idx,:]


                    googFile = (fileList[imgs].replace(parentPath,"").replace('inputs',"").replace('//',""))
                    #logger.info(googFile.split("_")[5])
                    tempList=numpy.append(tempList, float(googFile.split("_")[5].replace("d",".")))
                    tempList=numpy.append(tempList, float(googFile.split("_")[4].replace("a",".")))
                    tempList=numpy.append(tempList, allCountsArray[allcountscount][0])
                    tempList=numpy.append(tempList, allCountsArray[allcountscount][1])

                    #Differential Magnitude
                    tempList=numpy.append(tempList,-2.5 * numpy.log10(photFileArray[imgs][idx][4]/allCountsArray[allcountscount][0]))
                    #logger.info(numpy.append(tempList,-2.5 * numpy.log10(photFile[idx][4]/allCountsArray[countCount][0])))
                    tempList=numpy.append(tempList, magErrTotal)


                    #Add these things to compArray also
                    tempList=numpy.append(tempList, photFileArray[imgs][idx][4])
                    tempList=numpy.append(tempList, photFileArray[imgs][idx][5])

                    for j in range(compFile.shape[0]):

                        matchCoord=SkyCoord(ra=compFile[j][0]*u.degree, dec=compFile[j][1]*u.degree)
                        idx, d2d, d3d = matchCoord.match_to_catalog_sky(fileRaDec)
                        tempList=numpy.append(tempList, photFileArray[imgs][idx][4])

                    outputPhot.append(tempList)

                    fileCount.append(allCounts)
                    #countCount = countCount + 1
                    allcountscount=allcountscount+1

                else:
                    #logger.info('Star Error Too High - ' + str(magErrVar) + ' - measurement rejected')
                    starErrorRejCount=starErrorRejCount+1
                    starRejected=1
            else:
                    #logger.info('Star Distance Too High - ' + str(magErrVar) + ' - measurement rejected')
                    starDistanceRejCount=starDistanceRejCount+1
                    starRejected=1

            if ( starRejected == 1):

                    #templist is a temporary holder of the resulting file.
                    tempList=photFileArray[imgs][idx,:]


                    googFile = (fileList[imgs].replace(parentPath,"").replace('inputs',"").replace('//',""))
                    #logger.info(googFile.split("_")[5])
                    tempList=numpy.append(tempList, float(googFile.split("_")[5].replace("d",".")))
                    tempList=numpy.append(tempList, float(googFile.split("_")[4].replace("a",".")))
                    tempList=numpy.append(tempList, allCountsArray[allcountscount][0])
                    tempList=numpy.append(tempList, allCountsArray[allcountscount][1])

                    #Differential Magnitude
                    tempList=numpy.append(tempList,numpy.nan)

                    tempList=numpy.append(tempList,numpy.nan)

                    tempList=numpy.append(tempList, photFileArray[imgs][idx][4])
                    tempList=numpy.append(tempList, photFileArray[imgs][idx][5])

                    for j in range(compFile.shape[0]):

                        matchCoord=SkyCoord(ra=compFile[j][0]*u.degree, dec=compFile[j][1]*u.degree)
                        idx, d2d, d3d = matchCoord.match_to_catalog_sky(fileRaDec)
                        tempList=numpy.append(tempList, photFileArray[imgs][idx][4])

                    outputPhot.append(tempList)

                    fileCount.append(allCounts)
                    allcountscount=allcountscount+1

        imageReject=[]
        for j in range(numpy.asarray(outputPhot).shape[0]):
            if numpy.isnan(outputPhot[j][11]):
                imageReject.append(j)
                #logger.info("IMAGE REJECTED")
        outputPhot=numpy.delete(outputPhot, imageReject, axis=0)
        #compArray=numpy.delete(compArray, imageReject, axis=0)

        ## REMOVE MAJOR OUTLIERS FROM CONSIDERATION
        stdVar=numpy.nanstd(numpy.asarray(outputPhot)[:,10])
        avgVar=numpy.nanmean(numpy.asarray(outputPhot)[:,10])
        starReject=[]
        #numpy.savetxt("compArraybugFind.csv", compArray, delimiter=",", fmt='%0.8f')
        stdevReject=0
        for j in range(numpy.asarray(outputPhot).shape[0]):
            if outputPhot[j][10] > avgVar+(4*stdVar) or outputPhot[j][10] < avgVar-(4*stdVar) :
                starReject.append(j)
                #logger.info("REJECT")
                #logger.info(outputPhot[j][10])
                stdevReject=stdevReject+1

        logger.info("Rejected Stdev Measurements: " + str(stdevReject))
        logger.info("Rejected Error Measurements: " + str(starErrorRejCount))
        logger.info("Rejected Distance Measurements: " + str(starDistanceRejCount))
        logger.info("Variability of Comparisons")
        logger.info("Average : " +str(avgVar))
        logger.info("Stdev   : "+str(stdVar))

        #logger.info(outputPhot)
        outputPhot=numpy.delete(outputPhot, starReject, axis=0)

        if outputPhot.shape[0] > 2:
            numpy.savetxt(os.path.join(outcatPath,"doerPhot_V" +str(q+1) +".csv"), outputPhot, delimiter=",", fmt='%0.8f')

        return

def plot_lightcurves(parentPath=None):
    filterCode = 3 # u=0, g=1, r=2, i=3, z=4
    calibFlag = 0 # 0 = no calibration attempted, 1 = calibration attempted.

    if not parentPath:
        parentPath = os.getcwd()
    doerPath = os.path.join(parentPath,"outputcats")
    fileList = glob.glob("{}/doer*.csv".format(doerPath))
    outputPath = os.path.join(parentPath,"outputplots")
    checkPath = os.path.join(parentPath,"checkplots")
    outcatPath = doerPath

    for file in fileList:
        outputPhot=numpy.genfromtxt(file, delimiter=",", dtype='float')

        r = file.split("_")[-1].replace(".csv","")
        logger.info("Making Plots and Catalogues for Variable " + str(r))

        plt.cla()
        outplotx=numpy.asarray(outputPhot)[:,6]
        outploty=numpy.asarray(outputPhot)[:,10]

        plt.xlabel('BJD')
        plt.ylabel('Differential Mag')
        plt.plot(outplotx,outploty,'bo')

        plt.ylim(max(outploty)+0.02,min(outploty)-0.02,'k-')
        plt.xlim(min(outplotx)-0.01,max(outplotx)+0.01)
        plt.grid(True)
        plt.savefig(os.path.join(outputPath,str(r)+'_'+'EnsembleVarDiffMag.png'))
        plt.savefig(os.path.join(outputPath,str(r)+'_'+'EnsembleVarDiffMag.eps'))

        plt.cla()
        outplotx=numpy.asarray(outputPhot)[:,7]
        outploty=numpy.asarray(outputPhot)[:,10]

        plt.xlabel('Airmass')
        plt.ylabel('Differential Mag')
        plt.plot(outplotx,outploty,'bo')
        #plt.plot(linex,liney)
        plt.ylim(min(outploty)-0.02,max(outploty)+0.02,'k-')
        plt.xlim(min(outplotx)-0.01,max(outplotx)+0.01)
        plt.grid(True)
        plt.savefig(os.path.join(checkPath,str(r)+'_'+'AirmassEnsVarDiffMag.png'))
        plt.savefig(os.path.join(checkPath,str(r)+'_'+'AirmassEnsVarDiffMag.eps'))

        plt.cla()
        outplotx=numpy.asarray(outputPhot)[:,7]
        outploty=numpy.asarray(outputPhot)[:,8]

        plt.xlabel('Airmass')
        plt.ylabel('Variable Counts')
        plt.plot(outplotx,outploty,'bo')
        #plt.plot(linex,liney)
        plt.ylim(min(outploty)-1000,max(outploty)+1000,'k-')
        plt.xlim(min(outplotx)-0.01,max(outplotx)+0.01)
        plt.grid(True)
        plt.savefig(os.path.join(checkPath,str(r)+'_'+'AirmassVarCounts.png'))
        plt.savefig(os.path.join(checkPath,str(r)+'_'+'AirmassVarCounts.eps'))

        # Make a calibrated version
        if calibFlag == 1:
            logger.info("Calibrating Photometry")
            ensembleMag=preFile[:,3+(2*filterCode)]
            ensembleMag=-(ensembleMag)*0.4
            ensembleMag=sum(pow(10,ensembleMag))
            ensembleMag=-2.5*numpy.log10(ensembleMag)

            ensembleMagError=preFile[:,4+(2*filterCode)]
            ensembleMagError=numpy.average(ensembleMagError)*1/pow(ensembleMagError.size, 0.5)

            for i in range(outputPhot.shape[0]):
                outputPhot[i][10]=outputPhot[i][10]+ensembleMag
                outputPhot[i][11]=pow((pow(outputPhot[i][11],2)+pow(ensembleMagError,2)),0.5)

            numpy.savetxt(os.path.join(outcatPath,"doerPhot_" +str(r) +".csv"), outputPhot, delimiter=",", fmt='%0.8f')


        # Output Calibed peranso file
        outputPeransoCalib=[]
        for i in range(outputPhot.shape[0]):
            outputPeransoCalib.append([outputPhot[i][6],outputPhot[i][10],outputPhot[i][11]])
            #i=i+1

        numpy.savetxt(os.path.join(outcatPath,str(r)+'_'+"calibPeranso.txt"), outputPeransoCalib, delimiter=" ", fmt='%0.8f')
        numpy.savetxt(os.path.join(outcatPath,str(r)+'_'+"calibExcel.csv"), outputPeransoCalib, delimiter=",", fmt='%0.8f')

        # Output astroImageJ file
        outputPeransoCalib=[]
        for i in range(numpy.asarray(outputPhot).shape[0]):
            outputPeransoCalib.append([outputPhot[i][6]-2450000.0,outputPhot[i][10],outputPhot[i][11]])

        numpy.savetxt(os.path.join(outcatPath,str(r)+'_'+"calibAIJ.txt"), outputPeransoCalib, delimiter=" ", fmt='%0.8f')
        numpy.savetxt(os.path.join(outcatPath,str(r)+'_'+"calibAIJ.csv"), outputPeransoCalib, delimiter=",", fmt='%0.8f')
        return
