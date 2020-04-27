'''
Detrend: this approach is only appropriate for analysing exoplanet data
'''

from numpy import genfromtxt, savetxt, load, delete, asarray, polyfit
from astropy.coordinates import SkyCoord
import glob
import sys
import math
import os
import platform
import matplotlib.pyplot as plt
import click

import logging

logger = logging.getLogger('astrosource')

def detrend_data(paths, filterCode):
    polyFitRequest=1 # Currently only works with one or two coefficients


    # Load in list of used files
    fileList=[]
    for line in (paths['parent'] / "usedImages.txt").read_text():
        fileList.append(line.strip())

    fileList = paths['outcatPath'].glob('*diffExcel*csv')
    r=0
    #logger.debug(fileList)
    for file in fileList:
        photFile = load(paths['parent'] / file)
        exists=os.path.isfile(str(file).replace('diff','calib'))
        if exists:
            calibFile = genfromtxt(str(file).replace('diff','calib'), dtype=float, delimiter=',')
            logger.debug("Calibration difference")
            logger.debug(-(photFile[:,1]-calibFile[:,1])[0])
            calibDiff=-((photFile[:,1]-calibFile[:,1])[0])
        #logger.debug(photFile[:,1])
        #logger.debug(photFile[:,0])
        logger.debug(file)
        logger.debug(photFile[:,1])

        baseSubDate=min(photFile[:,0])
        logger.debug(baseSubDate)
        logger.debug(math.floor(baseSubDate))

        photFile[:,0]=photFile[:,0]-baseSubDate


        leftMost = click.prompt("Enter left side most valid date:")
        leftFlat = click.prompt("Enter left side end of flat region:")

        rightFlat = click.prompt("Enter right side start of flat region:")
        rightMost = click.prompt("Enter right side most valid date:")



        # Clip off edges
        clipReject=[]
        for i in range(photFile.shape[0]):
            if photFile[i,0] < float(leftMost) or photFile[i,0] > float(rightMost):
                clipReject.append(i)
                logger.debug(photFile[i,1])
                logger.debug("REJECT")
        logger.debug(photFile.shape[0])
        photFile=delete(photFile, clipReject, axis=0)
        logger.debug(photFile.shape[0])


        # Get an array holding only the flat bits
        transitReject=[]
        flatFile=asarray(photFile)
        for i in range(flatFile.shape[0]):
            if (flatFile[i,0] > float(leftMost) and flatFile[i,0] < float(leftFlat)) or (flatFile[i,0] > float(rightFlat) and flatFile[i,0] < float(rightMost)):
                logger.debug("Keep")
            else:
                transitReject.append(i)
                logger.debug(flatFile[i,0])
                logger.debug("REJECT")
        logger.debug(flatFile.shape[0])
        flatFile=delete(flatFile, transitReject, axis=0)
        logger.debug(flatFile.shape[0])

        #
        polyFit=polyfit(flatFile[:,0],flatFile[:,1],polyFitRequest)
        logger.debug(polyFit)

        #Remove trend from flat bits
        if polyFitRequest == 2:
            for i in range(flatFile.shape[0]):
                flatFile[i,1]=flatFile[i,1]-(polyFit[2]+(polyFit[1]*flatFile[i,0])+(polyFit[0]*pow(flatFile[i,0],2)))
        elif polyFitRequest ==1:
            for i in range(flatFile.shape[0]):
                flatFile[i,1]=flatFile[i,1]-(polyFit[1]+(polyFit[0]*flatFile[i,0]))


        #Remove trend from actual data
        if polyFitRequest == 2:
            for i in range(photFile.shape[0]):
                photFile[i,1]=photFile[i,1]-(polyFit[2]+(polyFit[1]*photFile[i,0])+(polyFit[0]*pow(photFile[i,0],2)))
        elif polyFitRequest ==1:
            for i in range(photFile.shape[0]):
                photFile[i,1]=photFile[i,1]-(polyFit[1]+(polyFit[0]*photFile[i,0]))


        #return basedate to the data
        photFile[:,0]=photFile[:,0]+baseSubDate

        # Output trimmed files
        savetxt(paths['outcatPath'] / 'V{}_diffPeranso.txt'.format(str(r+1)), photFile, delimiter=" ", fmt='%0.8f')
        savetxt(paths['outcatPath'] / 'V{}_diffExcel.csv'.format(str(r+1)), photFile, delimiter=",", fmt='%0.8f')

        # Output astroImageJ file
        outputPeransoCalib=[]
        #i=0
        for i in range(asarray(photFile).shape[0]):
            outputPeransoCalib.append([photFile[i][0]-2450000.0,photFile[i][1],photFile[i][2]])
            #i=i+1

        savetxt(paths['outcatPath'] / 'V{}_diffAIJ.txt'.format(str(r+1)), outputPeransoCalib, delimiter=" ", fmt='%0.8f')
        savetxt(paths['outcatPath'] / 'V{}_diffAIJ.csv'.format(str(r+1)), outputPeransoCalib, delimiter=",", fmt='%0.8f')

        # Output replot
        plt.cla()
        outplotx=photFile[:,0]
        outploty=photFile[:,1]
        plt.xlabel('BJD')
        plt.ylabel('Differential ' +filterCode+' Mag')
        plt.plot(outplotx,outploty,'bo')
        plt.ylim(max(outploty)+0.02,min(outploty)-0.02,'k-')
        plt.xlim(min(outplotx)-0.01,max(outplotx)+0.01)
        plt.grid(True)
        plt.savefig(paths['outputPath'] / 'V{}_EnsembleVarDiffMag.png'.format(str(r+1)))
        plt.savefig(paths['outputPath'] / 'V{}_EnsembleVarDiffMag.eps'.format(str(r+1)))
        plt.cla()
        plt.clf()
        plt.close()
        plt.close("all")

        r=r+1
    return
