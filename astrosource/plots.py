import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import math
import os

import logging

from astrosource.utils import photometry_files_to_array, AstrosourceException

logger = logging.getLogger('astrosource')

def make_plots(filterCode, paths):

    if (paths['parent'] / 'calibCompsUsed.csv').exists():
        calibFlag=1
    else:
        calibFlag=0

    fileList = paths['outcatPath'].glob("doer*.csv")
    # print([f for f in fileList])

    for file in fileList:

        outputPhot=np.genfromtxt(file, delimiter=",", dtype='float')
        r = file.stem.split("_")[-1]
        logger.info("Making Plots and Catalogues for Variable " + str(r))

        # Output Differential peranso file
        outputPeransoCalib=[]
        for i in range(outputPhot.shape[0]):
            outputPeransoCalib.append([outputPhot[i][6],outputPhot[i][10],outputPhot[i][11]])

        np.savetxt(paths['outcatPath'] / '{}_diffPeranso.txt'.format(r), outputPeransoCalib, delimiter=" ", fmt='%0.8f')
        np.savetxt(paths['outcatPath'] / '{}_diffExcel.csv'.format(r), outputPeransoCalib, delimiter=",", fmt='%0.8f')

        # Output Differential astroImageJ file
        outputPeransoCalib=[]
        for i in range(np.asarray(outputPhot).shape[0]):
            outputPeransoCalib.append([outputPhot[i][6]-2450000.0,outputPhot[i][10],outputPhot[i][11]])
        np.savetxt(paths['outcatPath'] / '{}_diffAIJ.txt'.format(r), outputPeransoCalib, delimiter=" ", fmt='%0.8f')
        np.savetxt(paths['outcatPath'] / '{}_diffAIJ.csv'.format(r), outputPeransoCalib, delimiter=",", fmt='%0.8f')

        plt.cla()
        outplotx=np.asarray(outputPhot)[:,6]
        outploty=np.asarray(outputPhot)[:,10]
        plt.xlabel('BJD')
        plt.ylabel('Differential ' +filterCode+' Mag')
        plt.plot(outplotx,outploty,'bo')
        plt.ylim(max(outploty)+0.02,min(outploty)-0.02,'k-')
        plt.xlim(min(outplotx)-0.01,max(outplotx)+0.01)
        plt.grid(True)
        plt.savefig(paths['outputPath'] / '{}_EnsembleVarDiffMag.png'.format(r))
        plt.savefig(paths['outputPath'] / '{}_EnsembleVarDiffMag.eps'.format(r))

        plt.cla()
        outplotx=np.asarray(outputPhot)[:,7]
        outploty=np.asarray(outputPhot)[:,10]
        plt.xlabel('Airmass')
        plt.ylabel('Differential ' +filterCode+' Mag')
        plt.plot(outplotx,outploty,'bo')
        plt.ylim(min(outploty)-0.02,max(outploty)+0.02,'k-')
        plt.xlim(min(outplotx)-0.01,max(outplotx)+0.01)
        plt.grid(True)
        plt.savefig(paths['checkPath'] / '{}_AirmassEnsVarDiffMag.png'.format(r))
        plt.savefig(paths['checkPath'] / '{}_AirmassEnsVarDiffMag.eps'.format(r))

        plt.cla()
        outplotx=np.asarray(outputPhot)[:,7]
        outploty=np.asarray(outputPhot)[:,8]
        plt.xlabel('Airmass')
        plt.ylabel('Variable Counts')
        plt.plot(outplotx,outploty,'bo')
        plt.ylim(min(outploty)-1000,max(outploty)+1000,'k-')
        plt.xlim(min(outplotx)-0.01,max(outplotx)+0.01)
        plt.grid(True)
        plt.savefig(paths['checkPath'] / '{}_AirmassVarCounts.png'.format(r))
        plt.savefig(paths['checkPath'] / '{}_AirmassVarCounts.eps'.format(r))

        # Make a calibrated version
        # Need to shift the shape of the curve against the lowest error in the catalogue.

        if calibFlag == 1:
            calibCompFile=np.genfromtxt(paths['parent'] / 'calibCompsUsed.csv', dtype=float, delimiter=',')
            compFile = np.genfromtxt(paths['parent'] / 'stdComps.csv', dtype=float, delimiter=',')
            logger.info("Calibrating Photometry")
            # Load in calibrated magnitudes and add them
            #logger.info(compFile.size)
            if compFile.shape[0] == 5 and compFile.size != 25:
                ensembleMag=calibCompFile[3]
            else:
                ensembleMag=calibCompFile[:,3]
            ensMag=0

            if compFile.shape[0] == 5 and compFile.size != 25:
                lenloop=1
            else:
                lenloop=len(calibCompFile[:,3])
            for q in range(lenloop):
                if compFile.shape[0] == 5 and compFile.size != 25:
                    ensMag=pow(10,-ensembleMag*0.4)
                else:
                    ensMag=ensMag+(pow(10,-ensembleMag[q]*0.4))
            #logger.info(ensMag)
            ensembleMag=-2.5*math.log10(ensMag)
            logger.info("Ensemble Magnitude: "+str(ensembleMag))


            #calculate error
            if compFile.shape[0] == 5 and compFile.size !=25:
                ensembleMagError=calibCompFile[4]
                #ensembleMagError=np.average(ensembleMagError)*1/pow(ensembleMagError.size, 0.5)
            else:
                ensembleMagError=calibCompFile[:,4]
                ensembleMagError=np.average(ensembleMagError)*1/pow(ensembleMagError.size, 0.5)

            #for file in fileList:
            for i in range(outputPhot.shape[0]):
                outputPhot[i][10]=outputPhot[i][10]+ensembleMag
                #outputPhot[i][11]=pow((pow(outputPhot[i][11],2)+pow(ensembleMagError,2)),0.5)


            plt.cla()
            outplotx=np.asarray(outputPhot)[:,6]
            outploty=np.asarray(outputPhot)[:,10]
            plt.xlabel('BJD')
            plt.ylabel('Calibrated ' +filterCode+' Mag')
            plt.plot(outplotx,outploty,'bo')
            plt.ylim(max(outploty)+0.02,min(outploty)-0.02,'k-')
            plt.xlim(min(outplotx)-0.01,max(outplotx)+0.01)
            plt.grid(True)
            plt.savefig(paths['outputPath'] / '{}_EnsembleVarCalibMag.png'.format(r))
            plt.savefig(paths['outputPath'] / '{}_EnsembleVarCalibMag.eps'.format(r))


            # Output Calibed peranso file
            outputPeransoCalib=[]
            r = file.stem.split("_")[-1]
            for i in range(outputPhot.shape[0]):
                outputPeransoCalib.append([outputPhot[i][6],outputPhot[i][10],outputPhot[i][11]])
            np.savetxt(paths['outcatPath'] / '{}_calibPeranso.txt'.format(r), outputPeransoCalib, delimiter=" ", fmt='%0.8f')
            np.savetxt(paths['outcatPath'] / '{}_calibExcel.csv'.format(r), outputPeransoCalib, delimiter=",", fmt='%0.8f')

            # Output astroImageJ file
            outputPeransoCalib=[]
            for i in range(np.asarray(outputPhot).shape[0]):
                outputPeransoCalib.append([outputPhot[i][6]-2450000.0,outputPhot[i][10],outputPhot[i][11]])
                #i=i+1

            np.savetxt(paths['outcatPath'] / '{}_calibAIJ.txt'.format(r), outputPeransoCalib, delimiter=" ", fmt='%0.8f')
            np.savetxt(paths['outcatPath'] / '{}_calibAIJ.csv'.format(r), outputPeransoCalib, delimiter=",", fmt='%0.8f')
    return

def phased_plots(paths, filterCode):

    # Load in list of used files
    fileList=[]
    for line in (paths['parent'] / "usedImages.txt").read_text():
        fileList.append(line.strip())

    fileList = paths['parent'].glob("*.p*")
    #logger.debug(fileList)

    targetFile = np.genfromtxt(paths['parent'] / 'targetstars.csv', dtype=float, delimiter=',')
    # Remove any nan rows from targetFile
    targetRejecter=[]
    if not (targetFile.shape[0] == 4 and targetFile.size ==4):
        for z in range(targetFile.shape[0]):
          if np.isnan(targetFile[z][0]):
            targetRejecter.append(z)
        targetFile=np.delete(targetFile, targetRejecter, axis=0)

    if targetFile.size == 4 and targetFile.shape[0] ==4:
        loopLength=1
    else:
        loopLength=targetFile.shape[0]
    for q in range(loopLength):
        filename = paths['outcatPath'] / 'V{}_calibExcel.csv'.format(q+1)
        calibFile = np.genfromtxt(filename, dtype=float, delimiter=',')

        logger.debug(targetFile.size)
        logger.debug(targetFile.shape[0])

        if targetFile.size == 4 and targetFile.shape[0] ==4:
            period = targetFile[2]
            phaseShift = targetFile[3]
        else:
            period = targetFile[q][2]
            phaseShift = targetFile[q][3]

        #logger.debug(calibFile[:,6])
        #logger.debug(calibFile[:,10])
        #logger.debug(calibFile[:,11])

        # Variable lightcurve

        plt.cla()
        outplotx=calibFile[:,0]
        outploty=calibFile[:,1]
        logger.debug(outplotx)
        logger.debug(outploty)
        plt.xlabel('BJD')
        plt.ylabel('Apparent {} Magnitude'.format(filterCode))
        plt.plot(outplotx,outploty,'bo')
        #plt.plot(linex,liney)
        plt.ylim(max(outploty)-0.04,min(outploty)+0.04,'k-')
        plt.xlim(min(outplotx)-0.01,max(outplotx)+0.01)
        plt.grid(True)
        plt.savefig(outputPath / 'Variable{}_{}_Lightcurve.png'.format(q+1,filterCode))
        plt.savefig(outputPath / 'Variable{}_{}_Lightcurve.eps'.format(q+1,filterCode))

        # Phased lightcurve

        plt.cla()
        fig = matplotlib.pyplot.gcf()
        outplotx=((calibFile[:,0]/period)+phaseShift)%1
        outploty=calibFile[:,1]
        outplotxrepeat=outplotx+1
        logger.debug(outplotx)
        logger.debug(outploty)
        plt.xlabel('Phase')
        plt.ylabel('Apparent ' + str(filterCode) + ' Magnitude')
        plt.plot(outplotx,outploty,'bo')
        plt.plot(outplotxrepeat,outploty,'ro')
        #plt.plot(linex,liney)
        plt.ylim(max(outploty)+0.04,min(outploty)-0.04,'k-')
        plt.xlim(-0.01,2.01)
        plt.errorbar(outplotx, outploty, yerr=3*calibFile[:,2], fmt='-o', linestyle='None')
        plt.errorbar(outplotxrepeat, outploty, yerr=3*calibFile[:,2], fmt='-o', linestyle='None')
        plt.grid(True)
        plt.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=0.17, wspace=0.3, hspace=0.4)
        fig.set_size_inches(6,3)
        plt.savefig(outputPath / 'Variable{}_{}_PhasedLightcurve.png'.format(q+1,filterCode))
        plt.savefig(outputPath / 'Variable{}_{}_PhasedLightcurve.eps'.format(q+1,filterCode))

        logger.info("Variable V{}_{}".format(q+1,filterCode))
        logger.info("Max Magnitude: "+ str(np.max(calibFile[:,1])))
        logger.info("Min Magnitude: "+ str(np.min(calibFile[:,1])))
        logger.info("Amplitude    : "+ str(np.max(calibFile[:,1])-np.min(calibFile[:,1])))
        logger.info("Mid Magnitude: "+ str((np.max(calibFile[:,1])+np.min(calibFile[:,1]))/2))

        with open(paths['parent'] / "LightcurveStats.txt", "w") as f:
            f.write("Lightcurve Statistics \n\n")
            f.write("Variable V{}_{}\n".format(str(q+1),filterCode))
            f.write("Max Magnitude: "+ str(np.max(calibFile[:,1]))+"\n")
            f.write("Min Magnitude: "+ str(np.min(calibFile[:,1]))+"\n")
            f.write("Amplitude    : "+ str(np.max(calibFile[:,1])-np.min(calibFile[:,1]))+"\n")
            f.write("Mid Magnitude: "+ str((np.max(calibFile[:,1])+np.min(calibFile[:,1]))/2)+"\n\n")


    return
