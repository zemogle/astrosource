from numpy import genfromtxt, savetxt, asarray, average, isnan, delete, min, max
from astropy.coordinates import SkyCoord
from pathlib import Path

import matplotlib.pyplot as plt
import math
import os

import logging

from astrosource.utils import photometry_files_to_array, AstrosourceException

logger = logging.getLogger('astrosource')


def output_files(paths, photometrydata):
    calibFlag = True if (paths['parent'] / 'calibCompsUsed.csv').exists() else False

    for j, outputPhot in enumerate(photometrydata):
        r = j+1
        logger.info("Outputting files Variable " + str(r))

        # Output Differential peranso file
        outputPeransoCalib=[]
        for i in range(outputPhot.shape[0]):
            outputPeransoCalib.append([outputPhot[i][6],outputPhot[i][10],outputPhot[i][11]])

        savetxt(paths['outcatPath'] / 'V{}_diffPeranso.txt'.format(r), outputPeransoCalib, delimiter=" ", fmt='%0.8f')
        savetxt(paths['outcatPath'] / 'V{}_diffExcel.csv'.format(r), outputPeransoCalib, delimiter=",", fmt='%0.8f')

        # Output Differential astroImageJ file
        outputPeransoCalib=[]
        for i in range(asarray(outputPhot).shape[0]):
            outputPeransoCalib.append([outputPhot[i][6]-2450000.0,outputPhot[i][10],outputPhot[i][11]])
        savetxt(paths['outcatPath'] / 'V{}_diffAIJ.txt'.format(r), outputPeransoCalib, delimiter=" ", fmt='%0.8f')
        savetxt(paths['outcatPath'] / 'V{}_diffAIJ.csv'.format(r), outputPeransoCalib, delimiter=",", fmt='%0.8f')

    if calibFlag:
        # Output Calibed peranso file
        outputPeransoCalib=[]
        for i in range(outputPhot.shape[0]):
            outputPeransoCalib.append([outputPhot[i][6],outputPhot[i][10],outputPhot[i][11]])
        savetxt(paths['outcatPath'] / f'V{r}_calibPeranso.txt', outputPeransoCalib, delimiter=" ", fmt='%0.8f')
        savetxt(paths['outcatPath'] / f'V{r}_calibExcel.csv', outputPeransoCalib, delimiter=",", fmt='%0.8f')

        # Output astroImageJ file
        outputPeransoCalib=[]
        for i in range(asarray(outputPhot).shape[0]):
            outputPeransoCalib.append([outputPhot[i][6]-2450000.0,outputPhot[i][10],outputPhot[i][11]])
            #i=i+1

        savetxt(paths['outcatPath'] / f'V{r}_calibAIJ.txt', outputPeransoCalib, delimiter=" ", fmt='%0.8f')
        savetxt(paths['outcatPath'] / f'V{r}_calibAIJ.csv', outputPeransoCalib, delimiter=",", fmt='%0.8f')

    return

def open_photometry_files(outcatPath):
    fileList = outcatPath.glob("doer*.csv")
    photometrydata = []
    for file in fileList:
        outputPhot=genfromtxt(file, delimiter=",", dtype='float')
        photometrydata.append(outputPhot)
    return photometrydata

def make_plots(filterCode, paths, photometrydata, fileformat='full'):

    for j, outputPhot in enumerate(photometrydata):
        r = j + 1
        plt.cla()
        outplotx=asarray(outputPhot)[:,6]
        outploty=asarray(outputPhot)[:,10]
        plt.xlabel('BJD')
        plt.ylabel('Differential ' +filterCode+' Mag')
        plt.plot(outplotx,outploty,'bo')
        plt.ylim(max(outploty)+0.02,min(outploty)-0.02,'k-')
        plt.xlim(min(outplotx)-0.01,max(outplotx)+0.01)
        plt.grid(True)
        if fileformat == 'full' or fileformat == 'png':
            plt.savefig(paths['outputPath'] / f'{r}_EnsembleVarDiffMag.png')
        if fileformat == 'full' or fileformat == 'eps':
            plt.savefig(paths['outputPath'] / f'{r}_EnsembleVarDiffMag.eps')

        plt.cla()
        outplotx=asarray(outputPhot)[:,7]
        outploty=asarray(outputPhot)[:,10]
        plt.xlabel('Airmass')
        plt.ylabel(f'Differential {filterCode} Mag')
        plt.plot(outplotx,outploty,'bo')
        plt.ylim(min(outploty)-0.02,max(outploty)+0.02,'k-')
        plt.xlim(min(outplotx)-0.01,max(outplotx)+0.01)
        plt.grid(True)
        if fileformat == 'full' or fileformat == 'png':
            plt.savefig(paths['checkPath'] / f'{r}_AirmassEnsVarDiffMag.png')
        if fileformat == 'full' or fileformat == 'eps':
            plt.savefig(paths['checkPath'] / f'{r}_AirmassEnsVarDiffMag.eps')

        plt.cla()
        outplotx=asarray(outputPhot)[:,7]
        outploty=asarray(outputPhot)[:,8]
        plt.xlabel('Airmass')
        plt.ylabel('Variable Counts')
        plt.plot(outplotx,outploty,'bo')
        plt.ylim(min(outploty)-1000,max(outploty)+1000,'k-')
        plt.xlim(min(outplotx)-0.01,max(outplotx)+0.01)
        plt.grid(True)
        if fileformat == 'full' or fileformat == 'png':
            plt.savefig(paths['checkPath'] / f'{r}_AirmassVarCounts.png')
        if fileformat == 'full' or fileformat == 'eps':
            plt.savefig(paths['checkPath'] / f'{r}_AirmassVarCounts.eps')

    return

def make_calibrated_plots(filterCode, paths, photometrydata):
    # Make a calibrated version
    # Need to shift the shape of the curve against the lowest error in the catalogue.
    for j, outputPhot in enumerate(photometrydata):
        r = j + 1
        calibCompFile=genfromtxt(paths['parent'] / 'calibCompsUsed.csv', dtype=float, delimiter=',')
        compFile = genfromtxt(paths['parent'] / 'stdComps.csv', dtype=float, delimiter=',')
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
            #ensembleMagError=average(ensembleMagError)*1/pow(ensembleMagError.size, 0.5)
        else:
            ensembleMagError=calibCompFile[:,4]
            ensembleMagError=average(ensembleMagError)*1/pow(ensembleMagError.size, 0.5)

        #for file in fileList:
        for i in range(outputPhot.shape[0]):
            outputPhot[i][10]=outputPhot[i][10]+ensembleMag

        plt.cla()
        outplotx=asarray(outputPhot)[:,6]
        outploty=asarray(outputPhot)[:,10]
        plt.xlabel('BJD')
        plt.ylabel('Calibrated ' +filterCode+' Mag')
        plt.plot(outplotx,outploty,'bo')
        plt.ylim(max(outploty)+0.02,min(outploty)-0.02,'k-')
        plt.xlim(min(outplotx)-0.01,max(outplotx)+0.01)
        plt.grid(True)
        plt.savefig(paths['outputPath'] / 'V{}_EnsembleVarCalibMag.png'.format(r))
        plt.savefig(paths['outputPath'] / 'V{}_EnsembleVarCalibMag.eps'.format(r))

    return

def phased_plots(paths, filterCode, targets, period, phaseShift):

    # Load in list of used files
    fileList=[]
    for line in (paths['parent'] / "usedImages.txt").read_text():
        fileList.append(line.strip())

    fileList = paths['parent'].glob("*.p*")
    #logger.debug(fileList)
    outputPath = paths['parent'] / 'outputplots'

    for q, target in enumerate(targets):
        filename = paths['outcatPath'] / 'V{}_calibExcel.csv'.format(q+1)
        calibFile = genfromtxt(filename, dtype=float, delimiter=',')

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
        fig = plt.gcf()
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
        logger.info("Max Magnitude: "+ str(max(calibFile[:,1])))
        logger.info("Min Magnitude: "+ str(min(calibFile[:,1])))
        logger.info("Amplitude    : "+ str(max(calibFile[:,1])-min(calibFile[:,1])))
        logger.info("Mid Magnitude: "+ str((max(calibFile[:,1])+min(calibFile[:,1]))/2))

        with open(paths['parent'] / "LightcurveStats.txt", "w") as f:
            f.write("Lightcurve Statistics \n\n")
            f.write("Variable V{}_{}\n".format(str(q+1),filterCode))
            f.write("Max Magnitude: "+ str(max(calibFile[:,1]))+"\n")
            f.write("Min Magnitude: "+ str(min(calibFile[:,1]))+"\n")
            f.write("Amplitude    : "+ str(max(calibFile[:,1])-min(calibFile[:,1]))+"\n")
            f.write("Mid Magnitude: "+ str((max(calibFile[:,1])+min(calibFile[:,1]))/2)+"\n\n")


    return
