from numpy import genfromtxt, savetxt, asarray, average, isnan, delete, min, max, median
from astropy.coordinates import SkyCoord
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
import os

import logging

from astrosource.utils import photometry_files_to_array, AstrosourceException

logger = logging.getLogger('astrosource')


def output_files(paths, photometrydata, mode='diff'):
    if mode == 'calib' and not (paths['parent'] / 'calibCompsUsed.csv').exists():
        raise AstrosourceException("No calibrated photometry available")

    for j, outputPhot in enumerate(photometrydata):
        r = j+1
        logger.info("Outputting files Variable " + str(r))

        if mode =='calib':
            calibIndex=asarray(outputPhot).shape[1]-1
            magColumn=outputPhot[:,calibIndex-1]
            magerrColumn=outputPhot[:,calibIndex]
        else:
            magColumn=outputPhot[:,10]
            magerrColumn=outputPhot[:,11]
            

        outputPeransoCalib = [x for x in zip(outputPhot[:,6],magColumn,magerrColumn)]

        savetxt(paths['outcatPath'] / f'V{r}_{mode}Peranso.txt', outputPeransoCalib, delimiter=" ", fmt='%0.8f')
        savetxt(paths['outcatPath'] / f'V{r}_{mode}Excel.csv', outputPeransoCalib, delimiter=",", fmt='%0.8f')

        #output for EXOTIC modelling
        outputEXOTICCalib = [x for x in zip(outputPhot[:,6],magColumn,magerrColumn,outputPhot[:,7])]

       
        outputEXOTICCalib=asarray(outputEXOTICCalib)
        exoMedian=median(outputEXOTICCalib[:,1])
        #outputEXOTICCalib[:,1]=(outputEXOTICCalib[:,1]-numpy.median(outputEXOTICCalib[:,1]))
        for q in range (outputEXOTICCalib.shape[0]):
            #print (PhotFile[q][0])
            outputEXOTICCalib[q][1]=(1-pow(10,((outputEXOTICCalib[q][1]-exoMedian)/2.5)))+1
            outputEXOTICCalib[q][2]=(outputEXOTICCalib[q][2]/1.0857)*outputEXOTICCalib[q][1]

        savetxt(paths['outcatPath'] / f'V{r}_{mode}EXOTIC.csv', outputEXOTICCalib, delimiter=",", fmt='%0.8f')

        # Output Differential astroImageJ file
        outputaijCalib = [x for x in zip(outputPhot[:,6]-2450000.0,magColumn,magerrColumn)]

        savetxt(paths['outcatPath'] / f'V{r}_{mode}AIJ.txt', outputaijCalib, delimiter=" ", fmt='%0.8f')
        savetxt(paths['outcatPath'] / f'V{r}_{mode}AIJ.csv', outputaijCalib, delimiter=",", fmt='%0.8f')
    return


def open_photometry_files(outcatPath):
    fileList = outcatPath.glob("doer*.csv")
    photometrydata = []
    for file in fileList:
        outputPhot=genfromtxt(file, delimiter=",", dtype='float')
        photometrydata.append(outputPhot)
    return photometrydata


def plot_variability(output, parentPath):
    # star Variability Plot

    plt.cla()
    outplotx = asarray(output)[:, 2]
    outploty = asarray(output)[:, 3]
    plt.xlabel('Mean Differential Magnitude of a Given Star')
    plt.ylabel('Standard Deviation of Differential Magnitudes')
    plt.plot(outplotx, outploty, 'bo')
    # plt.plot(linex, liney)
    plt.ylim(min(outploty)-0.04, max(outploty)+0.04, 'k-')
    plt.xlim(min(outplotx)-0.1, max(outplotx)+0.1)
    plt.grid(True)
    plt.savefig(parentPath / 'starVariability.png')
    plt.savefig(parentPath / 'starVariability.eps')
    return


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
            plt.savefig(paths['outputPath'] / f'V{r}_EnsembleVarDiffMag.png')
        if fileformat == 'full' or fileformat == 'eps':
            plt.savefig(paths['outputPath'] / f'V{r}_EnsembleVarDiffMag.eps')

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
            plt.savefig(paths['checkPath'] / f'V{r}_AirmassEnsVarDiffMag.png')
        if fileformat == 'full' or fileformat == 'eps':
            plt.savefig(paths['checkPath'] / f'V{r}_AirmassEnsVarDiffMag.eps')

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
            plt.savefig(paths['checkPath'] / f'V{r}_AirmassVarCounts.png')
        if fileformat == 'full' or fileformat == 'eps':
            plt.savefig(paths['checkPath'] / f'V{r}_AirmassVarCounts.eps')

    return


def make_calibrated_plots(filterCode, paths, photometrydata):
    # Make a calibrated version
    # Need to shift the shape of the curve against the lowest error in the catalogue.
    for j, outputPhot in enumerate(photometrydata):
        calibIndex=asarray(outputPhot).shape[1]-2
        plt.cla()
        outplotx = asarray(outputPhot)[:, 6]
        outploty = asarray(outputPhot)[:, calibIndex]
        plt.xlabel('BJD')
        plt.ylabel(f'Calibrated {filterCode} Mag')
        plt.plot(outplotx, outploty, 'bo')
        plt.ylim(max(outploty)+0.02, min(outploty)-0.02, 'k-')
        plt.xlim(min(outplotx)-0.01, max(outplotx)+0.01)
        plt.grid(True)
        plt.savefig(paths['outputPath'] / f'V{j+1}_EnsembleVarCalibMag.png')
        plt.savefig(paths['outputPath'] / f'V{j+1}_EnsembleVarCalibMag.eps')

    return


def phased_plots(paths, filterCode, targets, period, phaseShift):

    # Load in list of used files
    fileList = []
    for line in (paths['parent'] / "usedImages.txt").read_text():
        fileList.append(line.strip())

    fileList = paths['parent'].glob("*.p*")
    outputPath = paths['parent'] / 'outputplots'

    for q, target in enumerate(targets):
        filename = paths['outcatPath'] / 'V{}_calibExcel.csv'.format(q+1)
        calibFile = genfromtxt(filename, dtype=float, delimiter=',')
        for i in range(calibFile.shape[0]):
            calibFile[i][1] = calibFile[i][1]

        # Variable lightcurve

        plt.cla()
        outplotx = calibFile[:, 0]
        outploty = calibFile[:, 1]
        plt.xlabel('BJD')
        plt.ylabel('Apparent {} Magnitude'.format(filterCode))
        plt.plot(outplotx, outploty, 'bo')
        # plt.plot(linex, liney)
        plt.ylim(max(outploty)-0.04, min(outploty)+0.04, 'k-')
        plt.xlim(min(outplotx)-0.01, max(outplotx)+0.01)
        plt.grid(True)
        plt.savefig(outputPath / 'Variable{}_{}_Lightcurve.png'.format(q+1,filterCode))
        plt.savefig(outputPath / 'Variable{}_{}_Lightcurve.eps'.format(q+1,filterCode))

        # Phased lightcurve

        plt.cla()
        fig = plt.gcf()
        outplotx=((calibFile[:,0]/period)+phaseShift)%1
        outploty=calibFile[:,1]
        outplotxrepeat=outplotx+1
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
        plt.subplots_adjust(left=0.15, right=0.98, top=0.98, bottom=0.17, wspace=0.3, hspace=0.4)
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
