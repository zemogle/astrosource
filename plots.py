import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
import glob
import sys
import matplotlib
from pathlib import Path
matplotlib.use("TkAgg") # must be before pyplot

# import matplotlib.pyplot as plt
import pylab
import math
import os

import logging

from utils import photometry_files_to_array

logger = logging.getLogger(__name__)

def make_plots(filterCode, parentPath):
    outputPath = parentPath / "outputplots"
    outcatPath = parentPath / "outputcats"
    checkPath = parentPath / "checkplots"

    if (parentPath / 'calibCompsUsed.csv').exists():
        calibFlag=1
    else:
        calibFlag=0

    #create directory structure
    if not outputPath.exists():
        os.makedirs(outputPath)

    if not outcatPath.exists():
        os.makedirs(outcatPath)

    if not checkPath.exists():
        os.makedirs(checkPath)

    fileList = outcatPath.glob("doer*.csv")
    # print([f for f in fileList])

    for file in fileList:

        outputPhot=np.genfromtxt(file, delimiter=",", dtype='float')
        r = file.stem.split("_")[-1]
        print ("Making Plots and Catalogues for Variable " + str(r))

        # Output Differential peranso file
        outputPeransoCalib=[]
        for i in range(outputPhot.shape[0]):
            outputPeransoCalib.append([outputPhot[i][6],outputPhot[i][10],outputPhot[i][11]])

        np.savetxt(os.path.join(outcatPath,str(r)+'_'+"diffPeranso.txt"), outputPeransoCalib, delimiter=" ", fmt='%0.8f')
        np.savetxt(os.path.join(outcatPath,str(r)+'_'+"diffExcel.csv"), outputPeransoCalib, delimiter=",", fmt='%0.8f')

        # Output Differential astroImageJ file
        outputPeransoCalib=[]
        for i in range(np.asarray(outputPhot).shape[0]):
            outputPeransoCalib.append([outputPhot[i][6]-2450000.0,outputPhot[i][10],outputPhot[i][11]])
        np.savetxt(os.path.join(outcatPath,str(r)+'_'+"diffAIJ.txt"), outputPeransoCalib, delimiter=" ", fmt='%0.8f')
        np.savetxt(os.path.join(outcatPath,str(r)+'_'+"diffAIJ.csv"), outputPeransoCalib, delimiter=",", fmt='%0.8f')

        pylab.cla()
        outplotx=np.asarray(outputPhot)[:,6]
        outploty=np.asarray(outputPhot)[:,10]
        pylab.xlabel('BJD')
        pylab.ylabel('Differential ' +filterCode+' Mag')
        pylab.plot(outplotx,outploty,'bo')
        pylab.ylim(max(outploty)+0.02,min(outploty)-0.02,'k-')
        pylab.xlim(min(outplotx)-0.01,max(outplotx)+0.01)
        pylab.grid(True)
        pylab.savefig(os.path.join(outputPath,str(r)+'_'+'EnsembleVarDiffMag.png'))
        pylab.savefig(os.path.join(outputPath,str(r)+'_'+'EnsembleVarDiffMag.eps'))

        pylab.cla()
        outplotx=np.asarray(outputPhot)[:,7]
        outploty=np.asarray(outputPhot)[:,10]
        pylab.xlabel('Airmass')
        pylab.ylabel('Differential ' +filterCode+' Mag')
        pylab.plot(outplotx,outploty,'bo')
        pylab.ylim(min(outploty)-0.02,max(outploty)+0.02,'k-')
        pylab.xlim(min(outplotx)-0.01,max(outplotx)+0.01)
        pylab.grid(True)
        pylab.savefig(os.path.join(checkPath,str(r)+'_'+'AirmassEnsVarDiffMag.png'))
        pylab.savefig(os.path.join(checkPath,str(r)+'_'+'AirmassEnsVarDiffMag.eps'))

        pylab.cla()
        outplotx=np.asarray(outputPhot)[:,7]
        outploty=np.asarray(outputPhot)[:,8]
        pylab.xlabel('Airmass')
        pylab.ylabel('Variable Counts')
        pylab.plot(outplotx,outploty,'bo')
        pylab.ylim(min(outploty)-1000,max(outploty)+1000,'k-')
        pylab.xlim(min(outplotx)-0.01,max(outplotx)+0.01)
        pylab.grid(True)
        pylab.savefig(os.path.join(checkPath,str(r)+'_'+'AirmassVarCounts.png'))
        pylab.savefig(os.path.join(checkPath,str(r)+'_'+'AirmassVarCounts.eps'))

        # Make a calibrated version
        # Need to shift the shape of the curve against the lowest error in the catalogue.

        if calibFlag == 1:
            calibCompFile=np.genfromtxt('calibCompsUsed.csv', dtype=float, delimiter=',')
            print ("Calibrating Photometry")
            # Load in calibrated magnitudes and add them
            #print (compFile.size)
            if compFile.shape[0] == 5 and compFile.size != 25:
                ensembleMag=calibCompFile[3]
            else:
                ensembleMag=calibCompFile[:,3]
            ensMag=0

            if compFile.shape[0] == 5 and compFile.size != 25:
                lenloop=1
            else:
                lenloop=len(calibCompFile[:,3])
            for r in range(lenloop):
                if compFile.shape[0] == 5 and compFile.size != 25:
                    ensMag=pow(10,-ensembleMag*0.4)
                else:
                    ensMag=ensMag+(pow(10,-ensembleMag[r]*0.4))
            #print (ensMag)
            ensembleMag=-2.5*math.log10(ensMag)
            print ("Ensemble Magnitude: "+str(ensembleMag))


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


            pylab.cla()
            outplotx=np.asarray(outputPhot)[:,6]
            outploty=np.asarray(outputPhot)[:,10]
            pylab.xlabel('BJD')
            pylab.ylabel('Calibrated ' +filterCode+' Mag')
            pylab.plot(outplotx,outploty,'bo')
            pylab.ylim(max(outploty)+0.02,min(outploty)-0.02,'k-')
            pylab.xlim(min(outplotx)-0.01,max(outplotx)+0.01)
            pylab.grid(True)
            pylab.savefig(os.path.join(outputPath,str(r)+'_'+'EnsembleVarCalibMag.png'))
            pylab.savefig(os.path.join(outputPath,str(r)+'_'+'EnsembleVarCalibMag.eps'))


            # Output Calibed peranso file
            outputPeransoCalib=[]
            r = file.split("_")[-1].replace(".csv","")
            for i in range(outputPhot.shape[0]):
                outputPeransoCalib.append([outputPhot[i][6],outputPhot[i][10],outputPhot[i][11]])
            np.savetxt(os.path.join(outcatPath,str(r)+'_'+"calibPeranso.txt"), outputPeransoCalib, delimiter=" ", fmt='%0.8f')
            np.savetxt(os.path.join(outcatPath,str(r)+'_'+"calibExcel.csv"), outputPeransoCalib, delimiter=",", fmt='%0.8f')

            # Output astroImageJ file
            outputPeransoCalib=[]
            for i in range(np.asarray(outputPhot).shape[0]):
                outputPeransoCalib.append([outputPhot[i][6]-2450000.0,outputPhot[i][10],outputPhot[i][11]])
                #i=i+1

            np.savetxt(os.path.join(outcatPath,str(r)+'_'+"calibAIJ.txt"), outputPeransoCalib, delimiter=" ", fmt='%0.8f')
            np.savetxt(os.path.join(outcatPath,str(r)+'_'+"calibAIJ.csv"), outputPeransoCalib, delimiter=",", fmt='%0.8f')
