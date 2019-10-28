from numpy import asarray, savetxt, std, max, min, genfromtxt, load
import sys
import os
import platform
import glob
import matplotlib.pyplot as plt

import logging

from astrosource.utils import photometry_files_to_array, AstrosourceException

logger = logging.getLogger('astrosource')

#########################################

def sortByPhase (phases, fluxes):
    phaseIndices = asarray(phases).argsort()
    sortedPhases = []
    sortedFluxes = []
    for i in range(0,len(phases)):
        sortedPhases.append(phases[phaseIndices[i]])
        sortedFluxes.append(fluxes[phaseIndices[i]])

    return (sortedPhases, sortedFluxes)

#########################################

def normalize(fluxes):

    normalizedFluxes = []

    for flux in fluxes:
        normalizedFlux = (flux - min(fluxes)) / (max(fluxes) - min(fluxes))
        normalizedFluxes.append(normalizedFlux)

    return(normalizedFluxes)

#########################################

def getPhases(julian_dates, fluxes, period):

    phases = []

    for n in range(0, len(julian_dates)):
        phases.append((julian_dates[n] / (period)) % 1)

    (sortedPhases, sortedFluxes) = sortByPhase(phases, fluxes)

    return(sortedPhases, sortedFluxes)

#########################################

# Find the value in array2 corresponding to the minimum value in array1.

def find_minimum(array1, array2):
    value_to_return = 0.0

    minimum = min(array1)

    for i in range(0, len(array1)):
        if (array1[i] == minimum):
          value_to_return = array2[i]

    return (value_to_return, minimum)

#########################################

def sum_distances (sortedPhases, sortedNormalizedFluxes):

    distanceSum = 0.0

    for i in range(0, (len(sortedPhases) - 1)):
        fluxdiff = sortedNormalizedFluxes[i + 1] - sortedNormalizedFluxes[i]
        phasediff = sortedPhases[i + 1] - sortedPhases[i]

    distanceSum = distanceSum + (((fluxdiff ** 2) + (phasediff ** 2)) ** 0.5)

    return(distanceSum)

#########################################

def sum_stdevs (sortedPhases, sortedNormalizedFluxes, numBins):

    stdevSum = 0.0

    for i in range (0, numBins):
        fluxes_inrange = []
        minIndex = (float(i) / float(numBins)) * float(len(sortedPhases))
        maxIndex = (float(i + 1) / float(numBins)) * float(len(sortedPhases))

    for j in range (0, len(sortedPhases)):
      if (j >= minIndex and j < maxIndex):
        fluxes_inrange.append(sortedNormalizedFluxes[j])

    stdev_of_bin_i = std(fluxes_inrange)
    stdevSum = stdevSum + stdev_of_bin_i

    return(stdevSum)

#########################################

def phase_dispersion_minimization(varData, periodsteps, minperiod, maxperiod, numBins, periodPath, variableName):

    periodguess_array = []

    distance_results = []
    stdev_results = []

    (julian_dates, fluxes) = (varData[:,0],varData[:,1])
    normalizedFluxes = normalize(fluxes)
    periodTrialMatrix=[]

    for r in range(periodsteps):
        periodguess = minperiod + (r * ((maxperiod-minperiod)/periodsteps))
        (sortedPhases, sortedNormalizedFluxes) = getPhases(julian_dates, normalizedFluxes, periodguess)

        distance_sum = sum_distances(sortedPhases, sortedNormalizedFluxes)
        stdev_sum = sum_stdevs(sortedPhases, sortedNormalizedFluxes, numBins)

        periodguess_array.append(periodguess)
        distance_results.append(distance_sum)
        stdev_results.append(stdev_sum)
        periodTrialMatrix.append([periodguess,distance_sum,stdev_sum])

    periodTrialMatrix=asarray(periodTrialMatrix)
    savetxt(os.path.join(periodPath,str(variableName)+'_'+"Trials.csv"), periodTrialMatrix, delimiter=",", fmt='%0.8f')

    (distance_minperiod, distance_min) = find_minimum(distance_results, periodguess_array)
    (stdev_minperiod, stdev_min) = find_minimum(stdev_results, periodguess_array)

    pdm = {}
    pdm["periodguess_array"] = periodguess_array
    pdm["distance_results"] = distance_results
    pdm["distance_minperiod"] = distance_minperiod
    pdm["stdev_results"] = stdev_results
    pdm["stdev_minperiod"] = stdev_minperiod

    # Estimating the error
    # stdev method

    # Get deviation to the left
    totalRange=max(stdev_results) - min(stdev_results)
    for q in range(len(periodguess_array)):
        if periodguess_array[q]==pdm["stdev_minperiod"]:
          beginIndex=q
          beginValue=stdev_results[q]
    #logger.debug(beginIndex)
    #logger.debug(beginValue)
    currentperiod=stdev_minperiod
    stepper=0
    thresholdvalue=beginValue+(0.5*totalRange)
    while True:
    #logger.debug(beginIndex-stepper)
    #logger.debug(stdev_results[beginIndex-stepper])
        if stdev_results[beginIndex-stepper] > thresholdvalue:
          #logger.debug("LEFTHAND PERIOD!")
          #logger.debug(periodguess_array[beginIndex-stepper])
          lefthandP=periodguess_array[beginIndex-stepper]
          #logger.debug(distance_results)
          break
        stepper=stepper+1

    stepper=0
    thresholdvalue=beginValue+(0.5*totalRange)
    #logger.debug(beginIndex)
    #logger.debug(periodsteps)

    while True:
    # logger.debug(beginIndex+stepper)
    #logger.debug(stdev_results[beginIndex+stepper])
        if beginIndex+stepper+1 == periodsteps:
          righthandP=periodguess_array[beginIndex+stepper]
          logger.debug("Warning: Peak period for stdev method too close to top of range")
          break
        if stdev_results[beginIndex+stepper] > thresholdvalue:
          #logger.debug("RIGHTHAND PERIOD!")
          #logger.debug(periodguess_array[beginIndex+stepper])
          righthandP=periodguess_array[beginIndex+stepper]
          #logger.debug(distance_results)
          break
        stepper=stepper+1


    #logger.debug("Stdev method error: " + str((righthandP - lefthandP)/2))
    pdm["stdev_error"] = (righthandP - lefthandP)/2


    # Estimating the error
    # stdev method
    #logger.debug(min(stdev_results))
    #logger.debug(max(stdev_results))
    # Get deviation to the left
    totalRange=max(distance_results) - min(distance_results)
    for q in range(len(periodguess_array)):
        if periodguess_array[q]==pdm["distance_minperiod"]:
          beginIndex=q
          beginValue=distance_results[q]
    #logger.debug(beginIndex)
    #logger.debug(beginValue)
    currentperiod=distance_minperiod
    stepper=0
    thresholdvalue=beginValue+(0.5*totalRange)
    while True:
        if distance_results[beginIndex-stepper] > thresholdvalue:

          lefthandP=periodguess_array[beginIndex-stepper]
          break
        stepper=stepper+1

    stepper=0
    thresholdvalue=beginValue+(0.5*totalRange)
    while True:
        if beginIndex+stepper+1 == periodsteps:
          righthandP=periodguess_array[beginIndex+stepper]
          logger.debug("Warning: Peak period for distance method too close to top of range")
          break
        if distance_results[beginIndex+stepper] > thresholdvalue:
          righthandP=periodguess_array[beginIndex+stepper]
          break
        stepper=stepper+1

    pdm["distance_error"] = (righthandP - lefthandP)/2

    return pdm

#########################################

def plot_with_period(paths, filterCode, numBins = 10, minperiod=0.2, maxperiod=1.2, periodsteps=10000):

    trialRange=[minperiod, maxperiod]

    # Get list of phot files
    periodPath = paths['parent'] / "periods"
    if not periodPath.exists():
        os.makedirs(periodPath)

    logger.debug("Filter Set: " + filterCode)

    fileList = paths['outcatPath'].glob('*_diffExcel.csv')
    with open(paths['parent'] / paths['parent'] / "periodEstimates.txt", "w") as f:
      f.write("Period Estimates \n\n")

    # Load in the files
    for file in fileList:
      logger.debug(file)
      variableName=file.stem.split('_')[0]
      #logger.debug(str(outcatPath).replace('//',''))
      logger.debug("Variable Name: {}".format(variableName))
      varData=load(file)
      calibFile = file.parent / "{}{}".format(file.stem.replace('diff','calib'), file.suffix)
      if calibFile.exists():
        calibData=genfromtxt(calibFile, dtype=float, delimiter=',')

      #logger.debug(minDate)

      pdm_results = {}

      pdm=phase_dispersion_minimization(varData, periodsteps, minperiod, maxperiod, numBins, periodPath, variableName)

      plt.figure(figsize=(15, 5))

      logger.debug("Distance Method Estimate (days): " + str(pdm["distance_minperiod"]))
      #logger.debug(pdm["distance_minperiod"] )
      logger.debug("Distance method error: " + str(pdm["distance_error"]))
      phaseTest=(varData[:,0] / (pdm["distance_minperiod"])) % 1
      with open(paths['parent'] / "periodEstimates.txt", "a+") as f:
        f.write("Variable : "+str(variableName) +"\n")
        f.write("Distance Method Estimate (days): " + str(pdm["distance_minperiod"])+"\n")
        f.write("Distance method error: " + str(pdm["distance_error"])+"\n")


      plt.plot(pdm["periodguess_array"], pdm["distance_results"])
      plt.gca().invert_yaxis()
      plt.title("Range {0} d  Steps: {1}".format(trialRange, periodsteps))
      plt.xlabel(r"Trial Period")
      plt.ylabel(r"Likelihood of Period")
      plt.savefig(os.path.join(periodPath,str(variableName)+'_'+"StringLikelihoodPlot.png"))
      plt.clf()

      plt.plot(phaseTest, varData[:,1], 'bo', linestyle='None')
      plt.plot(phaseTest+1, varData[:,1], 'ro', linestyle='None')
      plt.errorbar(phaseTest, varData[:,1], yerr=varData[:,2], linestyle='None')
      plt.errorbar(phaseTest+1, varData[:,1], yerr=varData[:,2], linestyle='None')
      plt.gca().invert_yaxis()
      plt.title("Period: {0} d  Steps: {1}".format(pdm["distance_minperiod"], periodsteps))
      plt.xlabel(r"Phase ($\phi$)")
      plt.ylabel(r"Differential " + str(filterCode) + " Magnitude")
      plt.savefig(os.path.join(periodPath,str(variableName)+'_'+"StringTestPeriodPlot.png"))
      plt.clf()

      if calibFile.exists():
        phaseTestCalib=(calibData[:,0] / (pdm["distance_minperiod"])) % 1
        plt.plot(phaseTestCalib, calibData[:,1], 'bo', linestyle='None')
        plt.plot(phaseTestCalib+1, calibData[:,1], 'ro', linestyle='None')
        plt.errorbar(phaseTestCalib, calibData[:,1], yerr=varData[:,2], linestyle='None')
        plt.errorbar(phaseTestCalib+1, calibData[:,1], yerr=varData[:,2], linestyle='None')
        plt.gca().invert_yaxis()
        plt.title("Period: {0} d  Steps: {1}".format(pdm["distance_minperiod"], periodsteps))
        plt.xlabel(r"Phase ($\phi$)")
        plt.ylabel(r"Calibrated " + str(filterCode) + " Magnitude")
        plt.savefig(os.path.join(periodPath,str(variableName)+'_'+"StringTestPeriodPlot_Calibrated.png"))
        plt.clf()

      tempPeriodCatOut=[]
      for g in range(len(phaseTest)):
        tempPeriodCatOut.append([phaseTest[g],varData[g,1]])
      tempPeriodCatOut=asarray(tempPeriodCatOut)
      savetxt(os.path.join(periodPath,str(variableName)+'_'+"StringTrial.csv"), tempPeriodCatOut, delimiter=",", fmt='%0.8f')

      tempPeriodCatOut=[]
      for g in range(len(calibData[:,0])):
        tempPeriodCatOut.append([(calibData[g,0]/(pdm["distance_minperiod"]) % 1), calibData[g,1], calibData[g,2]])
      tempPeriodCatOut=asarray(tempPeriodCatOut)
      savetxt(os.path.join(periodPath,str(variableName)+'_'+"String_PhasedCalibMags.csv"), tempPeriodCatOut, delimiter=",", fmt='%0.8f')

      tempPeriodCatOut=[]
      for g in range(len(varData[:,0])):
        tempPeriodCatOut.append([(varData[g,0]/(pdm["distance_minperiod"]) % 1), varData[g,1], varData[g,2]])
      tempPeriodCatOut=asarray(tempPeriodCatOut)
      savetxt(os.path.join(periodPath,str(variableName)+'_'+"String_PhasedDiffMags.csv"), tempPeriodCatOut, delimiter=",", fmt='%0.8f')


      logger.debug("PDM Method Estimate (days): "+ str(pdm["stdev_minperiod"]))
      #logger.debug(pdm["stdev_minperiod"])
      phaseTest=(varData[:,0] / (pdm["stdev_minperiod"])) % 1
      logger.debug("PDM method error: " + str(pdm["stdev_error"]))

      with open(paths['parent'] / "periodEstimates.txt", "a+") as f:
        f.write("PDM Method Estimate (days): "+ str(pdm["stdev_minperiod"])+"\n")
        f.write("PDM method error: " + str(pdm["stdev_error"])+"\n\n")


      plt.plot(pdm["periodguess_array"], pdm["stdev_results"])
      plt.gca().invert_yaxis()
      plt.title("Range {0} d  Steps: {1}".format(trialRange, periodsteps))
      plt.xlabel(r"Trial Period")
      plt.ylabel(r"Likelihood of Period")
      plt.savefig(os.path.join(periodPath,str(variableName)+'_'+"PDMLikelihoodPlot.png"))

      plt.clf()


      plt.plot(phaseTest, varData[:,1], 'bo', linestyle='None')
      plt.plot(phaseTest+1, varData[:,1], 'ro', linestyle='None')
      plt.errorbar(phaseTest, varData[:,1], yerr=varData[:,2], linestyle='None')
      plt.errorbar(phaseTest+1, varData[:,1], yerr=varData[:,2], linestyle='None')
      plt.gca().invert_yaxis()
      plt.title("Period: {0} d  Steps: {1}".format(pdm["stdev_minperiod"], periodsteps))
      plt.xlabel(r"Phase ($\phi$)")
      plt.ylabel(r"Differential " + str(filterCode) + " Magnitude")
      plt.savefig(os.path.join(periodPath,str(variableName)+'_'+"PDMTestPeriodPlot.png"))
      plt.clf()

      if calibFile.exists():
        phaseTestCalib=(calibData[:,0] / (pdm["stdev_minperiod"])) % 1
        plt.plot(phaseTestCalib, calibData[:,1], 'bo', linestyle='None')
        plt.plot(phaseTestCalib+1, calibData[:,1], 'ro', linestyle='None')
        plt.errorbar(phaseTestCalib, calibData[:,1], yerr=varData[:,2], linestyle='None')
        plt.errorbar(phaseTestCalib+1, calibData[:,1], yerr=varData[:,2], linestyle='None')
        plt.gca().invert_yaxis()
        plt.title("Period: {0} d  Steps: {1}".format(pdm["stdev_minperiod"], periodsteps))
        plt.xlabel(r"Phase ($\phi$)")
        plt.ylabel(r"Calibrated " + str(filterCode) + " Magnitude")
        plt.savefig(os.path.join(periodPath,str(variableName)+'_'+"PDMTestPeriodPlot_Calibrated.png"))
        plt.clf()

      tempPeriodCatOut=[]
      for g in range(len(phaseTest)):
        tempPeriodCatOut.append([phaseTest[g],varData[g,1]])
      tempPeriodCatOut=asarray(tempPeriodCatOut)
      savetxt(os.path.join(periodPath,str(variableName)+'_'+"PDMTrial.csv"), tempPeriodCatOut, delimiter=",", fmt='%0.8f')

      tempPeriodCatOut=[]
      for g in range(len(calibData[:,0])):
        tempPeriodCatOut.append([(calibData[g,0]/(pdm["stdev_minperiod"])) % 1, calibData[g,1], calibData[g,2]])
      tempPeriodCatOut=asarray(tempPeriodCatOut)
      savetxt(os.path.join(periodPath,str(variableName)+'_'+"PDM_PhasedCalibMags.csv"), tempPeriodCatOut, delimiter=",", fmt='%0.8f')

      tempPeriodCatOut=[]
      for g in range(len(varData[:,0])):
        tempPeriodCatOut.append([(varData[g,0]/(pdm["stdev_minperiod"])) % 1, varData[g,1], varData[g,2]])
      tempPeriodCatOut=asarray(tempPeriodCatOut)
      savetxt(os.path.join(periodPath,str(variableName)+'_'+"PDM_PhaseddiffMags.csv"), tempPeriodCatOut, delimiter=",", fmt='%0.8f')
