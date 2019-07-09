import numpy as np
import os
import shutil


# def check_parent(parentPath):

class AutovarException(Exception):
    ''' Used to halt code with message '''
    pass

def cleanup(parentPath):
    folders = ['calibcats', 'periods', 'checkplots', 'eelbs', 'outputcats','outputplots','trimcats']
    for fd in folders:
        if (parentPath / fd).exists():
            shutil.rmtree(parentPath / fd)

    files = ['calibCompsUsed.csv', 'calibStands.csv', 'compsUsed.csv','screenedComps.csv', \
     'starVariability.csv', 'stdComps.csv', 'usedImages.txt', 'LightcurveStats.txt', \
     'periodEstimates.txt','calibrationErrors.txt']

    for fname in files:
        if (parentPath / fname).exists():
            (parentPath / fname).unlink()
    return

def folder_setup(parentPath):
    #create directory structure for output files
    paths = {
    'parent'     : parentPath,
    'outputPath' : parentPath / "outputplots",
    'outcatPath' : parentPath / "outputcats",
    'checkPath'  : parentPath / "checkplots"
    }
    if not paths['outputPath'].exists():
        os.makedirs(paths['outputPath'])

    if not paths['outcatPath'].exists():
        os.makedirs(paths['outcatPath'])

    if not paths['checkPath'].exists():
        os.makedirs(paths['checkPath'])
    return paths

def photometry_files_to_array(parentPath):
    # Load in list of used files
    fileList=[]
    with open(parentPath / "usedImages.txt", "r") as f:
      for line in f:
        fileList.append(line.strip())

    # LOAD Phot FILES INTO LIST
    photFileArray=[]
    for file in fileList:
        loadPhot=np.genfromtxt(file, dtype=float, delimiter=',')
        if loadPhot.shape[1] > 6:
            loadPhot=np.delete(loadPhot,6,1)
            loadPhot=np.delete(loadPhot,6,1)
        photFileArray.append(loadPhot)
    photFileArray=np.asarray(photFileArray)

    return photFileArray, fileList

def get_targets(targetfile):
    targets = np.genfromtxt(targetfile, dtype=float, delimiter=',')
    # Remove any nan rows from targets
    targetRejecter=[]
    if not (targets.shape[0] == 4 and targets.size ==4):
        for z in range(targets.shape[0]):
          if np.isnan(targets[z][0]):
            targetRejecter.append(z)
        targets=np.delete(targets, targetRejecter, axis=0)
    return targets
