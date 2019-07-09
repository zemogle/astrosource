import numpy as np

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
