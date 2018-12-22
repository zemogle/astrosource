import numpy
from astropy import units as u
from astropy.coordinates import SkyCoord
import glob
import sys
import os
import logging

logger = logging.getLogger(__name__)

def gather_files(filetype="psx"):
    # Get list of phot files
    parentPath = os.getcwd()
    inputPath = os.path.join(parentPath,"inputs")
    fileList = glob.glob("{}/*.{}".format(inputPath,filetype))
    return fileList

def find_stars(ra, dec):
    fileList = gather_files()
    #Initialisation values
    #exoplanetRows=[]
    acceptDistance=1.0 # Furtherest distance in arcseconds for matches
    minimumCounts=20000 # look for comparisons brighter than this
    maximumCounts=500000 # look for comparisons dimmer than this
    imageFracReject=0.33 # This is a value which will reject images based on number of stars detected
    starFracReject=0.15 # This ia a value which will reject images that reject this fraction of available stars after....
    rejectStart=7 # This many initial images (lots of stars are expected to be rejected in the early images)
    minCompStars=5 # This is the minimum number of comp stars required
    usedImages=[]
    # Generate a blank targetstars.csv file
    targetStars=[(0,0,0,0),(ra,dec,0,0)]
    numpy.savetxt("targetstars.csv", targetStars, delimiter=",", fmt='%0.8f')

    # LOOK FOR REJECTING NON-WCS IMAGES
    # If the WCS matching has failed, this function will remove the image from the list
    #wcsReject=[]
    #q=0
    for file in fileList:
        photFile = numpy.genfromtxt(file, dtype=float, delimiter=',')
        if (( numpy.asarray(photFile[:,0]) > 360).sum() > 0) :
            logger.info("REJECT")
            logger.info(file)
            fileList.remove(file)
        elif (( numpy.asarray(photFile[:,1]) > 90).sum() > 0) :
            logger.info("REJECT")
            logger.info(file)
            fileList.remove(file)
            #q=q+1

    # Sort through and find the largest file and use that as the reference file
    fileSizer=0
    logger.info("Finding image with most stars detected")
    for file in fileList:
        photFile = numpy.genfromtxt(file, dtype=float, delimiter=',')
        if photFile.size > fileSizer:
            referenceFrame=photFile
            logger.info(photFile.size)
            fileSizer=photFile.size

    logger.info("Setting up reference Frame")
    fileRaDec = SkyCoord(ra=referenceFrame[:,0]*u.degree, dec=referenceFrame[:,1]*u.degree)

    logger.info("Removing stars with low or high counts")
    rejectStars=[]
    # Check star has adequate counts
    for j in range(referenceFrame.shape[0]):
        if ( referenceFrame[j][4] < minimumCounts or referenceFrame[j][4] > maximumCounts ):
            rejectStars.append(int(j))
    logger.info("Number of stars prior")
    logger.info(referenceFrame.shape[0])
    referenceFrame=numpy.delete(referenceFrame, rejectStars, axis=0)
    logger.info("Number of stars post")
    logger.info(referenceFrame.shape[0])
    #logger.info(referenceFrame)


    imgsize=imageFracReject * fileSizer # set threshold size
    rejStartCounter=0
    imgReject=0 # Number of images rejected due to high rejection rate
    loFileReject=0 # Number of images rejected due to too few stars in the photometry file
    for file in fileList:
        rejStartCounter=rejStartCounter +1
        photFile = numpy.genfromtxt(file, dtype=float, delimiter=',')
        fileRaDec = SkyCoord(ra=photFile[:,0]*u.degree, dec=photFile[:,1]*u.degree)

        logger.info('Image Number: ' + str(rejStartCounter))
        logger.info(file)
        logger.info("Image treshold size: "+str(imgsize))
        logger.info("Image catalogue size: "+str(photFile.size))
        if photFile.size > imgsize:

            # Checking existance of stars in all photometry files
            rejectStars=[] # A list to hold what stars are to be rejected

            # Find whether star in reference list is in this phot file, if not, reject star.
            for j in range(referenceFrame.shape[0]):
                photRAandDec=SkyCoord(ra=photFile[:,0]*u.degree, dec=photFile[:,1]*u.degree)
                testStar=SkyCoord(ra=referenceFrame[j][0]*u.degree, dec=referenceFrame[j][1]*u.degree)
                _, d2d, _ = testStar.match_to_catalog_sky(photRAandDec)
                if (d2d.arcsecond > acceptDistance):
                    #"No Match! Nothing within range."
                    rejectStars.append(int(j))


            # if the rejectstar list is not empty, remove the stars from the reference List
            if rejectStars != []:

                if not (((len(rejectStars) / referenceFrame.shape[0]) > starFracReject) and rejStartCounter > rejectStart):
                    referenceFrame=numpy.delete(referenceFrame, rejectStars, axis=0)
                    logger.info('**********************')
                    logger.info('Stars Removed  : ' +str(len(rejectStars)))
                    logger.info('Remaining Stars: ' +str(referenceFrame.shape[0]))
                    logger.info('**********************')
                    usedImages.append(file)
                else:
                    logger.info('**********************')
                    logger.info('Image Rejected due to too high a fraction of rejected stars')
                    logger.info(len(rejectStars) / referenceFrame.shape[0])
                    logger.info('**********************')
                    imgReject=imgReject+1
            else:
                logger.info('**********************')
                logger.info('All Stars Present')
                logger.info('**********************')
                usedImages.append(file)

            # If we have removed all stars, we have failed!
            if (referenceFrame.shape[0]==0):
                logger.info("All Stars Removed. Try removing problematic files or raising the imageFracReject")
                return

            if (referenceFrame.shape[0]< minCompStars):
                logger.info("There are less than the requested number of Comp Stars. Try removing problematic files or raising the imageFracReject")
                return
        else:
            logger.error('**********************')
            logger.error("CONTAINS TOO FEW STARS")
            logger.error('**********************')
            loFileReject=loFileReject+1

    # Construct the output file containing candidate comparison stars
    outputComps=[]
    for j in range (referenceFrame.shape[0]):
        outputComps.append([referenceFrame[j][0],referenceFrame[j][1]])

    logger.info("These are the identified common stars of sufficient brightness that are in every image")
    logger.info(outputComps)

    logger.info(' ')
    logger.info('Images Rejected due to high star rejection rate: ' + str(imgReject))
    logger.info('Images Rejected due to low file size: ' + str(loFileReject))
    logger.info('Out of this many original images: ' + str(len(fileList)))
    logger.info(' ')

    logger.info("Number of candidate Comparison Stars Detected: " + str(len(outputComps)))
    logger.info(' ')
    logger.info('Output sent to screenedComps.csv ready for use in CompDeviation')
    numpy.savetxt("screenedComps.csv", outputComps, delimiter=",", fmt='%0.8f')
    # The list of non-rejected images are saved to this text file and are used throughout the rest of the procedure.
    logger.info('UsedImages ready for use in CompDeviation')
    with open("usedImages.txt", "w") as f:
        for s in usedImages:
            f.write(str(s) +"\n")

    return
