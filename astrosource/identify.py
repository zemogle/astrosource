import glob
from pathlib import Path
import sys
import os
import logging
import pickle

from numpy import genfromtxt, delete, asarray, save, savetxt, load, transpose, isnan, zeros, max, min, nan, where, average, cos, hstack, array, column_stack
from astropy import units as u
from astropy.units import degree, arcsecond
from astropy import wcs
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.io import fits
from astropy.time import Time
from barycorrpy import utc_tdb

from astrosource.utils import AstrosourceException
from astrosource.comparison import catalogue_call

logger = logging.getLogger('astrosource')

def rename_data_file(prihdr, bjd=False):

    prihdrkeys = prihdr.keys()

    if any("OBJECT" in s for s in prihdrkeys):
        objectTemp=prihdr['OBJECT'].replace('-','d').replace('+','p').replace('.','d').replace(' ','').replace('_','').replace('=','e').replace('(','').replace(')','').replace('<','').replace('>','').replace('/','')
    else:
        objectTemp="UNKNOWN"

    if 'FILTER' in prihdr:
        filterOne=(prihdr['FILTER'])
    else:
        filters = []
        filters.append(prihdr['FILTER1'])
        filters.append(prihdr['FILTER2'])
        filters.append(prihdr['FILTER3'])
        filter =list(set(filters))
        filter.remove('air')
        filterOne = filter[0]

    expTime=(str(prihdr['EXPTIME']).replace('.','d'))
    dateObs=(prihdr['DATE-OBS'].replace('-','d').replace(':','d').replace('.','d'))
    airMass=(str(prihdr['AIRMASS']).replace('.','a'))
    instruMe=(prihdr['INSTRUME']).replace(' ','').replace('/','').replace('-','')



    if (prihdr['MJD-OBS'] == 'UNKNOWN'):
        timeobs = 'UNKNOWN'
    elif bjd:
        timeobs = convert_mjd_bjd(prihdr)
    else:
        timeobs = '{0:.10f}'.format(prihdr['MJD-OBS']).replace('.','d')
    newName=f"{objectTemp}_{filterOne}_{timeobs}_{dateObs}_{airMass}_{expTime}_{instruMe}.npy"

    return newName

def export_photometry_files(filelist, indir, filetype='csv', bjd=False, ignoreedgefraction=0.05, lowestcounts=1800):
    phot_dict = []
    new_files = []
    photFileHolder=[]
    photSkyCoord=[]
    for f in filelist:
        s3 = False
        try:
            fitsobj = Path(f)
        except TypeError:
            fitsobj = f.open()
            s3 = True

        filepath, photFile = extract_photometry(fitsobj, indir, bjd=bjd, ignoreedgefraction=0.05, lowestcounts=1800)
        if s3:
            f.close()
            filename = f.name
        else:
            filename = fitsobj.name
        #print (photFile.size)
        if photFile.size > 100:
            phot_dict.append(filepath)
            
            photFileHolder.append(photFile)
            photSkyCoord.append(SkyCoord(ra=photFile[:,0]*u.degree, dec=photFile[:,1]*u.degree))

    return phot_dict, photFileHolder, photSkyCoord

def extract_photometry(infile, parentPath, outfile=None, bjd=False, ignoreedgefraction=0.05, lowestcounts=1800,  racut=-99.9, deccut=-99.9, radiuscut=-99.9):

    #new_files = []
    #photFileHolder=[]
    #photSkyCoord=[]
    with fits.open(infile) as hdulist:

        if not outfile:
            outfile = rename_data_file(hdulist[1].header, bjd=bjd)
            
        #outname = outfile
        #outfile = parentPath / outfile
        w = wcs.WCS(hdulist[1].header)
        data = hdulist[2].data
        xpixel = data['x']
        ypixel = data['y']
        ra, dec = w.wcs_pix2world(xpixel, ypixel, 1)
        counts = data['flux']
        countserr = data['fluxerr']

        zerosphot=zeros(counts.shape[0], dtype=float)

        photFile = transpose([ra, dec, xpixel, ypixel, counts, countserr, zerosphot, zerosphot])
        photFile = asarray(photFile)
        
        if photFile.size > 16: #ignore zero sized files and files with only one or two entries
            if max(photFile[:,0]) < 360 and max(photFile[:,1]) < 90:
                photFile=photFile[~isnan(photFile).any(axis=1)]
                
                
                # if radial cut do that otherwise chop off image edges
                if racut != -99.9 and deccut !=-99.9 and radiuscut !=-99.9:
                    #tempCoord = SkyCoord(ra=photFile[:,0]*u.degree, dec=photFile[:,1]*u.degree)
                    #logger.info("doing a radial cut")
                    #matchCoord = SkyCoord(ra=float(racut)*u.degree, dec=float(deccut)*u.degree)
                    #idx, d2d, _ = matchCoord.match_to_catalog_sky(tempCoord)
                    #print (idx)
                    distanceArray=pow((pow(photFile[:,0]-float(racut),2)+pow(photFile[:,1]-float(deccut),2)),0.5)
                    #print (distanceArray)
                    #print (distanceArray[distanceArray > float(radiuscut)/60])
                    #print (where(distanceArray > float(radiuscut)/60))
                    photFile=delete(photFile, where(distanceArray > float(radiuscut)/60), axis=0)
                else:   
                
                    # Remove edge detections
                    raRange=(max(photFile[:,0])-min(photFile[:,0]))
                    decRange=(max(photFile[:,1])-min(photFile[:,1]))
                    raMin=min(photFile[:,0])
                    raMax=max(photFile[:,0])
                    decMin=min(photFile[:,1])
                    decMax=max(photFile[:,1])                
                    raClip=raRange*ignoreedgefraction
                    decClip=decRange*ignoreedgefraction
                    photFile[:,0][photFile[:,0] < raMin + raClip ] = nan
                    photFile[:,0][photFile[:,0] > raMax - raClip ] = nan
                    photFile[:,1][photFile[:,1] > decMax - decClip ] = nan
                    photFile[:,1][photFile[:,1] < decMin + decClip ] = nan     
                #remove odd zero entries
                photFile[:,0][photFile[:,0] == 0.0 ] = nan
                photFile[:,0][photFile[:,0] == 0.0 ] = nan
                photFile[:,1][photFile[:,1] == 0.0 ] = nan
                photFile[:,1][photFile[:,1] == 0.0 ] = nan  
                
                photFile=photFile[~isnan(photFile).any(axis=1)]
                
                #remove lowcounts
                rejectStars=where(photFile[:,4] < lowestcounts)[0]
                #print (d2d)
                #print (max_sep)
                photFile=delete(photFile, rejectStars, axis=0)

                

                #new_files.append(outname)
               # photFileHolder.append(photFile)
                #photSkyCoord.append(SkyCoord(ra=photFile[:,0]*u.degree, dec=photFile[:,1]*u.degree))

        #save(outfile, photFile)
        
        #print (len(photFileHolder))

    return outfile, photFile

def convert_photometry_files(filelist, ignoreedgefraction=0.05, lowestcounts=1800,  racut=-99.9, deccut=-99.9, radiuscut=-99.9):
    
    
    
    new_files = []
    photFileHolder=[]
    photSkyCoord=[]
    for fn in filelist:
        photFile = genfromtxt(fn, dtype=float, delimiter=',')
        # reject nan entries in file
        if photFile.size > 16: #ignore zero sized files and files with only one or two entries
            if max(photFile[:,0]) < 360 and max(photFile[:,1]) < 90:
                if (photFile.size > 50):
                    if not (( asarray(photFile[:,0]) > 360).sum() > 0) :
                        if not(( asarray(photFile[:,1]) > 90).sum() > 0) :               
                
                            photFile=photFile[~isnan(photFile).any(axis=1)]
                            

                            
                            # if radial cut do that otherwise chop off image edges
                            if racut != -99.9 and deccut !=-99.9 and radiuscut !=-99.9:
                                #tempCoord = SkyCoord(ra=photFile[:,0]*u.degree, dec=photFile[:,1]*u.degree)
                                #logger.info("doing a radial cut")
                                #matchCoord = SkyCoord(ra=float(racut)*u.degree, dec=float(deccut)*u.degree)
                                #idx, d2d, _ = matchCoord.match_to_catalog_sky(tempCoord)
                                #print (idx)
                                distanceArray=pow((pow(photFile[:,0]-float(racut),2)+pow(photFile[:,1]-float(deccut),2)),0.5)
                                #print (distanceArray)
                                #print (distanceArray[distanceArray > float(radiuscut)/60])
                                #print (where(distanceArray > float(radiuscut)/60))
                                photFile=delete(photFile, where(distanceArray > float(radiuscut)/60), axis=0)
                            else:                            
                            
                                # Remove edge detections
                                raRange=(max(photFile[:,0])-min(photFile[:,0]))
                                decRange=(max(photFile[:,1])-min(photFile[:,1]))
                                raMin=min(photFile[:,0])
                                raMax=max(photFile[:,0])
                                decMin=min(photFile[:,1])
                                decMax=max(photFile[:,1])                
                                raClip=raRange*ignoreedgefraction
                                decClip=decRange*ignoreedgefraction
                                photFile[:,0][photFile[:,0] < raMin + raClip ] = nan
                                photFile[:,0][photFile[:,0] > raMax - raClip ] = nan
                                photFile[:,1][photFile[:,1] > decMax - decClip ] = nan
                                photFile[:,1][photFile[:,1] < decMin + decClip ] = nan   
                            #remove odd zero entries
                            photFile[:,0][photFile[:,0] == 0.0 ] = nan
                            photFile[:,0][photFile[:,0] == 0.0 ] = nan
                            photFile[:,1][photFile[:,1] == 0.0 ] = nan
                            photFile[:,1][photFile[:,1] == 0.0 ] = nan  
                            photFile=photFile[~isnan(photFile).any(axis=1)]
                            
                            #remove lowcounts
                            rejectStars=where(photFile[:,4] < lowestcounts)[0]
                            #print (d2d)
                            #print (max_sep)
                            photFile=delete(photFile, rejectStars, axis=0)
                            #print (len(rejectStars))                                           
                            
                            
                            filepath = Path(fn).with_suffix('.npy')
                            #save(filepath, photFile)
                            if photFile.size > 16:
                                new_files.append(filepath.name)
                                photFileHolder.append(photFile)
                                photSkyCoord.append(SkyCoord(ra=photFile[:,0]*u.degree, dec=photFile[:,1]*u.degree))
                                
                        else:
                            logger.debug("REJECT")
                            logger.debug(fn)
                    else:
                        logger.debug("REJECT")
                        logger.debug(fn)
                else:
                    logger.debug("REJECT")
                    logger.debug(fn)
            else:
                logger.debug("REJECT")
                logger.debug(fn)
        else:
            logger.debug("REJECT")
            logger.debug(fn)
            
    #sys.exit()
    if new_files ==[] or photFileHolder==[] :
        raise AstrosourceException("Either there are no files of this photometry type or radial Cut seems to be outside of the range of your images... check your racut, deccut and radiuscut")
    #print (new_files)
    #print (photFileHolder)
    
    return new_files, photFileHolder, photSkyCoord

def convert_mjd_bjd(hdr):
    pointing = SkyCoord(hdr['RA'], hdr['DEC'], unit=(u.degree, u.degree), frame='icrs')
    location = EarthLocation.from_geodetic(hdr['LONGITUD'], hdr['LATITUDE'], hdr['HEIGHT'])
    t = Time(hdr['MJD-OBS'], format='mjd',scale='utc', location=location)

    tdbholder= (utc_tdb.JDUTC_to_BJDTDB(t, ra=float(pointing.ra.degree), dec=float(pointing.dec.degree), lat=hdr['LATITUDE'], longi=hdr['LONGITUD'], alt=hdr['HEIGHT'], leap_update=True))

    return tdbholder[0][0]


def gather_files(paths, filelist=None, filetype="fz", bjd=False, ignoreedgefraction=0.05, lowest=1800,  racut=-99.9, deccut=-99.9, radiuscut=-99.9):
    # Get list of files
    sys.stdout.write('💾 Inspecting input files\n')

    #print (ignoreedgefraction)
    #sys.exit()



    # Remove old npy files
    #for fname in paths['parent'].glob("*.npy"):
    #    os.remove(fname)

    if not filelist:
        if filetype not in ['fits', 'fit', 'fz']:
            filelist = paths['parent'].glob("*.{}".format(filetype))
        else:
            filelist = paths['parent'].glob("*e91*.{}".format(filetype)) # Make sure only fully reduced LCO files are used.
    if filetype not in ['fits', 'fit', 'fz']:
        # Assume we are not dealing with image files but photometry files
        phot_list, photFileHolder, photSkyCoord = convert_photometry_files(filelist, ignoreedgefraction, lowestcounts=lowest, racut=racut, deccut=deccut, radiuscut=radiuscut)
    else:

        phot_list, photFileHolder, photSkyCoord = export_photometry_files(filelist, paths['parent'], bjd=bjd, ignoreedgefraction=ignoreedgefraction, lowestcounts=lowest, racut=racut, deccut=deccut, radiuscut=radiuscut)
        #Convert phot_list from dict to list
        #phot_list_temp = phot_list_temp.keys()
        #phot_list = []
        #for key in phot_list_temp:
        #    phot_list.append(key) #SLAERT: convert dict to just the list of npy files.
        

    if not phot_list:
        raise AstrosourceException("No files of type '.{}' found in {}".format(filetype, paths['parent']))
    
    #print (phot_list)
    filters = set([f.split('_')[1] for f in phot_list])

    filterCode = list(filters)[0]

    if filterCode  == 'clear' or filterCode  == 'air':
        filterCode  = 'CV'

    logger.debug("Filter Set: {}".format(filterCode))
    
    
    
    if len(filters) > 1:
        raise AstrosourceException("Check your images, the script detected multiple filters in your file list. Astrosource currently only does one filter at a time.")
    
    file1=open(paths['parent'] / "filterCode","wb")
    pickle.dump(filterCode, file1)
    file1.close
        
    return phot_list, filterCode, photFileHolder, photSkyCoord

def find_stars(targets, paths, fileList, nopanstarrs=False, nosdss=False, closerejectd=5.0, photCoords=None, photFileHolder=None, mincompstars=0.1, mincompstarstotal=-99, starreject=0.1 , acceptDistance=1.0, lowcounts=2000, hicounts=3000000, imageFracReject=0.0,  rejectStart=3, maxcandidatestars=10000, restrictcompcolourcentre=-99.0, restrictcompcolourrange=-99.0, filterCode=None, restrictmagbrightest=-99.0, restrictmagdimmest=99.0):
    """
    Finds stars useful for photometry in each photometry/data file

    Parameters
    ----------
    targets : list
            List of target tuples in the format (ra, dec, 0, 0). ra and dec must be in decimal
    indir : str
            Path to files
    filelist : str
            List of photometry files to try
    acceptDistance : float
            Furtherest distance in arcseconds for matches
    lowcounts : int
            look for comparisons brighter than this
    hicounts : int
            look for comparisons dimmer than this
    imageFracReject: float
            This is a value which will reject images based on number of stars detected
    starFracReject : float
            This ia a value which will reject images that reject this fraction of available stars after....
    rejectStart : int
            This many initial images (lots of stars are expected to be rejected in the early images)
    minCompStars : int
            This is the minimum number of comp stars required

    Returns
    -------
    used_file : str
            Path to newly created file containing all images which are usable for photometry
    """
    sys.stdout.write("🌟 Identify comparison stars for photometry calculations\n")

    # LOOK FOR REJECTING NON-WCS IMAGES
    # If the WCS matching has failed, this function will remove the image from the list
    
    #If a new usedimages.txt has been made, make sure that there is no photcoords in directory
    if os.path.exists(paths['parent'] / "photSkyCoord"):
        os.remove(paths['parent'] / "photSkyCoord")
    if os.path.exists(paths['parent'] / "photFileHolder"):
        os.remove(paths['parent'] / "photFileHolder")



    fileSizer=0
    logger.info("Finding image with most stars detected and reject ones with bad WCS")
    referenceFrame = None

    counter=0
    #print (photFileHolder)
    for file in fileList:
        photFile = photFileHolder[counter]
        
        # if (photFile.size < 50):
        #     logger.debug("REJECT")
        #     logger.debug(file)
        #     fileList.remove(file)
        # elif (( asarray(photFile[:,0]) > 360).sum() > 0) :
        #     logger.debug("REJECT")
        #     logger.debug(file)
        #     fileList.remove(file)
        # elif (( asarray(photFile[:,1]) > 90).sum() > 0) :
        #     logger.debug("REJECT")
        #     logger.debug(file)
        #     fileList.remove(file)
        # else:
            # Sort through and find the largest file and use that as the reference file
        if photFile.size > fileSizer:
            #if (( photFile[:,0] > 360).sum() == 0) and ( photFile[0][0] != 'null') and ( photFile[0][0] != 0.0) :
            referenceFrame = photFile
            fileSizer = photFile.size
            fileRaDec = photCoords[counter]
            logger.debug("{} - {}".format(photFile.size, file))
        counter=counter+1

    if not referenceFrame.size:
        raise AstrosourceException("No suitable reference files found")

    logger.debug("Setting up reference Frame")
    #fileRaDec = SkyCoord(ra=referenceFrame[:,0]*u.degree, dec=referenceFrame[:,1]*u.degree)


    ##### Iterate hicounts and locounts until a reasonable sample < 250 stars are found in the middle range.
    

    logger.debug("Removing stars with low or high counts")
    rejectStars=[]
    # Check star has adequate counts
    logger.debug("Number of stars prior")
    logger.debug(referenceFrame.shape[0])
    logger.debug("Initial count range, Low: " +str(lowcounts)+ " High: "+ str(hicounts))
    
    #for j in range(referenceFrame.shape[0]):
    #    if ( referenceFrame[j][4] < lowcounts or referenceFrame[j][4] > hicounts ):
    #        rejectStars.append(int(j))
            
    #print (rejectStars)
    #print (referenceFrame[:,4])
    #print (where((referenceFrame[:,4] < lowcounts) | (referenceFrame[:,4] > hicounts))[0])
    #sys.exit()
    #print (referenceFrame)
    rejectStars=where((referenceFrame[:,4] < lowcounts) | (referenceFrame[:,4] > hicounts))[0]
    referenceFrame=delete(referenceFrame, rejectStars, axis=0)
    logger.debug("Number of stars after first cut")
    logger.debug(referenceFrame.shape[0])
        
    
    settled=0
    while settled==0:
        rejectStars=[]
        if referenceFrame.shape[0] > maxcandidatestars:
            
            #for j in range(referenceFrame.shape[0]):
            #    if ( referenceFrame[j][4] < lowcounts or referenceFrame[j][4] > hicounts ):
            #        rejectStars.append(int(j))
            rejectStars=where((referenceFrame[:,4] < lowcounts) | (referenceFrame[:,4] > hicounts))[0]
        else:
            settled=1
                 
    
        logger.debug("Number of stars after attempting to reduce number of sample comparison stars")
        logger.debug(delete(referenceFrame, rejectStars, axis=0).shape[0])
        
        if delete(referenceFrame, rejectStars, axis=0).shape[0] < maxcandidatestars:
            settled=1
        else:
            lowcounts=lowcounts+(0.05*lowcounts)
            hicounts=hicounts-(0.05*hicounts)
        

    logger.debug("Number of stars post")
    referenceFrame=delete(referenceFrame, rejectStars, axis=0)
    logger.debug("Final count range, Low: " +str(int(lowcounts))+ " High: "+ str(int(hicounts)))

    #Prepping files.
    #photSkyCoord=[]
    #photFileHolder=[]
    # for file in fileList: 
    #     photFile = load(paths['parent'] / file)
    #     photFileHolder.append(photFile)
    #     photSkyCoord.append(SkyCoord(ra=photFile[:,0]*u.degree, dec=photFile[:,1]*u.degree))

    originalReferenceFrame=referenceFrame
    originalfileList=fileList
    originalphotSkyCoord=photCoords
    originalphotFileHolder=photFileHolder
    originalRejectStart=rejectStart
    compchecker=0

    mincompstars=int(referenceFrame.shape[0]*mincompstars) # Transform mincompstars variable from fraction of stars into number of stars.
    if mincompstars < 1: # Always try to get at least ten comp candidates initially -- just because having a bunch is better than having 1.
        mincompstars=1
    if mincompstars > 100: # Certainly 100 is a maximum number of necessary candidate comparison stars.
        mincompstars=100
    if mincompstarstotal != -99:
        mincompstars=mincompstarstotal
    initialMincompstars=mincompstars
    
    rfCounter=0
    ##### Looper function to automatically cycle through more restrictive values for imageFracReject and starreject
    while (compchecker < mincompstars): # Keep going until you get the minimum number of Comp Stars
        imgsize=imageFracReject * fileSizer # set threshold size
        rejStartCounter = 0
        #usedImages=[] # Set up used images array
        imgReject = 0 # Number of images rejected due to high rejection rate
        loFileReject = 0 # Number of images rejected due to too few stars in the photometry file
        wcsFileReject=0
        
        
            
            
        q=0
        photReject=[]

        for Nholder in range(len(photFileHolder)):
            #print (q)
            #print (photSkyCoord[q])
            
            
            if ( not referenceFrame.shape[0] < mincompstars):
                rejStartCounter = rejStartCounter +1
                #photFile = load(paths['parent'] / file)
                photFile = photFileHolder[q]
                logger.debug('Image Number: ' + str(rejStartCounter))
                logger.debug(file)
                logger.debug("Image threshold size: "+str(imgsize))
                logger.debug("Image catalogue size: "+str(photFile.size))
                imgRejFlag=0
                if photFile.size > imgsize and photFile.size > 7 :
                    phottmparr = asarray(photFile)
                    photRAandDec = photCoords[q]
                    
                    # Checking existance of stars in all photometry files
                    rejectStars=[] # A list to hold what stars are to be rejected
                    
                    if (( phottmparr[:,0] > 360).sum() == 0) and ( phottmparr[0][0] != 'null') and ( phottmparr[0][0] != 0.0) :

                        

                        # # Find whether star in reference list is in this phot file, if not, reject star.
                        # for j in range(referenceFrame.shape[0]):
                            
                        #     testStar = SkyCoord(ra = referenceFrame[j][0]*u.degree, dec = referenceFrame[j][1]*u.degree)
                        #     # This is the function in the whole package which requires scipy
                        #     idx, d2d, d3d = testStar.match_to_catalog_sky(photRAandDec)
                        #     if (d2d.arcsecond > acceptDistance):
                        #         #"No Match! Nothing within range."
                        #         rejectStars.append(int(j))
                        #         #print (j)
                        #         #print (idx)
                                
                        # Faster way than loop
                        testStars=SkyCoord(ra = referenceFrame[:,0]*u.degree, dec = referenceFrame[:,1]*u.degree)
                        idx, d2d, _ = testStars.match_to_catalog_sky(photRAandDec)
                        rejectStars=where(d2d.arcsecond > acceptDistance)[0]
                        #print (idx)
                        #print (d2d)
                        #print (rejectStars)

                    
                    else:
                        logger.debug('**********************')
                        logger.debug('Image Rejected due to problematic entries in Photometry File')
                        logger.debug('**********************')
                        imgReject=imgReject+1
                        #fileList.remove(file)
                        photReject.append(q)
                        imgRejFlag=1

                    # if the rejectstar list is not empty, remove the stars from the reference List
                    if rejectStars != []:

                        if not (((len(rejectStars) / referenceFrame.shape[0]) > starreject) and rejStartCounter > rejectStart):
                            #print (len(rejectStars))
                            
                            #print (rejectStars)
                            #print (referenceFrame.shape())
                            referenceFrame = delete(referenceFrame, rejectStars, axis=0)
                            logger.debug('**********************')
                            logger.debug('Stars Removed  : ' +str(len(rejectStars)))
                            logger.debug('Remaining Stars: ' +str(referenceFrame.shape[0]))
                            logger.debug('**********************')
                            #usedImages.append(file)
                        else:
                            logger.debug('**********************')
                            logger.debug('Image Rejected due to too high a fraction of rejected stars')
                            logger.debug(len(rejectStars) / referenceFrame.shape[0])
                            logger.debug('**********************')
                            imgReject=imgReject+1
                            photReject.append(q)
                            #fileList.remove(file)
                    elif imgRejFlag==0:
                        logger.debug('**********************')
                        logger.debug('All Stars Present')
                        logger.debug('**********************')
                        #usedImages.append(file)

                    # If we have removed all stars, we have failed!
                    if (referenceFrame.shape[0]==0):
                        logger.error("Problem file - {}".format(file))
                        logger.error("Running Loop again")

                elif photFile.size < 7:
                    logger.error('**********************')
                    logger.error("WCS Coordinates broken")
                    logger.error('**********************')
                    wcsFileReject=wcsFileReject+1
                    photReject.append(q)
                    #fileList.remove(file)
                else:
                    logger.error('**********************')
                    logger.error("CONTAINS TOO FEW STARS")
                    logger.error('**********************')
                    loFileReject=loFileReject+1
                    photReject.append(q)
                    #fileList.remove(file)
                sys.stdout.write('.')
                sys.stdout.flush()
            q=q+1
        
        # Remove files and Hold the photSkyCoords in memory
        photCoords=delete(photCoords, photReject, axis=0)
        photFileHolder=delete(photFileHolder, photReject, axis=0)
        fileList=delete(fileList, photReject, axis=0)
        
        # Raise values of imgreject and starreject for next attempt
        starreject=starreject-0.025
        imageFracReject=imageFracReject+0.05
        compchecker = referenceFrame.shape[0]
        if starreject < 0.15:
            starreject=0.15
        if imageFracReject > 0.8:
            imageFracReject = 0.8

        if starreject == 0.15 and imageFracReject == 0.8 and mincompstars ==1 and rejectStart==1 and len(originalphotFileHolder) <5:
            logger.error("Failed to find any comparison candidates with the maximum restrictions. There is something terribly wrong!")
            raise AstrosourceException("Unable to find sufficient comparison stars with the most stringent conditions in this dataset.")

        if starreject == 0.15 and imageFracReject == 0.8 and mincompstars ==1 and rejectStart==1:
            logger.error("Number of Candidate Comparison Stars found this cycle: " + str(compchecker))
            
            logger.error("Kicking out the reference image and trying a random reference image.")
            #originalphotFileHolder=delete(originalphotFileHolder, 0, axis=0)
            #originalphotSkyCoord=delete(originalphotSkyCoord, 0, axis=0)
            #fileList=delete(fileList, 0, axis=0)
            starreject=0.3
            imageFracReject=0.05
            mincompstars=initialMincompstars
            
            logger.debug("Setting up reference Frame")
            
            referenceFrame=originalphotFileHolder[rfCounter]
            originalReferenceFrame=referenceFrame
            rfCounter=rfCounter+1
            
            #print (referenceFrame)
           
            ##### Iterate hicounts and locounts until a reasonable sample < 250 stars are found in the middle range.
            

            logger.debug("Removing stars with low or high counts")
            rejectStars=[]
            # Check star has adequate counts
            logger.debug("Number of stars prior")
            logger.debug(referenceFrame.shape[0])
            logger.debug("Initial count range, Low: " +str(lowcounts)+ " High: "+ str(hicounts))
            
            rejectStars=where((referenceFrame[:,4] < lowcounts) | (referenceFrame[:,4] > hicounts))[0]
            referenceFrame=delete(referenceFrame, rejectStars, axis=0)
            logger.debug("Number of stars after first cut")
            logger.debug(referenceFrame.shape[0])
            settled=0
            while settled==0:
                rejectStars=[]
                if referenceFrame.shape[0] > maxcandidatestars:
                    rejectStars=where((referenceFrame[:,4] < lowcounts) | (referenceFrame[:,4] > hicounts))[0]
                else:
                    settled=1
                                   
                logger.debug("Number of stars after attempting to reduce number of sample comparison stars")
                logger.debug(delete(referenceFrame, rejectStars, axis=0).shape[0])
                
                if delete(referenceFrame, rejectStars, axis=0).shape[0] < maxcandidatestars:
                    settled=1
                else:
                    lowcounts=lowcounts+(0.05*lowcounts)
                    hicounts=hicounts-(0.05*hicounts)
                
            logger.debug("Number of stars post")
            referenceFrame=delete(referenceFrame, rejectStars, axis=0)
            logger.debug("Final count range, Low: " +str(int(lowcounts))+ " High: "+ str(int(hicounts)))
            
            
            rejectStart=originalRejectStart
            fileList=originalfileList
            photCoords=originalphotSkyCoord
            photFileHolder=originalphotFileHolder       
                        
        elif starreject == 0.15 and imageFracReject == 0.8 and mincompstars ==1 and rejectStart > 1:
            logger.error("Number of Candidate Comparison Stars found this cycle: " + str(compchecker))
            logger.error("Trying again while rejecting less initial images.")
            rejectStart=rejectStart-1
            starreject=0.3
            imageFracReject=0.05
            mincompstars=initialMincompstars
            referenceFrame=originalReferenceFrame
            fileList=originalfileList
            photCoords=originalphotSkyCoord
            photFileHolder=originalphotFileHolder


        if starreject == 0.15 and imageFracReject == 0.8 and mincompstars !=1:
            logger.error("Maximum number of Candidate Comparison Stars found this cycle: " + str(compchecker))
            logger.error("Failed to find sufficient comparison candidates with the maximum restrictions, trying with a lower value for mincompstars")
            compchecker=0
            mincompstars=int(mincompstars*0.8)
            if mincompstars < 1:
                mincompstars =1
            starreject=0.3
            imageFracReject=0.05
            referenceFrame=originalReferenceFrame
            fileList=originalfileList
            photCoords=originalphotSkyCoord
            photFileHolder=originalphotFileHolder

        elif (compchecker < mincompstars):
            logger.error("Number of Candidate Comparison Stars found this cycle: " + str(compchecker))
            logger.error("Failed to find sufficient comparison candidates, adjusting starreject and imgreject and trying again.")
            logger.error("Now trying starreject " +str(starreject) + " and imgreject " +str(imageFracReject))
            referenceFrame=originalReferenceFrame

    #print (len(photCoords))
    #print (len(photFileHolder))
    #print (len(fileList))

    # Remove files and Hold the photSkyCoords in memory
    #photCoords=delete(photCoords, photReject, axis=0)
    #photFileHolder=delete(photFileHolder, photReject, axis=0)
    

    
    # Construct the output file containing candidate comparison stars
    outputComps=[]
    for j in range (referenceFrame.shape[0]):
        outputComps.append([referenceFrame[j][0],referenceFrame[j][1]])

    logger.debug("These are the identified common stars of sufficient brightness that are in every image")
    logger.debug(outputComps)

    logger.info('Images Rejected due to high star rejection rate: {}'.format(imgReject))
    logger.info('Images Rejected due to low file size: {}'.format(loFileReject))
    logger.info('Out of this many original images: {}'.format(len(originalfileList)))

    logger.info("Number of candidate Comparison Stars Detected: " + str(len(outputComps)))
    logger.info('Output sent to screenedComps.csv ready for use in Comparison')

    screened_file = paths['parent'] / "screenedComps.csv"
    outputComps = asarray(outputComps)

    # Reject targetstars immediately

    # Remove targets from consideration
    if targets is not None:
        if targets.shape == (4,):
            targets = [targets]
    
        while True:
            targetRejects=[]
            if outputComps.shape[0] ==2 and outputComps.size ==2:
                fileRaDec=SkyCoord(ra=outputComps[0]*u.degree,dec=outputComps[1]*u.degree)
            else:
                fileRaDec=SkyCoord(ra=outputComps[:,0]*u.degree,dec=outputComps[:,1]*u.degree)
    
            for target in targets:
                varCoord = SkyCoord(target[0],(target[1]), frame='icrs', unit=u.deg) # Need to remove target stars from consideration
    
                idx, d2d, _ = varCoord.match_to_catalog_sky(fileRaDec)
                if d2d.arcsecond < 5.0: # anything within 5 arcseconds of the target
                    targetRejects.append(idx)
    
            if targetRejects==[]:
                break
            #Remove target and restore skycoord list
            outputComps=delete(outputComps, targetRejects, axis=0)
            logger.info(outputComps)
            if len(outputComps) == 0:
                logger.info("The only comparisons detected where also target stars. No adequate comparisons were found.")
                sys.exit()
            fileRaDec = SkyCoord(ra=outputComps[:,0]*u.degree, dec=outputComps[:,1]*u.degree)

    
    # If a new usedimages.txt has been made, make sure that there is no photcoords in directory
    #if os.path.exists(paths['parent'] / "photSkyCoord"):
    #    os.remove(paths['parent'] / "photSkyCoord")

    sys.stdout.write('\n')
    
    # print (len(photFileHolder))
    # print (len(photCoords))
    # print (len(usedImages))
    # sys.exit()

        
    file1=open(paths['parent'] / "photSkyCoord","wb")
    pickle.dump(photCoords, file1)
    file1.close
    
    file1=open(paths['parent'] / "photFileHolder","wb")
    pickle.dump(photFileHolder, file1)
    file1.close
    
    # Remove candidate comparisons that are out of the restricted range of colours or magnitudes
    
    #print (restrictcompcolourcentre)
    #print (restrictcompcolourrange)
    
    
    # Get Average RA and Dec from file
    #print (outputComps)
    #print (outputComps.shape[0])
    #print (outputComps.size)
    #print (restrictmagbrightest)
    #print (restrictmagdimmest)
    if restrictmagbrightest != -99.0 or restrictmagdimmest !=99.0 or restrictcompcolourcentre != -99.0 or restrictcompcolourrange != -99.0:
    
        if outputComps.shape[0] == 1 and outputComps.size == 2:
            #logger.debug(compFile[0])
            #logger.debug(compFile[1])
            avgCoord=SkyCoord(ra=(outputComps[0])*degree, dec=(outputComps[1]*degree))
    
        else:
            #logger.debug(average(compFile[:,0]))
            #logger.debug(average(compFile[:,1]))
    
    
            #remarkably dumb way of averaging around zero RA and just normal if not
            resbelow= any(ele >350.0 and ele<360.0 for ele in outputComps[:,0].tolist())
            resabove= any(ele >0.0 and ele<10.0 for ele in outputComps[:,0].tolist())
            if resbelow and resabove:
                avgRAFile=[]
                for q in range(len(outputComps[:,0])):
                    if outputComps[q,0] > 350:
                        avgRAFile.append(outputComps[q,0]-360)
                    else:
                        avgRAFile.append(outputComps[q,0])
                avgRA=average(avgRAFile)
                if avgRA <0:
                    avgRA=avgRA+360
                avgCoord=SkyCoord(ra=(avgRA*degree), dec=((average(outputComps[:,1])*degree)))
            else:
                avgCoord=SkyCoord(ra=(average(outputComps[:,0])*degree), dec=((average(outputComps[:,1])*degree)))
    
            logger.debug(f"Average: RA {avgCoord.ra}, Dec {avgCoord.dec}")
        
        #print (avgCoord)
        
        #print (max(outputComps[:,0]))
        #print (min(outputComps[:,0]))
        #print (max(outputComps[:,1]))
        #print (min(outputComps[:,1]))
        
        radius= 0.5 * pow(  pow(max(outputComps[:,0])-min(outputComps[:,0]),2) + pow(max((outputComps[:,1])-min(outputComps[:,1]))*cos((min(outputComps[:,1])+max(outputComps[:,1]))/2),2) , 0.5)
        #print (1.5*radius)
        
    
        
        FILTERS = {
                    'B' : {'APASS' : {'filter' : 'Bmag', 'error' : 'e_Bmag', 'colmatch' : 'Vmag', 'colerr' : 'e_Vmag', 'colname' : 'B-V', 'colrev' : '0'}},
                    'V' : {'APASS' : {'filter' : 'Vmag', 'error' : 'e_Vmag', 'colmatch' : 'Bmag', 'colerr' : 'e_Bmag', 'colname' : 'B-V', 'colrev' : '1'}},
                    'CV' : {'APASS' : {'filter' : 'Vmag', 'error' : 'e_Vmag', 'colmatch' : 'Bmag', 'colerr' : 'e_Bmag', 'colname' : 'B-V', 'colrev' : '1'}},
                    'up' : {'SDSS' : {'filter' : 'umag', 'error' : 'e_umag', 'colmatch' : 'gmag', 'colerr' : 'e_gmag', 'colname' : 'u-g', 'colrev' : '0'},
                            'SkyMapper' : {'filter' : 'uPSF', 'error' : 'e_uPSF', 'colmatch' : 'gPSF', 'colerr' : 'e_gPSF', 'colname' : 'u-g', 'colrev' : '0'}},
                    'gp' : {'SDSS' : {'filter' : 'gmag', 'error' : 'e_gmag', 'colmatch' : 'rmag', 'colerr' : 'e_rmag', 'colname' : 'g-r', 'colrev' : '0'},
                            'SkyMapper' : {'filter' : 'gPSF', 'error' : 'e_gPSF', 'colmatch' : 'rPSF', 'colerr' : 'e_rPSF', 'colname' : 'g-r', 'colrev' : '0'},
                            'PanSTARRS': {'filter' : 'gmag', 'error' : 'e_gmag', 'colmatch' : 'rmag', 'colerr' : 'e_rmag', 'colname' : 'g-r', 'colrev' : '0'},
                            'APASS' : {'filter' : 'g_mag', 'error' : 'e_g_mag', 'colmatch' : 'r_mag', 'colerr' : 'e_r_mag', 'colname' : 'g-r', 'colrev' : '0'}},
                    'rp' : {'SDSS' : {'filter' : 'rmag', 'error' : 'e_rmag', 'colmatch' : 'imag', 'colerr' : 'e_imag', 'colname' : 'r-i', 'colrev' : '0'},
                            'SkyMapper' : {'filter' : 'rPSF', 'error' : 'e_rPSF', 'colmatch' : 'iPSF', 'colerr' : 'e_iPSF', 'colname' : 'r-i', 'colrev' : '0'},
                            'PanSTARRS': {'filter' : 'rmag', 'error' : 'e_rmag', 'colmatch' : 'imag', 'colerr' : 'e_imag', 'colname' : 'r-i', 'colrev' : '0'},
                            'APASS' : {'filter' : 'r_mag', 'error' : 'e_r_mag', 'colmatch' : 'i_mag', 'colerr' : 'e_i_mag', 'colname' : 'r-i', 'colrev' : '0'}},
                    'ip' : {'SDSS' : {'filter' : 'imag', 'error' : 'e_imag', 'colmatch' : 'rmag', 'colerr' : 'e_rmag', 'colname' : 'r-i', 'colrev' : '1'},
                            'SkyMapper' : {'filter' : 'iPSF', 'error' : 'e_iPSF', 'colmatch' : 'rPSF', 'colerr' : 'e_rPSF', 'colname' : 'r-i', 'colrev' : '1'},
                            'PanSTARRS': {'filter' : 'imag', 'error' : 'e_imag', 'colmatch' : 'rmag', 'colerr' : 'e_rmag', 'colname' : 'r-i', 'colrev' : '1'},
                            'APASS' : {'filter' : 'i_mag', 'error' : 'e_i_mag', 'colmatch' : 'r_mag', 'colerr' : 'e_r_mag', 'colname' : 'r-i', 'colrev' : '1'}},
                    'zs' : {'PanSTARRS': {'filter' : 'zmag', 'error' : 'e_zmag', 'colmatch' : 'rmag', 'colerr' : 'e_rmag', 'colname' : 'r-zs', 'colrev' : '1'},
                            'SkyMapper' : {'filter' : 'zPSF', 'error' : 'e_zPSF', 'colmatch' : 'rPSF', 'colerr' : 'e_rPSF', 'colname' : 'r-zs', 'colrev' : '1'},
                            'SDSS' : {'filter' : 'zmag', 'error' : 'e_zmag', 'colmatch' : 'rmag', 'colerr' : 'e_rmag', 'colname' : 'r-zs', 'colrev' : '1'}},
                    }
        
        try:
            catalogues = FILTERS[filterCode]
        except IndexError:
            raise AstrosourceException(f"{filterCode} is not accepted at present")
        
        coords=[]
        for cat_name, opt in catalogues.items():
            try:
                if coords ==[]: #SALERT - Do not search if a suitable catalogue has already been found
                    logger.info("Searching " + str(cat_name))
                    if cat_name == 'PanSTARRS' and nopanstarrs==True:
                        logger.info("Skipping PanSTARRS")
                    elif cat_name == 'SDSS' and nosdss==True:
                        logger.info("Skipping SDSS")
                    else:
    
                        coords = catalogue_call(avgCoord, 1.5*radius, opt, cat_name, targets=targets, closerejectd=closerejectd)
                        # If no results try next catalogue
                        #print (len(coords.ra))
                        if len(coords.ra) == 0:
                            coords=[]
                            raise AstrosourceException("Empty catalogue produced from catalogue call")
            except AstrosourceException as e:
                logger.debug(e)
        
        if coords.cat_name == 'PanSTARRS' or coords.cat_name == 'APASS':
            max_sep=2.5 * arcsecond
        else:
            max_sep=1.5 * arcsecond
        
        #print (outputComps)
        #print (coords.ra)
        #print (coords.dec)
        
        catCoords=SkyCoord(ra=coords.ra*degree, dec=coords.dec*degree)
        
        #Get calib mags for least variable IDENTIFIED stars.... not the actual stars in compUsed!! Brighter, less variable stars may be too bright for calibration!
        #So the stars that will be used to calibrate the frames to get the OTHER stars.
        calibStands=[]
    
        if outputComps.shape[0] ==1 and outputComps.size ==2:
            lenloop=1
        else:
            lenloop=len(outputComps[:,0])
    
        for q in range(lenloop):
            if outputComps.shape[0] ==1 and outputComps.size ==2:
                compCoord=SkyCoord(ra=outputComps[0]*degree, dec=outputComps[1]*degree)
            else:
                compCoord=SkyCoord(ra=outputComps[q][0]*degree, dec=outputComps[q][1]*degree)
            idxcomp,d2dcomp,_=compCoord.match_to_catalog_sky(catCoords)
            if d2dcomp < max_sep:
                if not isnan(coords.mag[idxcomp]) and not isnan(coords.emag[idxcomp]):
                    if outputComps.shape[0] ==13 and outputComps.size ==13:
                        calibStands.append([outputComps[0],outputComps[1],0,coords.mag[idxcomp],coords.emag[idxcomp],0,coords.colmatch[idxcomp],coords.colerr[idxcomp],0])
                    else:
                        calibStands.append([outputComps[q][0],outputComps[q][1],0,coords.mag[idxcomp],coords.emag[idxcomp],0,coords.colmatch[idxcomp],coords.colerr[idxcomp],0])
    
    
    
        #print (len(asarray(calibStands)[:,0]))
        
    
        ### remove stars that that brighter (--restrictmagbrighter) or dimmer (--restrictmagdimmer) than requested or colour from calib standards.
        calibStandsReject=[]
        if (asarray(calibStands).shape[0] != 9 and asarray(calibStands).size !=9) and calibStands != []:
            for q in range(len(asarray(calibStands)[:,0])):
    
                if (calibStands[q][3] > restrictmagdimmest) or (calibStands[q][3] < restrictmagbrightest):
                    calibStandsReject.append(q)
                    #logger.info(calibStands[q][3])
                    
                if opt['colrev'] ==0:
                    
                    if (calibStands[q][3]-calibStands[q][6] > (restrictcompcolourcentre + restrictcompcolourrange)) or (calibStands[q][3]-calibStands[q][6] < (restrictcompcolourcentre - restrictcompcolourrange)) :
                        calibStandsReject.append(q)
                else:
                    #print (calibStands[q][6]-calibStands[q][3])
                    if (calibStands[q][6]-calibStands[q][3] > (restrictcompcolourcentre + restrictcompcolourrange)) or (calibStands[q][6]-calibStands[q][3] < (restrictcompcolourcentre - restrictcompcolourrange)) :
                        calibStandsReject.append(q)
       
            if len(calibStandsReject) != len(asarray(calibStands)[:,0]):
                calibStands=delete(calibStands, calibStandsReject, axis=0)
    
        #print (restrictcompcolourcentre)
        #print (restrictcompcolourrange)
        
        
        # NOW only keep those stars in outputComps that match calibStands
        
        
        #outputComps=hstack(array(([[calibStands[:,0]],[calibStands[:,1]]])))
        
        #racol=array([calibStands[:,0]])
        #deccol=array([calibStands[:,1]])
        outputComps=column_stack((calibStands[:,0],calibStands[:,1]))
        print (asarray(outputComps))        
        
        #np.hstack(np.array([[coords.ra],[coords.dec],[coords.mag],[coords.emag],[coords.colmatch],[coords.colerr]]))
        
        
        logger.info('Removed ' + str(len(calibStandsReject)) + ' Candidate Comparison Stars for being too bright or too dim or the wrong colour')
    
        #sys.exit()
    
        ### If looking for colour, remove those without matching colour information
    
        #calibStandsReject=[]
    
        # if (asarray(calibStands).shape[0] != 1 and asarray(calibStands).size !=2) and calibStands != []:
        #     for q in range(len(asarray(calibStands)[:,0])):
        #         reject=0
        #         if colourdetect == True:
        #             if np.isnan(calibStands[q][6]): # if no matching colour
        #                 reject=1
        #             elif calibStands[q][6] == 0:
        #                 reject=1
        #             elif np.isnan(calibStands[q][7]):
        #                 reject=1
        #             elif calibStands[q][7] == 0:
        #                 reject=1
        #         if np.isnan(calibStands[q][3]): # If no magnitude info
        #             reject=1
        #         elif calibStands[q][3] == 0:
        #             reject=1
        #         elif np.isnan(calibStands[q][4]):
        #             reject=1
        #         elif calibStands[q][4] == 0:
        #             reject=1
    
        #         if reject==1:
        #             calibStandsReject.append(q)
    
        #     if len(calibStandsReject) != len(asarray(calibStands)[:,0]):
        #         calibStands=delete(calibStands, calibStandsReject, axis=0)
        
        #coords = catalogue_call(avgCoord, 1.5*radius, opt, cat_name, targets=targets, closerejectd=closerejectd)
        
        
        #sys.exit()
    
    savetxt(screened_file, outputComps, delimiter=",", fmt='%0.8f')
    used_file = paths['parent'] / "usedImages.txt"
    with open(used_file, "w") as f:
        for s in fileList:
            filename = Path(s).name
            f.write(str(filename) +"\n")


    return fileList, outputComps, photFileHolder, photCoords
