from os import listdir
from skimage.morphology import disk, square
from utils import populateFiles, readImages, saveImageFile, applySkeletonization, applyThinning, applyMedialAxis

# Let's apply median, gaussian and mean
# all of the morphed images

sourceDir = 'step3'
destDir = 'step4'

# get morphed images
files = populateFiles(sourceDir)

# read each file and populate a dictionary
images = readImages(sourceDir, files)

# iterate over dictionary and apply thresholds
for name, image in images.items():
    skelImage = applySkeletonization(image)
    saveImageFile(skelImage, destDir + '/' + name + '_skel.bmp')
    thinImage = applyThinning(image)
    saveImageFile(thinImage, destDir + '/' + name + '_thin.bmp')
    medialImage = applyMedialAxis(image)
    saveImageFile(medialImage, destDir + '/' + name + '_medial.bmp')
