from os import listdir
from skimage.morphology import disk, square
from utils import populateFiles, readImages, saveImageFile, applyCustomThreshold, applyOtsuThreshold, applyAdaptiveThreshold

# Let's apply median, gaussian and mean
# all of the morphed images

sourceDir = 'step2'
destDir = 'step3'

# get morphed images
files = populateFiles(sourceDir)

# read each file and populate a dictionary
images = readImages(sourceDir, files)

# iterate over dictionary and apply thresholds
for name, image in images.items():
    # let's derive a custom threshold from the histogram, say 75
    customImage = applyCustomThreshold(image, 75)
    saveImageFile(customImage, destDir + '/' + name + '_custom75.bmp')
    otsuImage, thr = applyOtsuThreshold(image)
    saveImageFile(otsuImage, destDir + '/' + name + '_otsu' + str(thr) + '.bmp')
    # blocksize 35 and offset 5 seems OK
    adaptiveImage = applyAdaptiveThreshold(image, 35, 3)
    saveImageFile(adaptiveImage, destDir + '/' + name + '_adaptive.bmp')
