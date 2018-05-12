from skimage.morphology import disk, square
from utils import readImages, populateFiles, saveImageFile, applyGaussian, applyMedian, applyMean

# Let's apply median, gaussian and mean
# all of the morphed images

sourceDir = 'step1'
destDir = 'step2'

# get morphed images
files = populateFiles(sourceDir)

# read each file and populate a dictionary
images = readImages(sourceDir, files)

# iterate over dictionary and apply filters
for name, image in images.items():
    medianImage = applyMedian(image, disk(1))
    saveImageFile(medianImage, destDir + '/' + name + '_median.bmp')
    meanImage = applyMean(image, disk(1))
    saveImageFile(meanImage, destDir + '/' + name + '_mean.bmp')
    gaussianImage = applyGaussian(image)
    saveImageFile(gaussianImage, destDir + '/' + name + '_gaussian.bmp')
