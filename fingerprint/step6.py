import numpy as np
from utils import openImageFile, saveImageFile, applyOpening, applyClosing, applyGaussian, applyMedian, applySegmentation, applyOrientation, applyFrequency, applyGaborFilter, applySkeletonization, applyThinning
from skimage.morphology import disk
from skimage.util import invert

image = openImageFile('fingerprint.bmp')

image = applyClosing(image, disk(1))
image = applyOpening(image, disk(1))
image = applyMedian(image, disk(1))
image = applyGaussian(image)

# Assigned block size, which is divisive of 288 and 384
normImage, mask = applySegmentation(image, 12, 0.1)
np.savetxt('step6/mask.txt', mask)
orientImage = applyOrientation(normImage, 7, 1, 42, 7)
freqImage, meanFreq = applyFrequency(normImage, mask, orientImage, 24, 5, 5, 15)
gaborKernel, gaboredImage = applyGaborFilter(normImage, orientImage, meanFreq * mask, 0.75, 0.75)
thinnedImage = applyThinning(gaboredImage)
np.savetxt('step6/skel.txt', thinnedImage)
saveImageFile(thinnedImage, 'step6/skel.bmp')
