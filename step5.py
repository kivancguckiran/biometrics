import numpy as np
from utils import openImageFile, saveImageFile, applySegmentation, applyOrientation, applyFrequency, applyGaborFilter, applySkeletonization
from skimage.morphology import skeletonize
from skimage.util import invert

image = openImageFile('fingerprint.bmp')

# Assigned block size, which is divisive of 288 and 384
normImage, mask = applySegmentation(image, 12, 0.1)
saveImageFile(normImage, 'step5/norm.bmp')
saveImageFile(np.asarray(mask, dtype=int), 'step5/mask.bmp')

orientImage = applyOrientation(normImage, 7, 1, 42, 7)
saveImageFile(orientImage, 'step5/orient.bmp')

freqImage, meanFreq = applyFrequency(normImage, mask, orientImage, 24, 5, 5, 15)
saveImageFile(freqImage, 'step5/freq.bmp')

gaborKernel, gaboredImage = applyGaborFilter(normImage, orientImage, meanFreq * mask, 0.75, 0.75)
saveImageFile(gaborKernel, 'step5/gabor_kernel.bmp')
saveImageFile(gaboredImage, 'step5/gabor.bmp')

thinnedImage = np.asarray(invert(skeletonize(gaboredImage)), dtype=int)
saveImageFile(thinnedImage, 'step5/skel.bmp')
