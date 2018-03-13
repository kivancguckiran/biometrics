from os import listdir
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import median, gaussian, threshold_otsu, threshold_local, rank
from skimage.morphology import opening, closing, erosion, dilation, thin, skeletonize, medial_axis
from skimage.exposure import equalize_hist
from skimage.util import invert


def applyMean(image, selem):
    return rank.mean(image, selem)

def applyMedian(image, selem):
    return median(image, selem)

def applyGaussian(image):
    return gaussian(image)

def applyOpening(image, selem):
    return opening(image, selem)

def applyClosing(image, selem):
    return closing(image, selem)

def applyErosion(image, selem):
    return erosion(image, selem)

def applyDilation(image, selem):
    return dilation(image, selem)

def applyCustomThreshold(image, thr):
    binarized = np.copy(image)
    binarized[binarized > thr] = 255
    binarized[binarized <= thr ] = 0
    return binarized

def applyOtsuThreshold(image):
    otsu = threshold_otsu(image)
    return (applyCustomThreshold(image, otsu), otsu)

def applyAdaptiveThreshold(image, blockSize, offset):
    local = threshold_local(image, blockSize, offset=offset)
    binarized = np.copy(image)
    binarized[binarized > local] = 255
    binarized[binarized <= local ] = 0
    return binarized

def applyThinning(image):
    inverted = invert(image / 255)
    return np.asarray(invert(thin(inverted)), dtype=int)

def applySkeletonization(image):
    inverted = invert(image / 255)
    return np.asarray(invert(skeletonize(inverted)), dtype=int)

def applyMedialAxis(image):
    inverted = invert(image / 255)
    return np.asarray(invert(medial_axis(inverted)), dtype=int)

def applyEqualization(image):
    return equalize_hist(image)

def openImageFile(filename):
    return np.array(rgb2gray(misc.imread(filename)))

def saveImageFile(image, filename):
    misc.imsave(filename, image)

def saveHistogram(image, filename):
    plt.hist(image.ravel(), 256)
    plt.savefig(filename, bbox_inches='tight')

def populateFiles(dir):
    return [file.split('.')[0] for file in listdir(dir)]

def readImages(dir, files):
    images = dict()
    for file in files:
        images[file] = openImageFile(dir + '/' + file + '.bmp')
    return images
