from scipy import misc
from skimage.color import rgb2gray
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from utils import openImageFile, applyGaussian, applyMedian, applyOpening, applyClosing, applyEqualization, applyAdaptiveEqualization, applyOtsuThreshold
from segment import segment

from os import listdir
from os.path import isfile, join


fig, ax = plt.subplots(1)

def segmentIris(path, segmentedPath):
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

    for file in onlyfiles:
        im = openImageFile(path + file)

        ciriris, cirpupil, imwithnoise = segment(im)

        iris = Circle((ciriris[1], ciriris[0]), ciriris[2], color='r', fill=False)
        pupil = Circle((cirpupil[1], cirpupil[0]), cirpupil[2], color='b', fill=False)

        ax.imshow(im, cmap='gray')

        ax.add_patch(iris)
        ax.add_patch(pupil)

        plt.savefig(segmentedPath + file)
        plt.cla()

segmentIris('casia/', 'segmentedCasia/')
segmentIris('iris/', 'segmentedIris/')
