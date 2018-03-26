import math
import cv2
from os import listdir
from scipy import misc, ndimage, signal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.draw import circle_perimeter, set_color
from skimage.color import rgb2gray, gray2rgb
from skimage.filters import median, gaussian, threshold_otsu, threshold_local, rank
from skimage.morphology import opening, closing, erosion, dilation, thin, skeletonize, medial_axis, binary_erosion, binary_dilation, square
from skimage.exposure import equalize_hist
from skimage.util import invert

MINUTIAE_RIDGE_NONE = 0
MINUTIAE_RIDGE_BIFURCATION = 1
MINUTIAE_RIDGE_ENDING = 2

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
    return applyCustomThreshold(image, otsu)

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

def applyNormalization(image):
    return (image - np.mean(image)) / np.std(image)

def applySegmentation(image, blockSize, threshold):
    rowCount, colCount = image.shape

    normImage = applyNormalization(image)
    stdDevIm = np.zeros((rowCount, colCount))

    for i in range(0, rowCount, blockSize):
        for j in range(0, colCount, blockSize):
            block = normImage[i:i + blockSize][:, j:j + blockSize]
            stdDevIm[i:i + blockSize][:, j:j + blockSize] = np.std(block) * np.ones(block.shape)

    stdDevIm = stdDevIm[0:rowCount][:, 0:colCount]

    mask = stdDevIm > threshold
    normImage = (image - np.mean(image[mask])) / np.std(image[mask])

    return (normImage, mask)

def applyOrientation(image, gradientSize, gradientSigma, blockSize, blockSigma):
    gauss = cv2.getGaussianKernel(gradientSize, gradientSigma)
    f = gauss * gauss.T

    fy, fx = np.gradient(f)

    Gx = signal.convolve2d(image, fx, mode='same')    
    Gy = signal.convolve2d(image, fy, mode='same')

    Gxx = np.power(Gx, 2)
    Gyy = np.power(Gy, 2)
    Gxy = Gx * Gy

    gauss = cv2.getGaussianKernel(blockSize, blockSigma)
    f = gauss * gauss.T
    
    Gxx = ndimage.convolve(Gxx, f)
    Gyy = ndimage.convolve(Gyy, f)
    Gxy = 2 * ndimage.convolve(Gxy, f)

    denom = np.sqrt(np.power(Gxy , 2) + np.power((Gxx - Gyy), 2)) + np.finfo(float).eps
    
    sin2theta = Gxy / denom
    cos2theta = (Gxx - Gyy) / denom
    
    orientim = np.pi/2 + np.arctan2(sin2theta, cos2theta) / 2

    return orientim

def getFrequency(image, orientImage, windSize, minWaveLength, maxWaveLength):
    rows,cols = np.shape(image)

    cosorient = np.mean(np.cos(2 * orientImage))
    sinorient = np.mean(np.sin(2 * orientImage))  
    orient = math.atan2(sinorient , cosorient) / 2

    rotim = ndimage.rotate(image, orient / np.pi * 180 + 90, axes=(1,0), reshape = False, order = 3, mode = 'nearest')

    cropSize = int(np.fix(rows / np.sqrt(2)))
    offset = int(np.fix((rows - cropSize) / 2))
    rotim = rotim[offset:offset + cropSize][:, offset:offset + cropSize]

    proj = np.sum(rotim, axis = 0)
    dilation = ndimage.grey_dilation(proj, windSize, structure=np.ones(windSize))

    temp = np.abs(dilation - proj)
    
    peak_thresh = 2   
    
    maxpts = (temp < peak_thresh) & (proj > np.mean(proj))
    maxind = np.where(maxpts)
    
    rows_maxind, cols_maxind = np.shape(maxind)
    
    if(cols_maxind < 2):
        freqim = np.zeros(image.shape)
    else:
        peaks = cols_maxind
        waveLength = (maxind[0][cols_maxind-1] - maxind[0][0])/(peaks - 1)
        if waveLength>=minWaveLength and waveLength<=maxWaveLength:
            freqim = 1/np.double(waveLength) * np.ones(image.shape)
        else:
            freqim = np.zeros(image.shape)
        
    return freqim

def applyFrequency(image, mask, orient, blockSize, windSize, minWaveLength, maxWaveLength):
    rows,cols = image.shape
    freq = np.zeros((rows,cols))
    
    for r in range(0, rows - blockSize, blockSize):
        for c in range(0, cols - blockSize, blockSize):
            blkim = image[r:r + blockSize][:, c:c + blockSize]
            blkor = orient[r:r + blockSize][:, c:c + blockSize]

            freq[r:r + blockSize][:, c:c + blockSize] = getFrequency(blkim, blkor, windSize, minWaveLength, maxWaveLength)
    
    freq = freq * mask
    freq_1d = np.reshape(freq, (1, rows * cols))
    ind = np.where(freq_1d > 0)
    
    ind = np.array(ind)
    ind = ind[1, :]
    
    non_zero_elems_in_freq = freq_1d[0][ind] 
    
    meanFreq = np.mean(non_zero_elems_in_freq)

    return (freq, meanFreq)

def applyGaborFilter(image, orientImage, frequency, kx, ky):
    angleInc = 3
    image = np.double(image)
    rows,cols = image.shape
    newim = np.zeros((rows,cols))
    
    freq_1d = np.reshape(frequency, (1, rows * cols))
    ind = np.where(freq_1d > 0)
    
    ind = np.array(ind)
    ind = ind[1, :]
    
    non_zero_elems_in_freq = freq_1d[0][ind] 
    non_zero_elems_in_freq = np.double(np.round((non_zero_elems_in_freq * 100))) / 100
    
    unfreq = np.unique(non_zero_elems_in_freq)
    
    sigmax = 1 / unfreq[0] * kx
    sigmay = 1 / unfreq[0] * ky
    
    sze = np.round(3 * np.max([sigmax,sigmay]))

    x, y = np.meshgrid(np.linspace(-sze, sze, (2 * sze + 1)), np.linspace(-sze, sze, ( 2 * sze + 1)))
    
    reffilter = np.exp(-(((np.power(x, 2))/(sigmax * sigmax) + (np.power(y, 2)) / (sigmay * sigmay)))) * np.cos(2 * np.pi * unfreq[0] * x)

    filt_rows, filt_cols = reffilter.shape
    
    radialAngle = int(round(180 / angleInc))
    gabor_filter = np.array(np.zeros((radialAngle, filt_rows, filt_cols)))
    
    for o in range(0, radialAngle):        
        rot_filt = ndimage.rotate(reffilter, -(o * angleInc + 90), reshape = False)
        gabor_filter[o] = rot_filt
    
    maxsze = int(sze)   

    temp = frequency > 0    
    validr,validc = np.where(temp)    
    
    temp1 = validr > maxsze
    temp2 = validr < rows - maxsze
    temp3 = validc > maxsze
    temp4 = validc < cols - maxsze
    
    final_temp = temp1 & temp2 & temp3 & temp4    
    
    finalind = np.where(final_temp) 
    
    maxorientindex = np.round(180 / angleInc)
    orientindex = np.round(orientImage / np.pi * 180 / angleInc)
    
    for i in range(0,rows):
        for j in range(0,cols):
            if(orientindex[i][j] < 1):
                orientindex[i][j] = orientindex[i][j] + maxorientindex
            if(orientindex[i][j] > maxorientindex):
                orientindex[i][j] = orientindex[i][j] - maxorientindex

    finalind_rows, finalind_cols = np.shape(finalind)
    sze = int(sze)

    for k in range(0, finalind_cols):
        r = validr[finalind[0][k]]
        c = validc[finalind[0][k]]
        
        img_block = image[r-sze:r+sze + 1][:, c-sze:c+sze + 1]
        
        newim[r][c] = np.sum(img_block * gabor_filter[int(orientindex[r][c]) - 1])

    return (reffilter, np.asarray(newim < -3, dtype=int))

def checkMinutiae(pixels, i, j):
    cells = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    values = [pixels[i + k][j + l] for k, l in cells]

    crossings = 0
    for k in range(0, 8):
        crossings += abs(values[k] - values[k + 1])
    crossings /= 2

    result = MINUTIAE_RIDGE_NONE

    if pixels[i][j] == 1:
        if crossings == 1:
            result = MINUTIAE_RIDGE_ENDING
        if crossings == 3:
            result = MINUTIAE_RIDGE_BIFURCATION

    return result

def extractMinutiae(img):
    inverted = np.asarray(np.invert(img) / 255, dtype=int)
    x, y = inverted.shape
    minutiaeList = list()

    for i in range(1, x - 1):
        for j in range(1, y - 1):
            minutiae = checkMinutiae(inverted, i, j)
            if minutiae != MINUTIAE_RIDGE_NONE:
                minutiaeList.append({'type': minutiae, 'coordinates': [i, j]})

    return minutiaeList

def markMinutiae(img, minutiaeList):
    image = gray2rgb(img)
    radius = 4
    colors = {MINUTIAE_RIDGE_ENDING : [255, 0, 0], MINUTIAE_RIDGE_BIFURCATION : [0, 255, 0]}

    for minutiae in minutiaeList:
        i, j = minutiae['coordinates']
        rr, cc = circle_perimeter(i, j, radius)
        set_color(image, (rr, cc), colors[minutiae['type']])

    return image

def applyMaskToMinutiaeList(msk, minutiaeList):
    mask = np.asarray(msk / 255, dtype=int)
    paddedMask = np.asarray(binary_erosion(mask, square(35)), dtype=int)
    newMinutiaeList = list()

    for minutiae in minutiaeList:
        i, j = minutiae['coordinates']
        if paddedMask[i, j] == 1:
            newMinutiaeList.append(minutiae)

    return paddedMask, newMinutiaeList

