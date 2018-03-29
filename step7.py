import numpy as np
from utils import openImageFile, saveImageFile, extractMinutiae, markMinutiae, applyMaskToMinutiaeList, getOrientationsOfMinutiae, putDictionaryToFile
import matplotlib.pyplot as plt

image = np.asarray(np.loadtxt('step6/skel.txt'), dtype=int)
mask = np.asarray(np.loadtxt('step6/mask.txt'), dtype=int)

minutiaeList = extractMinutiae(image)
markedImage = markMinutiae(image, minutiaeList)

saveImageFile(markedImage, 'step7/all_minutiae.bmp')

newMask, newMinutiaeList = applyMaskToMinutiaeList(mask, minutiaeList)
newMarkedImage = markMinutiae(image, newMinutiaeList)

saveImageFile(newMask, 'step7/new_mask.bmp')
saveImageFile(newMarkedImage, 'step7/masked_minutiae.bmp')

orientedMinutiae = getOrientationsOfMinutiae(newMinutiaeList, image)

putDictionaryToFile(orientedMinutiae, 'step7/minutiae.txt')
