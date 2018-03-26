import numpy as np
from utils import openImageFile, saveImageFile, extractMinutiae, markMinutiae, applyMaskToMinutiaeList
import matplotlib.pyplot as plt

image = openImageFile('step6/skel.bmp')
mask = openImageFile('step5/mask.bmp')

minutiaeList = extractMinutiae(image)
markedImage = markMinutiae(image, minutiaeList)

saveImageFile(markedImage, 'step7/all_minutiae.bmp')

newMask, newMinutiaeList = applyMaskToMinutiaeList(mask, minutiaeList)
newMarkedImage = markMinutiae(image, newMinutiaeList)

saveImageFile(newMask, 'step7/new_mask.bmp')
saveImageFile(newMarkedImage, 'step7/masked_minutiae.bmp')
