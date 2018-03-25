import numpy as np
from utils import openImageFile, saveImageFile, extracMinutiaeToFile

image = openImageFile('step6/skel.bmp')

extracMinutiaeToFile(image, 'step7/minutiae.png')
