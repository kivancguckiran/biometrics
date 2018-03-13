from skimage.morphology import disk, square
from utils import openImageFile, saveImageFile, saveHistogram, applyErosion, applyDilation, applyOpening, applyClosing

# This is the first step for analysis

image = openImageFile('fingerprint.bmp')

# Let's see our histogram
saveHistogram(image, 'initial_histogram.jpg')

# Let's see erosion and dilation
erosionApplied = applyErosion(image, disk(1))
saveImageFile(erosionApplied, 'step1/erosion.bmp')

dilationApplied = applyDilation(image, disk(1))
saveImageFile(dilationApplied, 'step1/dilation.bmp')

# Let's see opening and closing
openingApplied = applyOpening(image, disk(1))
saveImageFile(openingApplied, 'step1/opening.bmp')

closingApplied = applyClosing(image, disk(1))
saveImageFile(closingApplied, 'step1/closing.bmp')

