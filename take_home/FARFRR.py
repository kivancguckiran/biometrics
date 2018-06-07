import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


geniunes = np.asarray([5, 3, 5, 4])
impostors = np.asarray([-1, 3, 2, 1, 2, 1])

minThreshold = 0
maxThreshold = 7

quantization = 1
thresholds = np.arange(minThreshold, maxThreshold, quantization)

FRR = []
FAR = []

for threshold in thresholds:
    FRR.append(np.sum(geniunes < threshold) / geniunes.size)
    FAR.append(np.sum(impostors >= threshold) / impostors.size)

red_patch = mpatches.Patch(color='r', label='FRR')
blue_patch = mpatches.Patch(color='b', label='FAR')

plt.legend(handles=[red_patch, blue_patch])

plt.plot(thresholds, FRR, color='r')
plt.plot(thresholds, FAR, color='b')

plt.savefig('farfrr.png')
plt.show()
