import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve


# 1 for geniune, 0 for impostor
scores = np.asarray([5, 3, 5, 4, 3, 2, 1, 2, 1])
labels = np.asarray([1, 1, 1, 1, 0, 0, 0, 0, 0])

fpr, tpr, thresholds = roc_curve(labels, scores)

plt.plot(fpr, tpr)

plt.savefig('roc.png')
plt.show()
