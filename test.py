import numpy as np
from scipy.signal import find_peaks_cwt
from matplotlib import pyplot as plt

cb = np.array([1,2,3,4,5,6,7,8,9,10,9,8,7,6,5,4,3,2,1])
indexes = find_peaks_cwt(cb,height = 0)
print(indexes)
x = np.arange(1,20,1)
plt.plot(indexes,cb[indexes],'x')
plt.plot(x,cb)
plt.show()

