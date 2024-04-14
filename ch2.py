import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#Exercise 13

x = np.random.normal(0,1,10000)
y = np.exp(x)
print(y)

xx = np.arange(0.01, 5, step=0.01)
f_Y =  norm.pdf(np.log(xx))/xx

plt.figure(figsize=(12,10))
plt.hist(y, bins=200, label ='histogram', density=True, histtype = 'step')
plt.xlim(0,5)
plt.plot(xx, f_Y)
plt.show()
