#exercise 4

import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt

alpha = 0.05
p = 0.4

N = 1000

nn = np.arange(1, N+1)
epsilon = np.sqrt((1/(2*nn))*np.log(2/alpha))

B = 5000

p_hat = np.empty((B, N))

for i in range(B):
    X = bernoulli.rvs(p, size = N, random_state = i)
    p_hat[i] = np.cumsum(X)/nn
    
conf = np.mean((p_hat + epsilon >=p) & (p_hat-epsilon <= p), axis=0)

#print(conf)
plt.figure(figsize=(12,6))
plt.plot(nn, conf)
plt.show()