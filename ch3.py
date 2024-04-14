#Exercise 9
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, cauchy, bernoulli

N = 10000

X = norm.rvs(size = N)
Y = cauchy.rvs(size = N)

nn = np.arange(1, N+1)

plt.figure(figsize=(12,8))

ax = plt.subplot(2,1,1)
ax.plot(nn, np.cumsum(X)/nn)
ax.set_title('N(0,1)')
ax.set_xlabel('n')
ax.set_ylabel(r'$\overline{X}_n$')

ax = plt.subplot(2,1,2)
ax.plot(nn, np.cumsum(Y)/nn)
ax.set_title('Cauchy')
ax.set_xlabel('n')
ax.set_ylabel(r'$\overline{Y}_n$')

plt.tight_layout()
plt.show()

#Exercise 11

B=20
N=10000

Y = 2*bernoulli.rvs(p = 1/2, size = (B,N)) - 1
X = np.cumsum(Y, axis = 1)

nn = np.arange(1, N+1)
plt.figure(figsize=(12,8))
z = norm.ppf(0.975)
plt.plot(nn, z*np.sqrt(nn), color='red')
plt.plot(nn, -z*np.sqrt(nn), color='red')
plt.fill_between(nn, z*np.sqrt(nn), -z*np.sqrt(nn), color='red', alpha=0.5)

for b in range(B):
    plt.plot(nn, X[b])
    
plt.show()
