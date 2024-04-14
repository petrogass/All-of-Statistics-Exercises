import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

from scipy.stats import norm, cauchy

#Exercise 3
def SamCDF(x,data):
	
	
	return sum(1 for y in data if y <= x)/len(data)
	
def error(size, alpha=0.05):
	return np.sqrt(np.log(2/alpha)/(2*size))
    
def ci( x, data, a=0.05):
        e = error(len(data), a)
        y = SamCDF(x, data)
        l = max(y - e, 0)
        u = min(y + e, 1)
        return (l, u)
	
	
def Exercise3(dist, iters = 100, size = 1000):
	correct=0
	for _ in range(iters):
		data = dist.rvs(size=size)
		
		epsilon = error(size, alpha = 0.05)
		
		
		c = sum(1 for x in data if max(SamCDF(x,data)-epsilon, 0)<=dist.cdf(x)<=min(SamCDF(x, data)+epsilon, 1)) == len(data)
		
		correct += c
	return correct/iters


'''confidence = Exercise3(norm)
print(confidence)
confidence = Exercise3(cauchy)
print(confidence)'''

def Exercise7():

    df = pd.read_csv('data/fijiquakes.csv', sep='\t')
    print(df.head())
    print(df.describe())
    data = df.mag
    F1 = SamCDF(4.9, data)
    F2 = SamCDF(4.3, data)
    print(F1-F2)
    df['ecdf'] = [SamCDF(x,data) for x in data]
    df['l'] = [ci(x, data)[0] for x in data]
    df['u'] = [ci(x, data)[1] for x in data]
    df = df.sort_values('mag')
    dtf = pd.melt(df, id_vars='mag', value_vars=['ecdf', 'l', 'u'])
    #g = sns.catplot(x='mag', y='value', data=dtf, hue='variable', 
    #           size=5, aspect=1.5)
    plt.scatter(df['mag'] , df['ecdf'], marker='.')
    plt.plot(df['mag'] , df['ecdf'])
    plt.scatter(df['mag'] , df['l'], marker='.')
    plt.plot(df['mag'] , df['l'])
    plt.scatter(df['mag'] , df['u'], marker='.')
    plt.plot(df['mag'] , df['u'])
    plt.show()
    print((ci(4.3, data)[0], ci(4.9, data)[1]))
#Exercise7()

def Exercise8():
    geysers = pd.read_csv('data/geysers.csv')
    
    wait_time = geysers['waiting']
    n = len(wait_time)
    mean = np.sum(wait_time)/n
    var = np.sum((wait_time - mean)**2)/n
    std = np.sqrt(var)
    ste = std/np.sqrt(n)
    print(mean, var, ste)
    z = 1.64 
    U = mean + z*ste
    L = mean - z*ste
    print(U, L)
    geysers['ecdf']= [SamCDF(x, geysers['waiting']) for x in geysers['waiting']]
    geysers = geysers.sort_values('waiting')
    print(SamCDF(76, geysers['waiting']))
     
    
#Exercise8()
def Exercise10():
    df = pd.read_csv('data/cloud_seeding.csv')
    seeded = df['Seeded_Clouds']
    unseeded = df['Unseeded_Clouds']
    
    n_s = len(seeded)
    n_u = len(unseeded)
    
    mu_s = np.mean(seeded)
    mu_u = np.mean(unseeded)
    
    mu_theta = mu_s - mu_u
    se_theta = np.sqrt(np.var(seeded)/n_s + np.var(unseeded)/n_u)
    
    print(mu_theta, se_theta)
    
    z = 1.96
    U = mu_theta + z*se_theta
    L = mu_theta - z*se_theta
    print(U, L)
    
Exercise10()

