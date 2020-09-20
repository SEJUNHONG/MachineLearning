import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import scipy as sp

import scipy.stats

plt.close("all")

dfLoad=pd.read_csv('https://sites.google.com/site/vlsicir/ClassificationSample2.txt', sep='\s+')
samples=np.array(dfLoad)
x=samples[:,0]
y=samples[:,1]

N=len(x)
numK=2

#Initialize categorial distribution
pi=np.ones([1, numK])*(1/numK)

mx=np.mean(x)
my=np.mean(y)
sx=np.std(x)
sy=np.std(y)



u0=np.array([mx-sx, my+sy]) # cluster 0 mean
u1=np.array([mx+sx, my-sy]) # cluster 1 mean

f1=plt.figure(1)
ax1=f1.add_subplot(111)
ax1.plot(x, y, "b.")
ax1.set_aspect("equal")
ax1.plot(u0[0], u0[1], "r*")
ax1.plot(u1[0], u1[1], "g*")

sig0=np.array([[sx*sx/4, 0], [0, sy*sy/4]])
sig1=np.array([[sx*sx/4, 0], [0, sy*sy/4]])

R=np.ones([N, numK])*(1/numK)

j=0
while True:
    j=j+1
    pi=[(1/N)*(np.sum(R[:,0])), (1/N)*(np.sum(R[:,1]))]
   
    N0=sp.stats.multivariate_normal.pdf(samples, u0, sig0)
    N1=sp.stats.multivariate_normal.pdf(samples, u1, sig1)
    
    Rold=np.copy(R)
    R=np.array([pi[0]*N0/(pi[0]*N0+pi[1]*N1), pi[1]*N1/(pi[0]*N0+pi[1]*N1)]).T
    
    if(np.linalg.norm(R-Rold)<0.001*N*numK):
        break 
    
    weightedSum=samples.T.dot(R)    # 2 dimensional of data * 2 cluster# matrix

    u0=weightedSum[:,0]/sum(R[:,0])
    u1=weightedSum[:,1]/sum(R[:,1])
    sig0=samples.T.dot(np.multiply(R[:,0].reshape(N,1),samples))/sum(R[:,0]) - u0.reshape(2, 1)*u0.reshape(2,1).T
    sig1=samples.T.dot(np.multiply(R[:,1].reshape(N,1),samples))/sum(R[:,1]) - u1.reshape(2, 1)*u1.reshape(2,1).T
clusterCol=np.round(1-R)[:,0]
dfCluster=pd.DataFrame(np.c_[samples, clusterCol])
dfCluster.columns=["x", "y", "K"]
dfGroup=dfCluster.groupby("K")

f2=plt.figure(2)
ax2=f2.add_subplot(111)

for cluster, dataGroup in dfGroup:
    ax2.plot(dataGroup.x, dataGroup.y, "*", label=cluster)
    ax2.set_aspect("equal")
    ax2.plot(u0[0], u0[1], "r*")
    ax2.plot(u1[0], u1[1], "g*")
# M step

ax2.plot(x,y,"b.")
