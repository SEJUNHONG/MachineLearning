import pandas as pd
import numpy as np
import matplotlib.pylab as plt

plt.close("all")

dfLoad=pd.read_csv('https://sites.google.com/site/vlsicir/ClassificationSample.txt', sep='\s+')
samples=np.array(dfLoad)
x=samples[:,0]
y=samples[:,1]

N=len(samples)
numK=2

f1=plt.figure(1)
ax1=f1.add_subplot(111)
ax1.plot(x, y, "b.")
ax1.set_aspect("equal")

mx=np.mean(x)
my=np.mean(y)

sx=np.std(x)
sy=np.std(y)

z0=np.array([mx-2*sx, my-2*sy]).reshape(1, 2)
z1=np.array([mx+2*sx, my+2*sy]).reshape(1, 2)
z=np.r_[z0, z1]
ax1.plot(z[0,0], z[0,1], "r*")
ax1.plot(z[1,0], z[1,1], "r*")


cluster=np.round(np.random.rand(580)>0.5)
iteration=0
while True:
    iteration=iteration+1
    clusterOld=np.copy(cluster)
# M step
    for i in range(N):
        cluster[i]=np.linalg.norm(samples[i,:]-z [0,:])>np.linalg.norm(samples[i,:]-z[1,:])
    
    if(np.alltrue(clusterOld==cluster)):
        break
    
    dfCluster=pd.DataFrame(np.c_[x, y, cluster])
    dfCluster.columns=["x", "y", "cluster"] 
    dfGroup=dfCluster.groupby("cluster")

# E step
    for j in range(numK):
        z[j,:]=dfGroup.mean().iloc[j]
    
f2=plt.figure(2)
ax2=f2.add_subplot(111)
ax2.plot(z[:,0], z[:,1], "r*")
ax2.set_aspect("equal")
for clusterName, group in dfGroup:
    ax2.plot(group.x, group.y,".", label=clusterName)
    
ax2.legend()
