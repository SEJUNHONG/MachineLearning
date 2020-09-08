import numpy as np
import pandas as pd
import matplotlib.pylab as plt

dfLoad=pd.read_csv('https://sites.google.com/site/vlsicir/testData_workHour_vs_passFail.txt', sep="\s+");
xxRaw=np.array(dfLoad.values[:,0])
yyRaw=np.array(dfLoad.values[:,1])
#plt.plot(xxRaw, yyRaw, "k.")

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

#xxTest=np.linspace(-10, 10, num=101)
#plt.plot(xxTest, sigmoid(xxTest), "k-")
    
N=len(xxRaw)
x_bias=np.c_[np.ones([N, 1]), xxRaw]
y=yyRaw.reshape(N, 1)
X=x_bias.T

eta=0.1
N_iterations=1000
wGD=np.zeros([2, 1])
wGDbuffer=np.zeros([2, N_iterations+1])

for iteration in range(N_iterations):
    mu=sigmoid(wGD.T.dot(x_bias)).T
    gradients=X.T.dot(mu-y)
    wGD=wGD-eta*gradients
    wGDbuffer[:,iteration+1]=[wGD[0], wGD[1]]
    
xxTest=np.linspace(0, 10, num=N).reshape(N, 1)
xxTest_bias=np.c_[np.ones([N, 1]), xxTest]
aaa=sigmoid(wGD.T.dot(xxTest_bias.T))
plt.plot(xxTest, sigmoid(wGD.T.dot(xxTest_bias.T)).T, "r-.")