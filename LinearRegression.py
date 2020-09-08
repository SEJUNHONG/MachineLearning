import numpy as np
import pandas as pd
import matplotlib.pylab as plt

dfLoad=pd.read_csv('https://sites.google.com/site/vlsicir/testData_LinearRegression.txt', sep="\s+");
xxRaw=dfLoad["xx"]
yyRaw=dfLoad["yy"]
plt.plot(xxRaw, yyRaw, "r. ")

N=len(xxRaw)
X=np.c_[np.ones([N, 1]), xxRaw]
y=np.array(yyRaw).reshape(N, 1)
wOLS=np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

xSample=np.arange(0, 2, 0.01)
xSamplePadding=np.c_[np.ones([200, 1]), xSample]
yPredict=xSamplePadding.dot(wOLS)
plt.plot(xSample, yPredict, "b.-")

N_iteration=10
wGD=np.zeros([2,1])
eta=0.1

for iteration in range(N_iteration):
    print(wGD)
    gradients=-(2/N)*(X.T.dot(y-X.dot(wGD)))
    wGD=wGD-eta*gradients