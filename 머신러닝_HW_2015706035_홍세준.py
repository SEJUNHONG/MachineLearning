import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

plt.close("all")

#[X, Y]=datasets.make_moons(n_samples=200, shuffle=True, noise=0.2, random_state=15)
[X, Y]=datasets.make_circles(n_samples=200, shuffle=True, noise=0.2, random_state=15, factor=0.99)

scaler=StandardScaler()
scaler.fit(X)
X_std=scaler.transform(X)

[X_train, X_test, Y_train, Y_test]=train_test_split(X_std, Y, test_size=0.4, random_state=10, shuffle=True)

df_clf=pd.DataFrame(np.c_[X, Y])
df_clf.columns=["x0", "x1", "target"]
df_group=df_clf.groupby("target")

f1=plt.figure(1)
ax1=f1.add_subplot(111)
#ax1=f1.add_subplot(111, projection='3d')

for target, group in df_group:
    ax1.plot(group.x0, group.x1, '.', label="target")
    
#svm_clf=SVC(C=0.1, kernel="poly", degree=2, gamma=2, coef0=10)
svm_clf=SVC(C=0.1, kernel="rbf", gamma=1)
svm_clf.fit(X_train, Y_train)
#svm_clf.decision_function(X)

delta=0.01
[xx0_min, xx0_max]=[min(X_std[:,0])-10*delta, max(X_std[:,0])+10*delta]
[xx1_min, xx1_max]=[min(X_std[:,1])-10*delta, max(X_std[:,0])+10*delta]
[xx0, xx1]=np.meshgrid(np.arange(xx0_min, xx0_max, delta), np.arange(xx1_min, xx1_max, delta))

h=svm_clf.decision_function(np.c_[xx0.ravel(), xx1.ravel()])
h=h.reshape(xx0.shape)
ax1.clabel(ax1.contour(xx0, xx1, h ))

print(svm_clf.score(X_test, Y_test))