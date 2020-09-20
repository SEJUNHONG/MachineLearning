import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

plt.close("all")

iris=datasets.load_iris()

X=iris["data"][50:150, (2, 3)]
Y=iris["target"][50:150]

scaler=StandardScaler()
scaler.fit(X)
X_std=scaler.transform(X)

[X_train, X_test, Y_train, Y_test]=train_test_split(X_std, Y, test_size=0.4, random_state=10, shuffle=True)
f1=plt.figure(1)
ax1=f1.add_subplot(111)

df_clf=pd.DataFrame(np.c_[X_std, Y])
df_clf.columns=["petalLength", "petalWidth", "target"]
df_clf_group=df_clf.groupby("target")

for target, group in df_clf_group:
    ax1.plot(group.petalLength, group.petalWidth, '.', label="target")
    
svm_clf=SVC(C=0.01, kernel="linear")
svm_clf.fit(X_train, Y_train)

delta=0.01
[xx0_min, xx0_max]=[min(X_std[:,0])-10*delta, max(X_std[:,0])+10*delta]
[xx1_min, xx1_max]=[min(X_std[:,1])-10*delta, max(X_std[:,0])+10*delta]
[xx0, xx1]=np.meshgrid(np.arange(xx0_min, xx0_max, delta), np.arange(xx1_min, xx1_max, delta))

h=svm_clf.decision_function(np.c_[xx0.ravel(), xx1.ravel()])
h=h.reshape(xx0.shape)
ax1.clabel(ax1.contour(xx0, xx1, h))

print(svm_clf.score(X_test, Y_test))

#svm_clf.decision_function(X)
#print(svm_clf.decision_function([[0.5, 0.5]]))
#print(svm_clf.predict([[1, 1]]))