from sklearn.datasets import load_iris
iris=load_iris()

from sklearn.datasets import load_breast_cancer
bCancer=load_breast_cancer()

from sklearn.model_selection import train_test_split
X=iris.data
y=iris.target
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.4, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred=knn.predict(X_test)
scores=metrics.accuracy_score(y_test, y_pred)