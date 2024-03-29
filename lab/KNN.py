import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from collections import Counter

iris =  load_iris()
X,y = iris.data, iris.target
class_names = iris.target_names
X_train, X_test,y_train, y_test= train_test_split( X, y, test_size=0.3, random_state = 1 )

class KNN:
    def __init__(self, k):
        self.k= k

    def fit(self, X, y):
        self.X_train  = X
        self.y_train = y
    
    def predict(self, x_test):
        y_pred = [ self._predict( x) for x in x_test]
        return np.array(y_pred)
    
    def _predict( self, x):
        distances = [np.linalg.norm( x-x_train) for x_train in self.X_train]
        k_indices = np.argsort( distances)[:self.k]
        maxi = Counter(k_indices).most_common(1)
        return self.y_train[maxi[0][0]]
    
knn = KNN(3)

knn.fit( X_train, y_train)
y_pred = knn.predict( X_test)
print("Accuracy Score : " , np.average(y_pred == y_test))
print( class_names[y_pred])