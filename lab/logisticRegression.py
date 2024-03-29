import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/(1+np.exp(-z))

def cost( h,y):
    return ( -y * np.log(h) -(1-y)*np.log(1-h))

def gradient(h,y, X):
    return np.dot( X.T, (h-y)) / y.shape[0]

def logisticRegression( X, y, learningRate = 0.1, numOfIteration=200):
    weights = np.zeros( X.shape[1])
    for _ in range( numOfIteration):
        z = np.dot( X, weights)
        h = sigmoid( z)
        gradient_val = gradient( h, y, X)

        weights-=learningRate*gradient_val

    return weights


iris = load_iris()
X  , y = iris.data[:, :2], (iris.target==0)*1
X_train, X_test, y_train, y_test = train_test_split( X,y, test_size=0.4, random_state=9)


sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

weights = logisticRegression(X_train_std, y_train)

y_pred = sigmoid(np.dot( X_test_std, weights))>0.5

print("Accuracy Score ", np.average( y_pred == y_test))


## plotting 
x_min, x_max = X_train_std[:,0].min()-1 , X_train_std[:,0].max()+1
y_min, y_max = X_train_std[:,1].min()-1, X_train_std[:, 1].max()+1
xx,yy = np.meshgrid(np.arange(x_min,x_max, 0.1), np.arange(y_min, y_max,0.1))

z = sigmoid(np.dot(np.c_[xx.ravel(), yy.ravel()], weights))>0.5
z = z.reshape(xx.shape)
plt.contourf(xx,yy,z,alpha = 0.4)
plt.scatter(X_train_std[:,0],X_train_std[: , 1],c=y_train, alpha=0.8)
plt.title("Logistic Regression Decision Boundry")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.savefig("plt.png")
plt.show()