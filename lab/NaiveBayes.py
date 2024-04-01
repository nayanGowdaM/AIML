import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class NaiveBayes:
    def fit( self, X,y):
        self.X= X
        self.y=y
        self.n_samples , self.n_features = X.shape
        self.classes = np.unique( y)
        self.n_classes = len(self.classes)

        self.mean = np.zeros( (self.n_classes, self.n_features) , dtype= np.float64)
        self.var = np.zeros( (self.n_classes, self.n_features) , dtype= np.float64)
        self.priors = np.zeros( self.n_classes, dtype = np.float64)

        for idx, c in enumerate( self.classes):
            x_c = X[y==c]
            self.mean[idx: ] =  x_c.mean( axis=0)
            self.var[idx: ] = x_c.var( axis = 0)
            self.priors[idx] = float( x_c.shape[0]) / float( self.n_samples)


    def predict(self, X):
        y_pred = [ self._predict( sample) for sample in X]
        return np.array( y_pred)
    
    def _predict( self, sample):
        posterior = []

        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx])
            class_conditional = np.sum( np.log( self._pdf( idx, sample)))
            posterior.append(prior + class_conditional)

        return self.classes[np.argmax( posterior)]
    
    def _pdf( self, class_idx, sample):
        mean = self.mean[class_idx]
        var = self.var[class_idx]

        numerator = np.exp( -(sample-mean)**2/(2*var))
        denominator = np.sqrt( 2 * np.pi * var)
        return numerator/denominator
    
iris = load_iris()
X , y = iris.data, iris.target
class_names = iris.target_names
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state = 42)

nbc = NaiveBayes()
nbc.fit(X_train, y_train)
y_pred = nbc.predict(X_test)
print("Predictions: " , class_names[y_pred])
print("Accuracty Scores: " , np.mean(y_pred==y_test))
