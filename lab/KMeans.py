import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class KMeans:
    def __init__( self, k=3, iterations=100):
        self.k=k
        self.iterations=iterations
        self.clusters = [[] for _ in range(self.k)]
        self.centroids = []

    def fit(self, X):
        self.X = X
        self.n_samples ,self.n_features = X.shape

        random_sample_idx = np.random.choice(self.n_samples, self.k, replace= False)
        self.centroids = [ self.X[idx] for idx in random_sample_idx]

        for _ in range( self.iterations):
            self.clusters = self._create_cluster( self.centroids)

            old_centroids = self.centroids
            self.centroids = self._get_centroids( self.clusters)

            if self._is_converged(self.centroids, old_centroids):
                break

    
    def _create_cluster( self, centroids):
        cluster = [[] for _ in range(self.k)]
        for idx, sample in enumerate( self.X):
            centroid_idx = self._closest_centroid( sample, centroids)
            cluster[centroid_idx].append(idx)
        
        return cluster
    
    def _closest_centroid( self, sample, centroids):
        distances = np.zeros(self.k)
        for idx, centroid in enumerate(centroids):
            distances[idx] = np.linalg.norm( sample-centroid)
        closest_idx = np.argmin( distances)
        return closest_idx

    
    def _get_centroids( self, clusters):
        centroids=np.zeros(( self.k, self.n_features))
        for idx , cluster in enumerate(clusters):
            centroids[idx] = np.mean( self.X[cluster], axis=0)
        return centroids
    
    def _is_converged( self, old_centroids, new_centroids):
        distances = [np.linalg.norm( old_centroids-new_centroids)]
        return sum(distances) == 0
    
    def predict( self, X):
        lables = np.empty(self.n_samples)
        for idx , sample in enumerate(X):
            for cluster_idx , cluster in enumerate(self.clusters):
                if( idx in cluster):
                    lables[idx]=cluster_idx
        return lables

iris = load_iris()
X = iris.data
X_train, X_test = train_test_split(X, test_size=0.3, random_state=1)
class_names = iris.target_names

k = KMeans( 3, 200)
k.fit(X_train)
y_pred = k.predict(X_train)
y_pred = y_pred.astype(int)

print("Predictions: ", class_names[y_pred])

print("CLass Caribles: " , class_names)