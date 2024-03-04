from sklearn.cluster import DBSCAN
from sklearn.datasets import make_classification
from sklearn.metrics import adjusted_rand_score

# create new classifier with data fake
X, y = make_classification(n_samples=10000, n_features=30, random_state=42)

# create DBSCAN classifier
dbscan = DBSCAN(eps=0.4, min_samples=5, metric='euclidean', algorithm='ball_tree')

# start training
dbscan.fit(X)

# metrics
res = adjusted_rand_score(y, dbscan.labels_)
print(res)