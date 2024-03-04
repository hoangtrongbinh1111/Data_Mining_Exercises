from sklearn.cluster import DBSCAN
from sklearn.datasets import make_classification
from sklearn.metrics import adjusted_rand_score
import time

def DBSCAN():
    """
    This function creates a DBSCAN classifier using the sklearn library.

    Parameters:
    -----------
    eps: float, optional (default=0.5)
        The maximum distance between two samples for them to be considered as in the same neighborhood.

    min_samples: int, optional (default=5)
        The minimum number of samples in a neighborhood for a point to be considered as a core point.

    metric: string or callable, optional (default='euclidean')
        The metric to use when calculating distance between instances in a feature array. If metric is a string, it must be one of the options allowed by scikit-learn for its metrics. If metric is a callable, it is used to compute the distance between instances.

    algorithm: {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional (default='auto')
        The algorithm to use for the tree.

    Returns:
    --------
    dbscan: sklearn.cluster.DBSCAN
        The trained DBSCAN classifier.

    res: float
        The adjusted rand score of the clustering.

    Time: float
        The time taken to run the algorithm.

    """
    # Record the starting time
    start_time = time.time()

    # create new classifier with data fake
    X, y = make_classification(n_samples=10000, n_features=30, random_state=42)

    # create DBSCAN classifier
    dbscan = DBSCAN(eps=0.4, min_samples=5, metric='euclidean', algorithm='ball_tree')

    # start training
    dbscan.fit(X)

    # metrics
    res = adjusted_rand_score(y, dbscan.labels_)
    print("DBSCAN: "+ res)

    # Record the ending time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print("Time taken:", elapsed_time, "seconds")