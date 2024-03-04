from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import time

def RandomForest(): 
    """
    This function creates a random forest classifier using the sklearn library.

    Parameters:
    -----------
    n_estimators: int, optional (default=10)
        The number of trees in the forest.

    random_state: int, optional (default=None)
        Controls the randomness of the estimator.

    criterion: string, optional (default="gini")
        The function to measure the quality of a split. Supported criteria are "gini" for the Gini impurity and "entropy" for the information gain.

    max_depth: int or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.

    min_samples_split: int, float, optional (default=2)
        The minimum number of samples required to split an internal node:

    min_samples_leaf: int, float, optional (default=1)
        The minimum number of samples required at each leaf node.

    Returns:
    --------
    random_forest: sklearn.ensemble.RandomForestClassifier
        The trained random forest classifier.
    """

    # Record the starting time
    start_time = time.time()

    # create new classifier with data fake
    X, y = make_classification(n_samples=10000, n_features=30, random_state=42)

    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # create random forest classifier with some number of trees and other hyperparameters
    random_forest = RandomForestClassifier(n_estimators=300, random_state=42, criterion='entropy', max_depth=10, min_samples_split=5, min_samples_leaf=1)

    # start training
    random_forest.fit(X_train, y_train)

    # predict with the trained model
    y_pred = random_forest.predict(X_test)

    # metrics to evaluate the model
    report = classification_report(y_test, y_pred)
    print("Report with Random Forest: " + report)

    # Record the ending time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print("Time taken:", elapsed_time, "seconds")