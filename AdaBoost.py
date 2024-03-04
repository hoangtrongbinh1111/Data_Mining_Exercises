from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import time

def AdaBoost():
    """
    This function creates an AdaBoost classifier using the sklearn library.

    Parameters:
    -----------
    n_estimators: int
        The number of weak learners in the ensemble.

    base_estimator: object
        The base estimator used for each round of boosting.

    random_state: int
        The random seed used for initialization.

    learning_rate: float
        The shrinkage parameter.

    Returns:
    --------
    adaboost: object
        The trained AdaBoost classifier.

    """
    # Record the starting time
    start_time = time.time()

    # create new classifier with data fake
    X, y = make_classification(n_samples=10000, n_features=30, random_state=42)

    # split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # create model AdaBoost Classifier
    adaboost = AdaBoostClassifier(n_estimators=100, base_estimator=DecisionTreeClassifier(max_depth=3), random_state=42, learning_rate=0.5)

    # start training
    adaboost.fit(X_train, y_train)

    # predict with the trained model
    y_pred = adaboost.predict(X_test)

    # metrics to evaluate the model
    report = classification_report(y_test, y_pred)
    print("Report with AdaBoost: " + report)

    # Record the ending time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time

    print("Time taken:", elapsed_time, "seconds")