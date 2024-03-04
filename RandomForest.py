from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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
print(report)