# For a given data set, do the following:
# (ii) Fit a regression tree
# [One may choose data sets like Car seats, Boston etc for the
#  purpose].

from sklearn import tree, datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
Y = iris.target

classification = tree.DecisionTreeRegressor()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

classification.fit(X_train, y_train)
print(classification.score(X_train, y_train))

prediction = classification.predict(X_test)
print(classification.score(X_test, y_test))
print(accuracy_score(y_test, prediction))
