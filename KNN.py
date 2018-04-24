# Fit a classification model using K Nearest Neighbour(KNN) Algorithm on a given data set. [One may use data sets like Caravan, Smarket, Weekly, Auto and Boston]

# Code: -
from sklearn import neighbors, datasets
import pandas as pd

breast_cancer = datasets.load_breast_cancer()

X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
Y = pd.DataFrame(breast_cancer.target)

kNN = neighbors.KNeighborsClassifier()

kNN.fit(X, Y.values.ravel())

print("Accuracy:- ", kNN.score(X, Y))
