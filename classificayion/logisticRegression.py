from sklearn import linear_model, datasets
import pandas as pd

breast_cancer = datasets.load_breast_cancer()

print(breast_cancer.DESCR)
X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
Y = pd.DataFrame(breast_cancer.target)
print(X.head())
logisticRegression = linear_model.LogisticRegression(C=100)

logisticRegression.fit(X, Y.values.ravel())
print()
print("Accuracy:- ", logisticRegression.score(X, Y))
