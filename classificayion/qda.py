from sklearn import datasets
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pandas as pd
import matplotlib.pyplot as plt

wine = datasets.load_wine()
print(wine.DESCR)
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target
print(df.head())

X = df['alcohol'].values.reshape(-1, 1)

Y = df['target'].values.reshape(-1, 1)

#print(X[:20])
#print(Y[:20])

qda = QuadraticDiscriminantAnalysis()
qda.fit(X, Y.ravel())
prediction = qda.predict(X)
score = qda.score(X, Y)
print("Accuracy:- ", score)
