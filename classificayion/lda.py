from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
import matplotlib.pyplot as plt

wine = datasets.load_wine()
print(wine.DESCR)
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target
print(df)

X = df['alcohol'].values.reshape(-1, 1)

Y = df['target'].values.reshape(-1, 1)

lda = LinearDiscriminantAnalysis()
lda.fit(X, Y.ravel())
prediction = lda.predict(X)
score = lda.score(X, Y)
print("Accuracy:- ", score)
