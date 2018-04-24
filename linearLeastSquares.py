from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

boston = datasets.load_boston()

DF_array = pd.DataFrame(boston.data, columns=boston.feature_names)

print(boston.DESCR)

print(DF_array.head())

DF_array['PRICE'] = boston.target

print(DF_array.head())

Y = (DF_array['PRICE']).values.reshape(-1, 1)
X = (DF_array['RM']).values.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

regression = linear_model.LinearRegression()

model = regression.fit(X_train, y_train)

prediction = regression.predict(X_test)
print(model.score(X_test, y_test))
