# Implement multiple regression model on a standard data set and plot the least square regression fit. Comment on the result. [One may use inbuilt data sets like Carseats, Boston etc].

# Code: -
from sklearn import linear_model, datasets
import pandas as pd

boston = datasets.load_boston()

X = pd.DataFrame(boston.data, columns=boston.feature_names)
Y = pd.DataFrame(boston.target, columns=["PRICE"])

mRegression = linear_model.LinearRegression()

mRegression.fit(X, Y)
prediction = mRegression.predict(X)
print(prediction[:50])
print("Accuracy = ", mRegression.score(X, Y))
