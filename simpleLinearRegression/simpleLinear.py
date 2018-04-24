# Implement simple linear regression model on a standard data set and plot the least square regression fit. Comment on the result. [One may use inbuilt data sets like Boston, Auto etc]
# Code: -
from sklearn import datasets, linear_model
import pandas as pd
import matplotlib.pyplot as plt

boston = datasets.load_boston()

DF_array = pd.DataFrame(boston.data, columns=boston.feature_names)

#print(boston.DESCR)

print(DF_array.head())

DF_array['PRICE'] = boston.target

print(DF_array.head())

Y = (DF_array['PRICE']).values.reshape(-1, 1)
X = (DF_array['RM']).values.reshape(-1, 1)

regression = linear_model.LinearRegression()

regression.fit(X, Y)

print(regression.score(X, Y))

print("*", regression.predict(7))

plt.scatter(X, Y, color= 'black')
plt.xlabel("Average number of rooms per dwelling")
plt.ylabel("Price")
plt.title("Housing Prices Vs Number of rooms per dwelling")
plt.plot(X, regression.predict(X), color= 'blue')
plt.show()
