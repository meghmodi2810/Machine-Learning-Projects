import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data  = pd.read_csv('allCSV/headbrain.csv')
# print(data.shape)
# print(data)

X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values

X = X.reshape((len(X), 1))

reg = LinearRegression()
reg = reg.fit(X, Y)
y_pred = reg.predict(X)

rSquare = reg.score(X, Y)
print(rSquare)