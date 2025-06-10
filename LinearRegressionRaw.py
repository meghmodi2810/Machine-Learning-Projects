import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams['figure.figsize'] = (20.0, 10.0)

data = pd.read_csv('allCSV/headbrain.csv')
# print(data.shape)
# print(data)

X = data['Head Size(cm^3)'].values
Y = data['Brain Weight(grams)'].values



# --- calculating the value of variables ---

mean_x = np.mean(X)
mean_y = np.mean(Y)

numerator = 0
denominator = 0
for i in range(len(X)):
    numerator += (X[i] - mean_x) * (Y[i] - mean_y)
    denominator += (X[i] - mean_x) ** 2

b1 = numerator / denominator
b0 = mean_y - (b1 * mean_x)

print(b1, b0)



# --- calculating the regression variables ---

min_x = np.min(X) + 100
max_x = np.max(X) - 100

x = np.linspace(min_x, max_x, 1000)
y = b0 + b1 * x



# --- plotting the line of regression ---

plt.plot(x, y, color="red", label="Regression line")
plt.scatter(X, Y, color="blue", label="Scatter plot")
plt.xlabel("Head Size in cms")
plt.ylabel("Brain weight in grams")
plt.legend()
plt.savefig("plots/LinearRegressionExample.png")


# --- finding the method of best fit (R square)

ss_t = 0
ss_r = 0
for i in range(len(X)):
    y_pred = b0 + b1 * X[i]
    ss_t = (Y[i] - mean_y) ** 2
    ss_r = (Y[i] - y_pred) ** 2

rSquare = 1 - (ss_r / ss_t)
print(rSquare)