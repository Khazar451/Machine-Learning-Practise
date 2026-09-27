import numpy as np
from sklearn.preprocessing import add_dummy_feature
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(seed=42)
m = 200 # number of instances
X = 2 * rng.random((m, 1)) # column vector
y = 4 + 3 * X + rng.standard_normal((m, 1)) # column vector

X_b = add_dummy_feature(X) # add x0 = 1 to each instance
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
# now we can make predictions using theta_best
X_new = np.array([[0], [2]])
X_new_b = add_dummy_feature(X_new) # add x0 = 1 to each instance
y_predict = X_new_b @ theta_best
# plot the model's prediction for a better view
plt.plot(X_new, y_predict, "r-", label="Predictions")
plt.plot(X, y, "b.")
# plt.show()
# now perform the linear regression usin sklearn
lin_reg = LinearRegression()
lin_reg.fit(X, y)
# print(lin_reg.intercept_, lin_reg.coef_)
# print(lin_reg.predict(X_new))
# lets look at another implementation
eta = 0.1 # learning rate
n_epochs = 1000
m = len(X_b) # number of instances
rng = np.random.default_rng(seed=42)
theta = rng.standard_normal((2, 1)) # randomly initialized model parameters

for epoch in range(n_epochs):
    gradients = 2 / m * X_b.T @ (X_b @ theta - y)
    theta = theta - eta * gradients

print(theta)
# print(y_predict)
