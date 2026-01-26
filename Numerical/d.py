from matplotlib import pyplot as plt
import numpy as np
from scipy.linalg import svd
#1
def g(x):
  return np.exp(-x**2)

x = np.arange(-10,50)

y = np.array([g(i) for i in x])
plt.scatter(x, y, color='red')
plt.show()

#2
def basis_expansion(x, d):
    return np.power(x, np.arange(d+1))

N = 60
x = np.arange(0, N)
y = g(x)
d = 3
reg = 10
X = np.zeros((N, d+1))
for i in range(N):
    X[i] = basis_expansion(x[i], d)

U, D, VT = svd(X, full_matrices=False)

D_reg = D / (reg + D**2)
D_inv_reg = np.diag(D_reg)

w_best = VT.T.dot(D_inv_reg).dot(U.T).dot(y)

y_predicted = X.dot(w_best)

plt.scatter(x, y, color='red')
plt.plot(x, y_predicted)
plt.show()

#3
def basis_expansion(x, d):
    return np.power(x, np.arange(d+1))

N = 60
x = np.arange(0, N)
y = g(x)
d = 3
reg = 100

X = np.zeros((N, d+1))
for i in range(N):
    X[i] = basis_expansion(x[i], d)
U, d, VT = svd(X, full_matrices=False)

for i in range(d.shape[0]):
  d[i] = d[i] / (reg + (d[i] * d[i]))

D = np.diag(d)

w_best = VT.T.dot(D).dot(U.T).dot(y)

y_predicted = X.dot(w_best)

plt.scatter(x, y, color='red')
plt.plot(x, y_predicted, color='green')
plt.show()

#4

def g(x):
    return np.sin((2 * np.pi * x) / 60) + np.random.random(len(x)) * 0.1
def basis_expansion(x, d):
    return np.power(x, np.arange(d + 1))

N = 60
x = np.arange(0, N)
y = g(x)
d = 3
reg = 100

X = np.zeros((N, d + 1))
for i in range(N):
    X[i] = basis_expansion(x[i], d)

U, S, VT = svd(X, full_matrices=False)

S_reg = S / (reg + S**2)
D_inv_reg = np.diag(S_reg)

w_best = VT.T.dot(D_inv_reg).dot(U.T).dot(y)

x_continuous = np.linspace(0, N-1, 500)
X_continuous = np.zeros((len(x_continuous), d + 1))
for i in range(len(x_continuous)):
    X_continuous[i] = basis_expansion(x_continuous[i], d)
y_predicted = X_continuous.dot(w_best)

plt.scatter(x, y, color='red')
plt.plot(x_continuous, y_predicted, color='green')
plt.show()