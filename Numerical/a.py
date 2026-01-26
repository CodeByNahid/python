
import numpy as np
from numpy.random import random
from math import pi,cos
from matplotlib import pyplot as plt
from scipy.linalg import svd

#1
def f(n):
  return  2*cos((2*pi*𝑛)/100)+random()

x = np.arange(100)

y = np.array([f(i) for i in x])
plt.scatter(x, y, color='red')

#2

def basis_expansion(x, d):
    return np.power(x, np.arange(d+1))

N = 100
x = np.arange(N)
y = 2*np.cos((2*pi*x)/100)+random(N)
d = 3
reg = 10

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


#3
def basis_expansion(x, d):
    return np.power(x, np.arange(d+1))

N = 100
x = np.arange(N)
y = 2*np.cos((2*pi*x)/100)+random(N)
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

y_filtered = y_predicted
plt.plot(x, y_filtered, color='brown')
plt.show()



