from matplotlib import pyplot as plt
import numpy as np

#1

x = np.array([0.5, 1, 1.5, 2,2.5])
y = np.array([np.log((1+t)) for t in x])

def find_b():
    d = x.shape[0]
    b_table = np.zeros((d, d))
    b_table[:, 0] = y[:]

    for col in range(1, d):
        for row in range(d-col):
            del_x =  x[col+row] - x[row]
            b_table[row, col] = (b_table[row+1, col-1] - b_table[row, col-1]) / del_x
    return b_table[0, :]

def ndd_polynomial(x_new):
    b = find_b()
    d = x.shape[0]
    s = b[0]
    for i in range(1, d):
        prod = 1
        for j in range(i):
            prod = prod * (x_new - x[j])
        s = s + b[i] * prod
    return s

y_ = np.array([ndd_polynomial(_) for _ in x])

plt.scatter(x, y, color='red')
plt.plot(x, y_)
plt.show()

#2
import numpy as np
from math import fabs, log

def tpre(tv, av):
    if tv != 0:
        return fabs((tv - av) / tv) * 100

def v(t):
    return np.log(1 + t)

def F(t):
    return t - np.log(1 + t)

def nseg_second_order(x, y):
    n_seg = x.shape[0] - 1
    h = (x[-1] - x[0]) / n_seg
    I = y[0] + y[-1]
    for i in range(1, n_seg, 2):
        I += 4 * y[i]
    for i in range(2, n_seg-1, 2):
        I += 2 * y[i]
    return (h * I) / 3

t0, t4 = 0.5, 2.5
n = 4
h = (t4 - t0) / (2 * n)
x = np.array([t0 + k * h for k in range(2 * n + 1)])
y = np.array([v(_) for _ in x])
tv = F(t4) - F(t0)
av = nseg_second_order(x, y)

print('True Value (TV) = ', tv)
print('Approximate Value (AV) = ', av)
print('True Percent Relative Error (TPRE) = ', tpre(tv=tv, av=av), '%')

#3
print("After using  1st order method--")
def nseg_first_order(f, a, b, n):
    h = (b - a) / n
    I = f(a) + f(b)
    for i in range(n):
        I = I + 2 * f(a + i*h)
    return (h * I) / 2

t0, t4 = 0.5, 2.5
n = 4

av = nseg_first_order(v, t0, t4, n)
print('TV= ', tv)
print('AV= ', av)
print('TPRE= ', tpre(tv=tv, av=av), '%')