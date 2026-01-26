import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad

#1
t = np.array([0.5,1, 1.5])
a = np.array([45,26,40])

def f(x_new):
    d = t.shape[0] - 1
    s = 0.0
    for i in range(d+1):
        p = a[i]
        for j in range(d+1):
            if i != j:
                p = p * ((x_new - t[j]) / (t[i] - t[j]))
        s = s + p
    return s

a_ = np.array([f(_) for _ in t])

plt.scatter(t, a, color='red')
plt.plot(t, a_)
plt.show()

#2
integral, error = quad(f, 0.5, 1.5)
average_velocity = integral / (1.5 - 0.5)
print("Average Velocity= ",average_velocity)

#3

def trapezoidal_rule(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    area = (h / 2) * (y[0] + 2 * sum(y[1:-1]) + y[-1])
    return area

def f_linear(x_new):
    return np.interp(x_new, t, a)

intt = trapezoidal_rule(f_linear, 0.5, 1.5, 2)

average_velocity_trapezoidal = intt / (1.5 - 0.5)
print("Average velocity (1st order integration): ",average_velocity_trapezoidal)

print("Difference in average velocity: ",average_velocity - average_velocity_trapezoidal)