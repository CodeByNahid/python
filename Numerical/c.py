import numpy as np

#1
def f(x):
  return np.exp(-x**2)

def f_deri(x,del_x):
  return (-f(x+del_x)+(8*f(x+del_x))-8*f(x-del_x)+f(x-2*del_x))/12*del_x
print("Derivative of that function: ",f_deri(1.75,0.05))

#2
def f(x):
    return np.exp(-x**2)

def gradient_descent(x_initial, learning_rate=1.0, max_iterations=500, tolerance=0.005):
    x = x_initial
    iterations = 0
    while iterations < max_iterations:
        grad = f_deri(x, 0.001)
        if abs(grad) < tolerance:
            break
        x = x - (learning_rate / (iterations + 1)) * grad
        iterations += 1
    return x, f(x), iterations

x_min, min_value, iterations = gradient_descent(1.0)
print("Minimum value of f(x) is: ",min_value)

#3

def tpr(tv,av):
  return (np.fabs(tv-av)/tv)*100

tv=2 * 1.75 * np.exp(-1.75**2)
av=f_deri(1.75,0.005)

print("True percent relative error: ",tpr(tv,av))