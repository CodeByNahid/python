import numpy as np
#1
def f(x, y):
    return np.exp(-(x**2 + y**2) / 2)
x = 1.75
y = 0.27
delta_x = 0.05
delta_y = 0.075

partial_f_x = (f(x + delta_x, y) - f(x, y)) / delta_x

partial_f_y = (f(x, y + delta_y) - f(x, y)) / delta_y

print("Partial derivative with respect to x at (1.75, 0.27):", partial_f_x)
print("Partial derivative with respect to y at (1.75, 0.27):", partial_f_y)

#2

def f(x, y):
    return np.exp(-(x**2 + y**2) / 2)

def partial_f_x(x, y, delta_x=0.05):
    return (f(x + delta_x, y) - f(x, y)) / delta_x

def partial_f_y(x, y, delta_y=0.075):
    return (f(x, y + delta_y) - f(x, y)) / delta_y

def gradient_descent(x0, y0, max_iter=500, tol=0.05):
    x, y = x0, y0
    for i in range(max_iter):
        grad_x = partial_f_x(x, y)
        grad_y = partial_f_y(x, y)
        x_new = x - grad_x
        y_new = y - grad_y
        if abs(grad_x) < tol and abs(grad_y) < tol:
            print(f"Converged after {i+1} iterations")
            break
        
        x, y = x_new, y_new
    
    return x, y, f(x, y)

x0, y0 = 0, 0
x_min, y_min, f_min = gradient_descent(x0, y0)

print("Minimized value of x:", x_min)
print("Minimized value of y:", y_min)
print("Function value at minimum:", f_min)