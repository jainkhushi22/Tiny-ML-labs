"""
Gradient Descent Visualization on y = x²
Author: khushi jain
Description:
    Demonstrates simple gradient descent optimization on the function y = x².
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the loss function and its gradient
def loss(x):
    return x ** 2

def gradient(x):
    return 2 * x

# Hyperparameters
learning_rate = 0.1
initial_x = 10
steps = 20

# Perform gradient descent
x = initial_x
x_values = [x]
for _ in range(steps):
    x -= learning_rate * gradient(x)
    x_values.append(x)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(x_values, marker='o', label="x values during descent")
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.title("Gradient Descent Optimization on y = x²")
plt.xlabel("Iteration")
plt.ylabel("x value")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
