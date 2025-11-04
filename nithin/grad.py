import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)

# Add bias term (x0 = 1)
X = np.c_[np.ones((100, 1)), x]

# Initialize parameters
theta = np.zeros((2, 1))
alpha = 0.01
m = len(y)
cost_history = []

# Gradient Descent
for _ in range(1000):
    error = X.dot(theta) - y
    theta -= alpha * (1/m) * X.T.dot(error)
    cost = (1/(2*m)) * np.sum(error**2)
    cost_history.append(cost)

# Print results
print("Optimized parameters (theta):")
print(theta)
print(f"Final Cost: {cost_history[-1]:.6f}")

# Plot regression line
plt.scatter(x, y, color='blue', label='Training Data')
plt.plot(x, X.dot(theta), color='red', label='Fitted Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression using Gradient Descent')
plt.legend()
plt.savefig("plot1.png")
plt.close()
#plt.show()

# Plot cost history
plt.plot(cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function Decrease')
#plt.show()
plt.savefig("plot2.png")
plt.close()