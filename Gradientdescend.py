import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X] # shape (100, 2)
theta = np.zeros((2, 1)) # [theta0, theta1]
alpha = 0.1 # learning rate
iterations = 1000
m = len(y) # number of samples
cost_history = []
for i in range(iterations):
    predictions = X_b.dot(theta) # h(x)
    errors = predictions - y # errors
    gradients = (1/m) * X_b.T.dot(errors) # gradient
    theta = theta - alpha * gradients # update rule
    cost = (1/(2*m)) * np.sum(errors ** 2)
    cost_history.append(cost)
    if i > 0 and abs(cost_history[-2] - cost_history[-1]) < 1e-6:
        print("Optimized parameters (theta):")
        print(theta)
        print(f"Final cost: {cost_history[-1]:.6f}")
        print(f"Total iterations: {i+1}")
plt.scatter(X, y, color='blue', label='Training data')
plt.plot(X, X_b.dot(theta), color='red', label='Fitted line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression using Gradient Descent')
plt.legend()
#plt.savefig("plot1.png")  # give each figure a different name
#plt.close()
plt.show()
plt.plot(cost_history)
plt.xlabel('Iteration')
plt.ylabel('Cost J(θ)')
plt.title('Cost Function Decrease over Iterations')
plt.show()
#plt.savefig("plot2.png")  # give each figure a different name
#plt.close()