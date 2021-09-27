import numpy as np
import matplotlib as mpt
from numpy import arange
from matplotlib import pyplot

# objective function
def objective(x):
	return x**2.0

# define range for input
r_min, r_max = -1.0, 1.0
# sample input range uniformly at 0.1 increments
inputs = arange(r_min, r_max+0.1, 0.1)
# compute targets
results = objective(inputs)
# create a line plot of input vs result
pyplot.plot(inputs, results)
# show the plot
pyplot.show()


# derivative of objective function
def derivative(x):
	return x * 2.0

# gradient descent algorithm
def gradient_descent(objective, derivative, bounds, step_size):
    solution = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    while abs(derivative(solution))>1e-5:
		# calculate gradient
    		gradient = derivative(solution)
		# take a step
    		solution = solution - step_size * gradient
		# evaluate candidate point
    		solution_eval = objective(solution)
		# report progress
    return [solution, solution_eval,gradient]

# define range for input
bounds = np.array([[-1.0, 1.0]])
# define the step size
step_size = 0.1
# perform the gradient descent search
best, score,gradient = gradient_descent(objective, derivative, bounds, step_size)
print('Done!')
print('f(%s) = %f' % (best, score))
print(abs(gradient))












