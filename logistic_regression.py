# Initial imports
import numpy as np
import theano.tensor as T
from theano import shared, function
from learning_functions import nesterov_momentum, apply_nesterov_momentum, sgd
# Create a sample logistic regression problem.
true_w = rng.randn(100)
true_b = rng.randn()
xdata = rng.randn(50, 100)
ydata = (np.dot(xdata, true_w) + true_b) > 0.0

# Step 1. Declare Theano variables
x = T.dmatrix()
y = T.dvector()
w = shared(rng.randn(100))
b = shared(numpy.zeros(()))
params = [w,b]
updates = nesterov_momentum(cost, params, learning_rate, 0.9)

# Step 2. Construct Theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))
xent = -y * T.log(p_1) - (1 - y) * T.log(1 - p_1)
prediction = p_1 > 0.5
cost = xent.mean() + 0.01 * (w ** 2).sum()
gw, gb = T.grad(cost, [w, b])

# Step 3. Compile expressions to functions
train = function(inputs=[x, y],
                 outputs=[prediction, xent],
                 updates=updates)

# Step 4. Perform computation
for loop in range(100):
    pval, xval = train(xdata, ydata)
    print xval.mean()
