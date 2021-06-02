import numpy as np

# N : batch size, D_in : input dimension, H : hidden layer dimension, D_out : output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input, output data
x = np.random.rand(N, D_in)
y = np.random.rand(N, D_out)

# Initialize weights with random value
w1 = np.random.rand(D_in, H)
w2 = np.random.rand(H, D_out)

learning_rate = 1e-6
for t in range(500):
    #forward propagation: compute predicted score y
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # Compute and print loss
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # compute gradiant and backpropagate
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # Compute 
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
