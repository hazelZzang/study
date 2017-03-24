######################
# deep learning from scratch
# example
#######################

import numpy as np

def init_network():
    network = {}
    #weight
    network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['W3'] = np.array([[0.1,0.3], [0.2,0.4]])
    #bias
    network['b1'] = np.array([0.1,0.2,0.3])
    network['b2'] = np.array([0.1, 0.2])
    network['b3'] = np.array([0.1, 0.2])
    return network

#signoid
def sigmoid(x):
    return 1 / ( 1 + np.exp(-x))

#ReLU
def relu(x):
    return np.maximum(0,x)

#softmax function
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def forward(network,actFunc, x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']

    #floor 1
    A1 = np.dot(X, W1) + b1
    z1 = actFunc(A1)

    #floor 2
    A2 = np.dot(z1, W2) + b2
    z2 = actFunc(A2)

    #output
    A3 = np.dot(z2, W3) + b3
    Y = A3
    return Y

#input
X = np.array([1.0,0.5])
network = init_network()
y = forward(network, relu, X)
y1 = forward(network, sigmoid, X)
