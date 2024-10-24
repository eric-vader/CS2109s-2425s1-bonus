import numpy as np
from functools import reduce

def relu(x):
    return x * (x > 0)

class FconLayer(object):
    def __init__(self, W, a_fn=relu, has_bias=True):
        self.W = W
        self.a_fn = a_fn
        self.has_bias = has_bias
    def __call__(self, X_T):
        raise NotImplemented


l1 = FconLayer(np.array([
    [0.1, 0.1],
    [-0.1, 0.2],
    [0.3, -0.4]
]))
l2 = FconLayer(np.array([
    [0.1, 0.1],
    [0.5, -0.6],
    [0.7, -0.8]
]))

class NNetwork(object):
    def __init__(self, layers):
        self.layers = layers 
    def __call__(self, X):
        raise NotImplemented # TODO Implement using reduce

nn_qc1 = NNetwork([l1, l2])
x_qc1 = np.array([[2,3]])
y_qc1 = np.array([[0.1, 0.9]])
y_pred = nn_qc1(x_qc1)

mse = ((y_pred - y_qc1)**2).mean()
print(f"qc1 MSE: {mse}")

Ws = [
    np.array([
        [1.4, 0.6],
        [0.8, 0.6]
    ]),
    np.array([
        [2.1, -0.5],
        [0.7, 1.9]
    ]),
    np.array([
        [1.2, -2.2],
        [1.2, 1.3]
    ])
]
    
nn_qd1_noact = NNetwork([ FconLayer(W, lambda x:x, has_bias=False) for i, W in enumerate(Ws) ])
nn_qd1_silu = NNetwork([ FconLayer(W, (lambda x:x/(1+np.exp(-x))) if i < 2 else (lambda x:x), has_bias=False) for i, W in enumerate(Ws) ])
x_qd1 = np.array([[1,0], [2,0]])

M = None # TODO Implement this using reduce
nn_qd1_l1 = NNetwork([ FconLayer(M, lambda x:x, has_bias=False) ])

is_equal = np.allclose(nn_qd1_noact(x_qd1), nn_qd1_l1(x_qd1))
print("qd1 compressed M results is equal - {is_equal}")
print("qd1 counter example:")
print(nn_qd1_silu(x_qd1))

# Expected output:
# Forward Propagation MSE: 0.48500000000000004
# Non-linear Activation Functions compressed M results is equal - {is_equal}
# Non-linear Activation Functions counter example:
# [[  3.05710826  -5.27270693]
#  [  7.72572945 -13.2458166 ]]