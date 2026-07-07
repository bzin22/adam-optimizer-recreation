import numpy as np

from utils import softmax

def log_reg_forward(params, inputs):
    W, b = params
    return softmax(np.dot(W, inputs.T)+ b)

def log_reg_backward(params, inputs, labels, lam):
    W, b = params
    p = log_reg_forward(params, inputs)
    batch = p.shape[1]

    loss = -np.mean(np.sum(labels * np.log(p + 1e-12), axis=0))
    grad_W = (1 / batch) * np.dot((p-labels),inputs) + 2 * lam * W
    grad_b = (1 / batch) * np.sum((p-labels), axis=1, keepdims=True)

    return loss, [grad_W, grad_b]