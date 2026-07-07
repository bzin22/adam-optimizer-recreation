import numpy as np

from utils import relu, softmax

_rng = np.random.default_rng()

def reseed(seed): # only used for gradcheck, not used in actual experiements
    global _rng
    _rng = np.random.default_rng(seed)

def mlp_forward(params, inputs, training=True):
    W1, b1, W2, b2, W3, b3  = params
    inputs = inputs.T

    p_keep_input = 0.8 # keep 80% of the inputs 
    p_keep_hidden = 0.5 # keep 50% of the hidden layers

    if training:
        scaled_mask_i = (1/p_keep_input) * (_rng.random(inputs.shape) < p_keep_input).astype(np.float32)
        inputs = inputs * scaled_mask_i

    # layer 1
    z1 = np.dot(W1,inputs) + b1
    a1 = relu(z1)
    if training:
        scaled_mask_1 = (1/p_keep_hidden) * (_rng.random(a1.shape) < p_keep_hidden).astype(np.float32)
        a1 *= scaled_mask_1

    # layer 2
    z2 = np.dot(W2, a1) + b2
    a2 = relu(z2)
    if training: 
        scaled_mask_2 = (1/p_keep_hidden) * (_rng.random(a2.shape) < p_keep_hidden).astype(np.float32)
        a2 *= scaled_mask_2

    # output layer
    z3 = np.dot(W3, a2) + b3
    out = softmax(z3)

    if training: 
        return [a1, a2, out, scaled_mask_i, scaled_mask_1, scaled_mask_2]
    return [a1, a2, out]

def mlp_backward(params, inputs, labels, lam):
    W1, b1, W2, b2, W3, b3  = params

    a1, a2, out, scaled_mask_i, scaled_mask_1, scaled_mask_2 = mlp_forward(params, inputs)
    inputs = inputs.T

    loss = -np.mean(np.sum(labels * np.log(out + 1e-12), axis=0)) \
            + (lam / 2) * (np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2))
    
    # output layer
    dLdz_3 = (out - labels) / inputs.shape[1]

    dLdW_3 = np.dot(dLdz_3, a2.T) + lam * W3
    dLdb_3 = np.sum(dLdz_3, axis=1, keepdims=True)
    dLda_2 = np.dot(W3.T, dLdz_3) * scaled_mask_2

    # layer 2
    dLdz_2 = dLda_2 * np.where(a2 > 0, 1, 0)

    dLdW_2 = np.dot(dLdz_2, a1.T) + lam * W2
    dLdb_2 = np.sum(dLdz_2, axis=1, keepdims=True)
    dLda_1 = np.dot(W2.T, dLdz_2) * scaled_mask_1

    # layer 1
    dLdz_1 = dLda_1 * np.where(a1 > 0, 1, 0) 

    dLdW_1 = np.dot(dLdz_1, (inputs * scaled_mask_i).T) + lam * W1
    dLdb_1 = np.sum(dLdz_1, axis=1, keepdims=True)

    grads = [dLdW_1, dLdb_1, dLdW_2, dLdb_2, dLdW_3, dLdb_3]

    return loss, grads