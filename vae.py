"""
Variational autoencoder for Figure 4 (the bias-correction ablation).

Matches the paper's Section 6.4 setup: MNIST, a single hidden layer of 500 units
with softplus nonlinearity in both the encoder and the decoder, a 50-dimensional
Gaussian latent, and a Bernoulli likelihood on the pixels.

Forward and backward are hand-derived, like logreg.py and mlp.py. The
reparameterization noise comes from a module-level RNG with the same reseed()
pattern as mlp.py, so gradcheck can hold the noise fixed across the plus and
minus evaluations of a central difference.

Parameter order, for pack()/unpack():
    [W_e, b_e, W_mu, b_mu, W_lv, b_lv, W_d, b_d, W_o, b_o]
"""

import numpy as np

from utils import sigmoid, softplus

_rng = np.random.default_rng()

def reseed(seed):
    global _rng
    _rng = np.random.default_rng(seed)

def init_vae(input_dim=784, hidden=500, latent=50, seed=0):
    rng = np.random.default_rng(seed)
    def W(out_dim, in_dim):
        return rng.normal(size=(out_dim, in_dim)) / np.sqrt(in_dim)
    return [
        W(hidden, input_dim),  np.zeros((hidden, 1)),   # encoder hidden
        W(latent, hidden),     np.zeros((latent, 1)),   # mu
        W(latent, hidden),     np.zeros((latent, 1)),   # log variance
        W(hidden, latent),     np.zeros((hidden, 1)),   # decoder hidden
        W(input_dim, hidden),  np.zeros((input_dim, 1)),# decoder output logits
    ]

def vae_forward(params, inputs, noise=None):
    W_e, b_e, W_mu, b_mu, W_lv, b_lv, W_d, b_d, W_o, b_o = params

    X = inputs.T                                    # (784, B)

    pre_e = np.dot(W_e, X) + b_e                    # (500, B)
    h_e = softplus(pre_e)

    mu = np.dot(W_mu, h_e) + b_mu                   # (50, B)
    log_var = np.dot(W_lv, h_e) + b_lv              # (50, B)

    sigma = np.exp(0.5 * log_var)
    eps = _rng.standard_normal(mu.shape) if noise is None else noise
    z = mu + sigma * eps                            # (50, B)

    pre_d = np.dot(W_d, z) + b_d                    # (500, B)
    h_d = softplus(pre_d)

    logits = np.dot(W_o, h_d) + b_o                 # (784, B)

    return X, pre_e, h_e, mu, log_var, sigma, eps, z, pre_d, h_d, logits

def vae_backward(params, inputs, labels=None, lam=0.0):
    """
    Negative ELBO and its gradients. `labels` is ignored; a VAE reconstructs its
    input. Reconstruction and KL are both summed over their dimensions and then
    averaged over the batch, which is the scale the paper's loss axis uses.
    """
    W_e, b_e, W_mu, b_mu, W_lv, b_lv, W_d, b_d, W_o, b_o = params

    X, pre_e, h_e, mu, log_var, sigma, eps, z, pre_d, h_d, logits = vae_forward(params, inputs)
    B = X.shape[1]

    # Bernoulli cross-entropy straight from the logits: max(l,0) - l*x + log1p(exp(-|l|))
    recon = np.sum(
        np.maximum(logits, 0) - logits * X + np.log1p(np.exp(-np.abs(logits))),
        axis=0,
    )
    kl = -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var), axis=0)
    loss = np.mean(recon + kl)

    # ---- decoder ----
    d_logits = (sigmoid(logits) - X) / B            # (784, B)

    dW_o = np.dot(d_logits, h_d.T)
    db_o = np.sum(d_logits, axis=1, keepdims=True)

    d_pre_d = np.dot(W_o.T, d_logits) * sigmoid(pre_d)   # softplus' = sigmoid
    dW_d = np.dot(d_pre_d, z.T)
    db_d = np.sum(d_pre_d, axis=1, keepdims=True)

    # ---- through the reparameterization, z = mu + exp(log_var/2) * eps ----
    dz = np.dot(W_d.T, d_pre_d)                     # (50, B)

    d_mu = dz + mu / B                              # + d(KL)/d(mu)
    d_log_var = dz * eps * sigma * 0.5 + 0.5 * (np.exp(log_var) - 1) / B

    dW_mu = np.dot(d_mu, h_e.T)
    db_mu = np.sum(d_mu, axis=1, keepdims=True)
    dW_lv = np.dot(d_log_var, h_e.T)
    db_lv = np.sum(d_log_var, axis=1, keepdims=True)

    # ---- encoder ----
    d_pre_e = (np.dot(W_mu.T, d_mu) + np.dot(W_lv.T, d_log_var)) * sigmoid(pre_e)
    dW_e = np.dot(d_pre_e, X.T)
    db_e = np.sum(d_pre_e, axis=1, keepdims=True)

    grads = [dW_e, db_e, dW_mu, db_mu, dW_lv, db_lv, dW_d, db_d, dW_o, db_o]

    return loss, grads
