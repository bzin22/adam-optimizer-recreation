"""
Recreation of: "Adam: A Method for Stochastic Optimization"
Kingma & Ba, ICLR 2015
https://arxiv.org/abs/1412.6980

Author: Bryan Zin
Date: June 2026

Variable names as they appear in the paper vs in the code: 

α: stepsize/learning rate
β1: decay_1
β2: decay_2
ζ: epsilon
θ: weights
g: grad (gradient)
t: timestep
"""

import numpy as np

class RMSProp:
    def __init__(self, stepsize, weights, decay, epsilon):
        self.stepsize = stepsize
        self.weights = weights
        self.decay = decay
        self.epsilon = epsilon
        self.v_t = np.zeros_like(weights)

    def update(self, grad):
        self.v_t = self.decay * self.v_t + (1 - self.decay) * grad**2
        self.weights -= self.stepsize * grad / (np.sqrt(self.v_t) + self.epsilon)

        return self.weights


class AdaDelta:
    def __init__(self, weights, decay=0.95, epsilon=1e-6, stepsize=1.0):
        self.weights = weights
        self.decay = decay
        self.epsilon = epsilon
        self.stepsize = stepsize

        self.eg2 = np.zeros_like(weights)      # running avg of squared gradients
        self.edx2 = np.zeros_like(weights)     # running avg of squared updates

    def update(self, grad):
        self.eg2 = self.decay * self.eg2 + (1 - self.decay) * grad**2

        rms_dx = np.sqrt(self.edx2 + self.epsilon)
        rms_g = np.sqrt(self.eg2 + self.epsilon)

        delta = -self.stepsize * (rms_dx / rms_g) * grad
        self.weights += delta

        self.edx2 = self.decay * self.edx2 + (1 - self.decay) * delta**2
        
        return self.weights

class AdaGrad:
    def __init__(self, stepsize, weights, epsilon):
        self.stepsize = stepsize
        self.weights = weights
        self.epsilon = epsilon
        self.v_t = np.zeros(self.weights.shape)

    def update(self, grad):
        self.v_t += grad**2
        self.weights -= self.stepsize * grad/(self.v_t**0.5 + self.epsilon)

        return self.weights

class SGD_Nesterov: 
    def __init__(self, stepsize, weights, decay):
        self.stepsize = stepsize
        self.weights = weights
        self.decay = decay

        self.v_t = np.zeros(self.weights.shape)
        self.v_t_minus_1 = np.zeros(self.weights.shape)

    def update(self, grad):
        self.v_t_minus_1 = self.v_t
        self.v_t = self.decay * self.v_t + self.stepsize * grad
        self.weights -= self.v_t + self.decay * (self.v_t - self.v_t_minus_1)

        return self.weights
         
class Adam:
    def __init__(self, stepsize, weights, decay_1, decay_2, epsilon, bias_correction=True):
        self.stepsize = stepsize
        self.weights = weights
        self.decay_1 = decay_1
        self.decay_2 = decay_2
        self.epsilon = epsilon
        self.bias_correction = bias_correction

        self.m_t, self.v_t = np.zeros_like(weights), np.zeros_like(weights)
        self.t = 0

    def update(self, grad):
        self.t += 1

        self.m_t = self.decay_1 * self.m_t + (1 - self.decay_1) * grad
        self.v_t = self.decay_2 * self.v_t + (1 - self.decay_2) * grad**2

        if self.bias_correction:
            m_hat = self.m_t / (1 - self.decay_1**self.t)
            v_hat = self.v_t / (1 - self.decay_2**self.t)
        else:
            # Algorithm 1 with the /(1-β1^t) and /(1-β2^t) terms removed. This is the
            # ablation measured in Figure 4.
            m_hat = self.m_t
            v_hat = self.v_t

        # step_size_modifier = (1/self.t**0.5)

        self.weights -= self.stepsize * m_hat / (v_hat**0.5 + self.epsilon)

        return self.weights
    
    # def step_epoch(self):
    #     self.t += 1
    #     return self.t

class AdaMax:
    """
    Algorithm 2 from the paper: Adam with the second moment replaced by an
    exponentially weighted infinity norm.

    Differs from Adam in three ways: u_t is a running max of |g| rather than a
    running average of g**2, u_t needs no bias correction (it is not an estimate
    of a moment), and there is no epsilon in the denominator.
    """
    def __init__(self, stepsize, weights, decay_1, decay_2, epsilon=1e-8):
        self.stepsize = stepsize
        self.weights = weights
        self.decay_1 = decay_1
        self.decay_2 = decay_2
        self.epsilon = epsilon

        self.m_t = np.zeros_like(weights)
        self.u_t = np.zeros_like(weights)
        self.t = 0

    def update(self, grad):
        self.t += 1

        self.m_t = self.decay_1 * self.m_t + (1 - self.decay_1) * grad

        # The paper writes u_t = max(β2·u_{t-1}, |g_t|) with no epsilon. Taken
        # literally, a coordinate whose gradient is exactly 0 at t=1 gives
        # u_t = 0 and m_t = 0, so the update is 0/0 and that weight is NaN for
        # the rest of the run. Dead ReLUs make this common on the first
        # minibatch. Floor |g| the way torch.optim.Adamax does: it only changes
        # the degenerate case and leaves the infinity norm intact everywhere else.
        self.u_t = np.maximum(self.decay_2 * self.u_t, np.abs(grad) + self.epsilon)

        stepsize_t = self.stepsize / (1 - self.decay_1**self.t)
        self.weights -= stepsize_t * self.m_t / self.u_t

        return self.weights
