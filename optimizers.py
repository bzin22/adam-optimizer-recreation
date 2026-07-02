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
    def __init__(self, stepsize, weights, decay_1, decay_2, epsilon):
        self.stepsize = stepsize
        self.weights = weights
        self.decay_1 = decay_1
        self.decay_2 = decay_2
        self.epsilon = epsilon

        self.m_t, self.v_t = np.zeros(self.weights.shape), np.zeros(self.weights.shape)
        self.t = 0

    def update(self, grad):
        self.m_t = self.decay_1 * self.m_t + (1 - self.decay_1) * grad
        self.v_t = self.decay_2 * self.v_t + (1 - self.decay_2) * grad**2

        m_hat = self.m_t/(1 - self.decay_1**self.t)
        v_hat = self.v_t/(1 - self.decay_2**self.t)

        self.weights -= self.stepsize * (1/self.t**0.5) * m_hat / (v_hat**0.5 + self.epsilon)

        return self.weights
    
    def step_epoch(self):
        self.t += 1
        return self.t