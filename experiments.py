import numpy as np

from optimizers import Adam, AdaGrad, SGD_Nesterov
from train import train, load_MNIST
from utils import one_hot, pack, unpack
from logreg import log_reg_backward, log_reg_forward
from mlp import reseed, mlp_backward, mlp_forward

def gradcheck(eps):
    rng = np.random.default_rng(seed=0)

    W1 = rng.normal(size=(5, 20)) / np.sqrt(20)
    b1 = np.zeros((5, 1))

    W2 = rng.normal(size=(5, 5))  / np.sqrt(5)
    b2 = np.zeros((5, 1))

    W3 = rng.normal(size=(3, 5))  / np.sqrt(5)
    b3 = np.zeros((3, 1))

    X = rng.normal(size=(20,4))
    Y = one_hot(3, rng.integers(0,3, size=4))

    params = [W1, b1, W2, b2, W3, b3] 
    shapes = [p.shape for p in params]

    theta = pack(params)

    reseed(0)
    loss, grads = mlp_backward(params, X, Y, lam=0.001)
    g_analytical = pack(grads)
    g_numerical = np.zeros_like(theta)

    for i in range(theta.shape[0]):
        theta_plus = theta.copy()
        theta_plus[i] += eps
        theta_minus = theta.copy()
        theta_minus[i] -= eps
        reseed(0)

        loss_plus, grad_plus = mlp_backward(unpack(theta_plus, shapes), X, Y, lam=0.001)
        reseed(0)
        loss_minus, grad_minus = mlp_backward(unpack(theta_minus, shapes), X, Y, lam=0.001)

        g_numerical[i] = (loss_plus-loss_minus) / (2*eps)

    rel_error = np.abs(g_analytical - g_numerical) / np.maximum(
        np.abs(g_analytical) + np.abs(g_numerical), 1e-12
    )
    print(f"max relative error: {rel_error.max():.2e} at index {rel_error.argmax()}")
    print(f"mean relative error: {rel_error.mean():.2e}")
    assert rel_error.max() < 1e-5, "gradcheck failed"

def MNIST_MLP(optimizer_fn, epochs, inputs, labels):
    np.random.seed(0)

    W1 = np.random.randn(1000, 784) / np.sqrt(784)
    b1 = np.zeros((1000, 1))

    W2 = np.random.randn(1000, 1000) / np.sqrt(1000)
    b2 = np.zeros((1000, 1))

    W3 = np.random.randn(10, 1000) / np.sqrt(1000)
    b3 = np.zeros((10, 1))

    params = [W1, b1, W2, b2, W3, b3] 

    theta = pack(params)
    optimizer = optimizer_fn(theta)

    return train(mlp_backward, params, optimizer, inputs, labels, epochs)

def MNIST_log_reg(optimizer_fn, epochs, inputs, labels):
    np.random.seed(0)
    
    W = np.random.randn(10, 784) * 0.01
    b = np.random.randn(10, 1) * 0.01
    params = [W, b]

    theta = pack(params)
    optimizer = optimizer_fn(theta)

    return train(log_reg_backward, params, optimizer, inputs, labels, epochs)

if __name__ == "__main__":
    inputs, labels = load_MNIST()

    epochs = 2
    stepsize = 0.001
    decay_1 = 0.9
    decay_2 = 0.999
    ζ = 1e-8

    gradcheck(1e-7)

#-----Experiment: Bias-Correction Term-----------------------------------------------------------


#-----Experiment: Convolutional Neural Networks--------------------------------------------------


#-----Experiment: Multi-layer Perceptron (MLP)---------------------------------------------------
MNIST_MLP_Adam = MNIST_MLP(
    lambda w: Adam(stepsize, w, decay_1, decay_2, ζ),
    epochs, inputs, labels
)
# MNIST_MLP_AdaGrad = MNIST_MLP(
#     lambda w: AdaGrad(stepsize, w, ζ),
#     epochs, inputs, labels
# )
# MNIST_MLP_SGD_Nesterov = MNIST_MLP(
#     lambda w: SGD_Nesterov(stepsize, w, decay_1),
#     epochs, inputs, labels
# )

# plot_results(
#     [
#         (MNIST_MLP_Adam,         'Adam'),
#         (MNIST_MLP_AdaGrad,      'AdaGrad'),
#         (MNIST_MLP_SGD_Nesterov, 'SGD + Nesterov'),
#     ],
#     'MNIST MLP + Dropout — Figure 2(a) Recreation',
#     'epoch',
#     'training loss'
# )

#-----Experiment: Logistic Regression------------------------------------------------------------
 
    # MNIST_log_reg_Adam = MNIST_log_reg(
    #     lambda w: Adam(stepsize, w, decay_1, decay_2, ζ),
    #     epochs, 
    #     inputs, 
    #     labels
    # )

    # MNIST_log_reg_AdaGrad = MNIST_log_reg(
    #     lambda w: AdaGrad(stepsize, w, ζ),
    #     epochs, 
    #     inputs, 
    #     labels
    # )

    # MNIST_log_reg_SGD_Nestrov = MNIST_log_reg(
    #     lambda w: SGD_Nesterov(stepsize, w, decay_1),
    #     epochs, 
    #     inputs, 
    #     labels
    # )

    # plot_results(
    #     [
    #         (MNIST_log_reg_Adam,         'Adam'),
    #         (MNIST_log_reg_AdaGrad,      'AdaGrad'),
    #         (MNIST_log_reg_SGD_Nestrov, 'SGD + Nesterov'),
    #     ],
    #     'MNIST Logistic Regression — Figure 1 Recreation',
    #     'epoch',
    #     'training loss'
    # )