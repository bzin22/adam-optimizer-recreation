from optimizers import Adam, AdaGrad, SGD_Nesterov
from train import train, load_MNIST
from utils import *
from logreg import *

def MNIST_log_reg(optimizer_fn, epochs, inputs, labels):
    W = np.random.randn(10, 784) * 0.01
    b = np.random.randn(10, 1) * 0.01
    params = [W, b]

    theta = pack(params)
    optimizer = optimizer_fn(theta)

    return train(log_reg_backward, params, optimizer, inputs, labels, epochs)

if __name__ == "__main__":
    inputs, labels = load_MNIST()

    epochs = 45
    stepsize = 0.001
    decay_1 = 0.9
    decay_2 = 0.999
    ζ = 1e-8

    MNIST_log_reg_Adam = MNIST_log_reg(
        lambda w: Adam(stepsize, w, decay_1, decay_2, ζ),
        epochs, 
        inputs, 
        labels
    )

    MNIST_log_reg_AdaGrad = MNIST_log_reg(
        lambda w: AdaGrad(stepsize, w, ζ),
        epochs, 
        inputs, 
        labels
    )

    MNIST_log_reg_SGD_Nestrov = MNIST_log_reg(
        lambda w: SGD_Nesterov(stepsize, w, decay_1),
        epochs, 
        inputs, 
        labels
    )

    plot_results(
        [
            (MNIST_log_reg_Adam,         'Adam'),
            (MNIST_log_reg_AdaGrad,      'AdaGrad'),
            (MNIST_log_reg_SGD_Nestrov, 'SGD + Nesterov'),
        ],
        'MNIST Logistic Regression — Figure 1 Recreation',
        'epoch',
        'training loss'
    )