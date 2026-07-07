import numpy as np

from optimizers import Adam, AdaGrad, SGD_Nesterov
from train import train, load_MNIST
from utils import one_hot, pack, unpack, plot_results
from logreg import log_reg_backward, log_reg_forward
from mlp import reseed, mlp_backward, mlp_forward

def gradcheck(eps):
    rng = np.random.default_rng(seed=0)

    W1 = rng.normal(size=(5, 20)) / np.sqrt(20)
    b1 = rng.normal(size=(5, 1)) * 0.1

    W2 = rng.normal(size=(5, 5))  / np.sqrt(5)
    b2 = rng.normal(size=(5, 1)) * 0.1

    W3 = rng.normal(size=(3, 5))  / np.sqrt(5)
    b3 = rng.normal(size=(3, 1)) * 0.1

    X = rng.normal(size=(4,20))
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
    reseed(0)           # dropout mask stream (mlp._rng)

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

    epochs = 200
    stepsize = 0.001 # best is still default settings from paper, 0.001
    decay_1 = 0.9
    decay_2 = 0.999
    ζ = 1e-8
    sgd_alpha_best = 0.03
    adagrad_alpha_best = 0.01

    gradcheck(1e-5)

    #-----Probe: SGD+Nesterov learning rate grid (15 epochs)------------------------------------------
#     sgd_grid = [0.003, 0.01, 0.03, 0.1]
#     sgd_results = {}
 
#     for alpha in sgd_grid:
#         print(f"\n--- SGD+Nesterov, alpha={alpha} ---")
#         history = MNIST_MLP(
#             lambda w, a=alpha: SGD_Nesterov(a, w, decay_1),
#             epochs, inputs, labels
#         )
#         sgd_results[alpha] = history
#         np.save(f'probe_sgd_nesterov_a{alpha}.npy', np.array(history))
 
# #-----Probe: AdaGrad learning rate grid (15 epochs)-----------------------------------------------
#     adagrad_grid = [0.001, 0.01, 0.05]
#     adagrad_results = {}
 
#     for alpha in adagrad_grid:
#         print(f"\n--- AdaGrad, alpha={alpha} ---")
#         history = MNIST_MLP(
#             lambda w, a=alpha: AdaGrad(a, w, ζ),
#             epochs, inputs, labels
#         )
#         adagrad_results[alpha] = history
#         np.save(f'probe_adagrad_a{alpha}.npy', np.array(history))

#-----Probe: Adam learning rate grid (15 epochs)-----------------------------------------------
    # adam_grid = [0.003]
    # adam_results = {}
 
    # for alpha in adam_grid:
    #     print(f"\n--- Adam, alpha={alpha} ---")
    #     history = MNIST_MLP(
    #         lambda w, a=alpha: Adam(a, w, decay_1, decay_2, ζ),
    #         epochs, inputs, labels
    #     )
    #     adam_results[alpha] = history
    #     np.save(f'probe_adam_a{alpha}.npy', np.array(history))
 
# #-----Probe summary-------------------------------------------------------------------------------
#     print("\n===== Probe summary (final 15-epoch loss) =====")
#     for alpha, history in sgd_results.items():
#         print(f"SGD+Nesterov alpha={alpha}: {history[-1]:.4f}")
#     for alpha, history in adagrad_results.items():
#         print(f"AdaGrad      alpha={alpha}: {history[-1]:.4f}")
    # for alpha, history in adam_results.items():
    #     print(f"Adam      alpha={alpha}: {history[-1]:.4f}")
 
#     plot_results(
#         [(h, f'SGD+Nesterov α={a}') for a, h in sgd_results.items()],
#         'SGD+Nesterov LR probe (15 epochs)',
#         'epoch',
#         'training loss'
#     )
 
#     plot_results(
#         [(h, f'AdaGrad α={a}') for a, h in adagrad_results.items()],
#         'AdaGrad LR probe (15 epochs)',
#         'epoch',
#         'training loss'
#     )

#-----Experiment: Bias-Correction Term-----------------------------------------------------------


#-----Experiment: Convolutional Neural Networks--------------------------------------------------


#-----Experiment: Multi-layer Perceptron (MLP)---------------------------------------------------
    MNIST_MLP_Adam = MNIST_MLP(
        lambda w: Adam(stepsize, w, decay_1, decay_2, ζ),
        epochs, inputs, labels
    )
    np.save('history_mlp_adam.npy', np.array(MNIST_MLP_Adam))

    MNIST_MLP_AdaGrad = MNIST_MLP(
        lambda w: AdaGrad(adagrad_alpha_best, w, ζ),
        epochs, inputs, labels
    )
    np.save('history_mlp_adam.npy', np.array(MNIST_MLP_AdaGrad))

    MNIST_MLP_SGD_Nesterov = MNIST_MLP(
        lambda w: SGD_Nesterov(sgd_alpha_best, w, decay_1),
        epochs, inputs, labels
    )
    np.save('history_mlp_adam.npy', np.array(MNIST_MLP_SGD_Nesterov))

    plot_results(
        [
            (MNIST_MLP_Adam,         'Adam'),
            # (MNIST_MLP_AdaGrad,      'AdaGrad'),
            # (MNIST_MLP_SGD_Nesterov, 'SGD + Nesterov'),
        ],
        'MNIST MLP + Dropout — Figure 2(a) Recreation',
        'epoch',
        'training loss'
    )

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