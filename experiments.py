from optimizers import Adam, AdaGrad, SGD_Nesterov
from train import test_MNIST_logRegression, plot_results

# simple_experiment = [
#     (test_simple(0.001, 0.9, 0.999), 'baseline (α=0.001, β₂=0.999)'),
#     (test_simple(0.01,  0.9, 0.999), 'large lr (α=0.01,  β₂=0.999)'),
#     (test_simple(0.001, 0.9, 0.99),  'small β₂ (α=0.001, β₂=0.99)'),
# ]
# plot_results(simple_experiment, 'Adam convergence on f(x) = x²', 'iteration', 'weights')

epochs = 45
ζ = 1e-8

MNIST_log_regression_Adam = test_MNIST_logRegression(
    lambda w: Adam(0.001, w, 0.9, 0.999, ζ),
    lambda b: Adam(0.001, b, 0.9, 0.999, ζ),
    epochs
)

MNIST_log_regression_AdaGrad = test_MNIST_logRegression(
    lambda w: AdaGrad(0.001, w, ζ),
    lambda b: AdaGrad(0.001, b, ζ),
    epochs
)

MNIST_log_regression_SGD_Nesterov = test_MNIST_logRegression(
    lambda w: SGD_Nesterov(0.001, w, decay=0.9),
    lambda b: SGD_Nesterov(0.001, b, decay=0.9),
    epochs
)

plot_results(
    [
        (MNIST_log_regression_Adam,         'Adam'),
        (MNIST_log_regression_AdaGrad,      'AdaGrad'),
        (MNIST_log_regression_SGD_Nesterov, 'SGD + Nesterov'),
    ],
    'MNIST Logistic Regression — Figure 1 Recreation',
    'epoch',
    'training loss'
)