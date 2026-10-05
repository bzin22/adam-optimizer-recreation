import numpy as np

from optimizers import Adam, AdaGrad, SGD_Nesterov, AdaDelta, RMSProp, AdaMax
from train import train, load_MNIST, load_CIFAR10
from utils import one_hot, pack, unpack, plot_results
from logreg import log_reg_backward
from mlp import reseed, mlp_backward
from vae import init_vae, vae_backward, reseed as vae_reseed
from cnn import build_cnn, cnn_params, make_cnn_backward, pick_device, N_PARAMS

def CIFAR_CNN(optimizer_fn, epochs, inputs, labels, dropout=True, seed=0, record_batches=False):
    model = build_cnn(dropout=dropout, seed=seed)
    device = pick_device()
    params = cnn_params(model)

    theta = pack(params)
    optimizer = optimizer_fn(theta)

    backward = make_cnn_backward(model, device)

    return train(backward, params, optimizer, inputs, labels, epochs,
                 record_batches=record_batches)

def gradcheck(eps, dropout=True):
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
    loss, grads = mlp_backward(params, X, Y, lam=0.001, dropout=dropout)
    g_analytical = pack(grads)
    g_numerical = np.zeros_like(theta)

    for i in range(theta.shape[0]):
        theta_plus = theta.copy()
        theta_plus[i] += eps
        theta_minus = theta.copy()
        theta_minus[i] -= eps
        reseed(0)

        loss_plus, grad_plus = mlp_backward(unpack(theta_plus, shapes), X, Y, lam=0.001, dropout=dropout)
        reseed(0)
        loss_minus, grad_minus = mlp_backward(unpack(theta_minus, shapes), X, Y, lam=0.001, dropout=dropout)

        g_numerical[i] = (loss_plus-loss_minus) / (2*eps)

    rel_error = np.abs(g_analytical - g_numerical) / np.maximum(
        np.abs(g_analytical) + np.abs(g_numerical), 1e-12
    )
    print(f"gradcheck (mlp, dropout={dropout}) max relative error: {rel_error.max():.2e} at index {rel_error.argmax()}")
    print(f"gradcheck (mlp, dropout={dropout}) mean relative error: {rel_error.mean():.2e}")
    assert rel_error.max() < 1e-5, "gradcheck failed"

def gradcheck_vae(eps):
    """
    Central-difference check of vae_backward on a tiny 20 -> 8 -> 4 latent VAE.
    The reparameterization noise is reseeded before every evaluation so the plus
    and minus passes see identical eps, the same trick gradcheck() uses for the
    MLP's dropout masks.
    """
    rng = np.random.default_rng(seed=0)

    params = init_vae(input_dim=20, hidden=8, latent=4, seed=1)
    shapes = [p.shape for p in params]
    theta = pack(params)

    X = rng.random((6, 20))   # pixels in [0,1], as the Bernoulli likelihood expects

    vae_reseed(0)
    loss, grads = vae_backward(params, X)
    g_analytical = pack(grads)
    g_numerical = np.zeros_like(theta)

    for i in range(theta.shape[0]):
        theta_plus = theta.copy()
        theta_plus[i] += eps
        theta_minus = theta.copy()
        theta_minus[i] -= eps

        vae_reseed(0)
        loss_plus, _ = vae_backward(unpack(theta_plus, shapes), X)
        vae_reseed(0)
        loss_minus, _ = vae_backward(unpack(theta_minus, shapes), X)

        g_numerical[i] = (loss_plus - loss_minus) / (2 * eps)

    rel_error = np.abs(g_analytical - g_numerical) / np.maximum(
        np.abs(g_analytical) + np.abs(g_numerical), 1e-12
    )
    print(f"gradcheck (vae) max relative error: {rel_error.max():.2e} at index {rel_error.argmax()}")
    print(f"gradcheck (vae) mean relative error: {rel_error.mean():.2e}")
    assert rel_error.max() < 1e-5, "vae gradcheck failed"

def check_adamax():
    """
    AdaMax against values that fall straight out of Algorithm 2.

    At t=1, m_1/(1-β1) = g and u_1 = |g|, so the first step is α·g/|g|, i.e. exactly
    α in magnitude whatever the gradient is. That is the property the infinity norm
    buys, so it is the one worth pinning.
    """
    decay_1, decay_2 = 0.9, 0.999

    weights = np.array([3.0, -2.0])
    optimizer = AdaMax(0.002, weights, decay_1, decay_2)
    before = weights.copy()
    optimizer.update(np.array([0.7, -4.0]))
    step = np.abs(optimizer.weights - before)
    print(f"AdaMax first step: {step} (expected 0.002 in every coordinate)")
    assert np.allclose(step, 0.002, atol=1e-7), "first step is not α in magnitude"

    # A coordinate whose gradient is exactly 0 at t=1 gives u_t = 0 and m_t = 0.
    # Read literally, Algorithm 2 divides 0/0 there and that weight is NaN for the
    # rest of the run. Dead ReLUs hit this on the first minibatch.
    optimizer = AdaMax(0.002, np.array([1.0, 2.0, 3.0]), decay_1, decay_2)
    with np.errstate(all='raise'):
        optimizer.update(np.array([0.0, 0.0, 0.5]))
    print(f"AdaMax after a zero gradient: {optimizer.weights}")
    assert np.all(np.isfinite(optimizer.weights)), "zero gradient produced NaN"
    assert optimizer.weights[0] == 1.0, "zero-gradient coordinate moved"

    # f(w) = 0.5·||w||², so grad = w and the minimum is the origin.
    optimizer = AdaMax(0.002, np.array([5.0, -5.0]), decay_1, decay_2)
    for _ in range(5000):
        optimizer.update(optimizer.weights.copy())
    print(f"AdaMax on 0.5·||w||²: max |w| = {np.abs(optimizer.weights).max():.2e}")
    assert np.all(np.abs(optimizer.weights) < 1e-3), "did not reach the minimum"

def check_bias_correction():
    """
    The Adam(bias_correction=False) switch that Figure 4 measures.

    On step 1 the corrected update is α·g/(|g|+ζ) ≈ α. Drop the correction terms and
    it becomes α·(1-β1)·g / (sqrt((1-β2)·g²)+ζ), so the uncorrected step is larger by
    (1-β1)/sqrt(1-β2) = 3.1623 at the default decays. That factor is the overshoot
    Figure 4 is about.
    """
    decay_1, decay_2, ζ = 0.9, 0.999, 1e-8
    grad = np.array([0.7])

    corrected = Adam(0.001, np.array([0.0]), decay_1, decay_2, ζ)
    corrected.update(grad.copy())
    uncorrected = Adam(0.001, np.array([0.0]), decay_1, decay_2, ζ, bias_correction=False)
    uncorrected.update(grad.copy())

    ratio = abs(uncorrected.weights[0]) / abs(corrected.weights[0])
    expected = (1 - decay_1) / np.sqrt(1 - decay_2)
    print(f"step 1 corrected: {corrected.weights[0]:.3e}, uncorrected: {uncorrected.weights[0]:.3e}")
    print(f"ratio: {ratio:.6f}, expected (1-β1)/sqrt(1-β2): {expected:.6f}")
    assert abs(ratio - expected) < 1e-4, "uncorrected overshoot is the wrong size"

    # The new keyword must not have disturbed the Adam that produced Figures 1 and 2(a).
    optimizer = Adam(0.001, np.array([1.0, 2.0, 3.0]), decay_1, decay_2, ζ)
    for i in range(10):
        optimizer.update(np.array([0.1, -0.2, 0.3]) * (i + 1))
    expected_weights = np.array([0.99015411, 2.00984589, 2.99015411])
    print(f"default Adam after 10 steps: {optimizer.weights}")
    assert np.allclose(optimizer.weights, expected_weights), "default Adam path changed"

def check_cnn_bridge():
    """
    The CNN is the one model whose gradients come from torch rather than by hand, so
    the thing to check is that the NumPy optimizer is still what moves it. After one
    update the module's weights must equal the optimizer's packed vector.
    """
    decay_1, decay_2, ζ = 0.9, 0.999, 1e-8
    rng = np.random.default_rng(0)
    inputs = rng.standard_normal((128, 3, 32, 32)).astype(np.float32)
    labels = one_hot(10, rng.integers(0, 10, size=128))

    model = build_cnn(dropout=False, seed=0)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"CNN parameters: {n_params} (paper's architecture on 32x32 gives {N_PARAMS})")
    assert n_params == N_PARAMS, "architecture does not match the paper"

    params = cnn_params(model)
    backward = make_cnn_backward(model, pick_device())

    loss, grads = backward(params, inputs, labels, 0.0)
    grad_vec = pack(grads)
    print(f"initial loss: {loss:.4f} (ln 10 = {np.log(10):.4f} for a 10-way uniform output)")
    assert abs(loss - np.log(10)) < 0.2, "untrained loss is not near ln 10"
    assert grad_vec.size == N_PARAMS, "gradient vector is the wrong length"
    assert np.all(np.isfinite(grad_vec)), "gradients are not finite"

    optimizer = Adam(1e-3, pack(params), decay_1, decay_2, ζ)
    optimizer.update(grad_vec)
    backward(unpack(optimizer.weights, [p.shape for p in params]), inputs, labels, 0.0)
    live = np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()])
    drift = np.abs(live - optimizer.weights).max()
    print(f"max |module weights - optimizer vector|: {drift:.2e}")
    assert drift < 1e-6, "the torch module and the NumPy optimizer have diverged"

def check_figure_2a(inputs, labels):
    """
    Adding the dropout flag to mlp_forward/mlp_backward must not have moved Figure
    2(a). Re-runs 5 epochs of Adam and compares against the saved 200-epoch history.
    """
    history = MNIST_MLP(
        lambda w: Adam(3e-4, w, 0.9, 0.999, 1e-8),
        5, inputs, labels
    )
    stored = np.load('history_mlp_adam.npy')[:5]
    diff = np.abs(np.array(history) - stored).max()
    print(f"re-run:  {np.array(history)}")
    print(f"stored:  {stored}")
    print(f"max abs difference: {diff:.3e}")
    assert diff == 0.0, "the dropout refactor changed Figure 2(a)"

def MNIST_VAE(optimizer_fn, epochs, inputs, seed=0):
    vae_reseed(seed)
    params = init_vae(seed=seed)

    theta = pack(params)
    optimizer = optimizer_fn(theta)

    # train() slices labels[:, idx] every step and hands them to the backward fn.
    # A VAE reconstructs its input, so pass a 1-row placeholder rather than a
    # second copy of the 784xN data.
    placeholder = np.zeros((1, inputs.shape[0]))

    return train(vae_backward, params, optimizer, inputs, placeholder, epochs)

def MNIST_MLP(optimizer_fn, epochs, inputs, labels, dropout=True):
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

    # train() calls backwards_fn positionally with 4 args, so the dropout flag
    # rides in through a closure.
    backward = lambda p, x, y, lam: mlp_backward(p, x, y, lam, dropout=dropout)

    return train(backward, params, optimizer, inputs, labels, epochs)

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
    # cifar_inputs, cifar_labels = load_CIFAR10()

    epochs = 200
    stepsize = 0.001 # default stepsize from original paper
    decay_1 = 0.9
    decay_2 = 0.999
    ζ = 1e-8
    sgd_alpha_best = 0.03
    adagrad_alpha_best = 0.01
    adam_alpha_best = 0.0003
    rms_alpha_best = 3e-4

    # Figure 2(b) re-probes every stepsize, because the four above were tuned with
    # dropout on and the deterministic cost is a different problem. 15-epoch final
    # losses, best in each row:
    #   Adam         1e-5 .0927  3e-5 .0278  1e-4 .0042  3e-4 .0081  1e-3 .0192  3e-3 .0424
    #   AdaMax       1e-4 .0324  2e-4 .0099  5e-4 .0013  1e-3 .0014  2e-3 .0034  5e-3 .0067
    #   AdaGrad      1e-3 .0370  3e-3 .0038  1e-2 .0007  3e-2 .0017
    #   SGD+Nesterov 3e-3 .0219  1e-2 .0021  3e-2 .0003  1e-1 .0085
    adam_alpha_best_no_dropout = 1e-4
    adamax_alpha_best_no_dropout = 5e-4
    adagrad_alpha_best_no_dropout = 0.01
    sgd_alpha_best_no_dropout = 0.03

    # Figure 3, selected on a 3-epoch probe, keyed by (optimizer, dropout). Adam at
    # 3e-3 diverged to exactly ln 10 = 2.3029 with and without dropout.
    cifar_alpha_best = {
        ('adam',         False): 1e-3,  ('adam',         True): 3e-4,
        ('adagrad',      False): 3e-3,  ('adagrad',      True): 3e-3,
        ('sgd_nesterov', False): 1e-2,  ('sgd_nesterov', True): 1e-2,
    }



# #------Probe: RMSProp learning rate grid (15 epochs)------------------------------------------
    # rmsprop_grid = [1e-4, 3e-4, 1e-3, 3e-3]
    # rms_results = {}
 
    # for alpha in rmsprop_grid:
    #     print(f"\n--- RMSProp, alpha={alpha} ---")
    #     history = MNIST_MLP(
    #         lambda w, a=alpha: RMSProp(a, w, decay_1, ζ),
    #         epochs, inputs, labels
    #     )
    #     rms_results[alpha] = history
    #     np.save(f'probe_rmsprop_a{alpha}.npy', np.array(history))

# #------Probe: SGD+Nesterov learning rate grid (15 epochs)------------------------------------------
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
#     adagrad_grid = [1e-3, 3e-3, 1e-2, 3e-2]
#     adagrad_results = {}
 
#     for alpha in adagrad_grid:
#         print(f"\n--- AdaGrad, alpha={alpha} ---")
#         history = MNIST_MLP(
#             lambda w, a=alpha: AdaGrad(a, w, ζ),
#             epochs, inputs, labels
#         )
#         adagrad_results[alpha] = history
#         np.save(f'probe_adagrad_a{alpha}.npy', np.array(history))

# #-----Probe: Adam learning rate grid (15 epochs)-----------------------------------------------
#     adam_grid = [1e-4, 3e-4, 1e-3, 3e-3]
#     adam_results = {}
 
#     for alpha in adam_grid:
#         print(f"\n--- Adam, alpha={alpha} ---")
#         history = MNIST_MLP(
#             lambda w, a=alpha: Adam(a, w, decay_1, decay_2, ζ),
#             epochs, inputs, labels
#         )
#         adam_results[alpha] = history
#         np.save(f'probe_adam_a{alpha}.npy', np.array(history))
 
# #-----Probe summary-------------------------------------------------------------------------------
    # print("\n===== Probe summary (final 15-epoch loss) =====")
    # for alpha, history in sgd_results.items():
    #     print(f"SGD+Nesterov alpha={alpha}: {history[-1]:.4f}")
    # for alpha, history in adagrad_results.items():
    #     print(f"AdaGrad      alpha={alpha}: {history[-1]:.4f}")
    # for alpha, history in adam_results.items():
    #     print(f"Adam      alpha={alpha}: {history[-1]:.4f}")
    # for alpha, history in rms_results.items():
    #     print(f"RMSProp alpha={alpha}: {history[-1]:.4f}")
    
    # plot_results(
    #     [(h, f'RMSProp α={a}') for a, h in rms_results.items()],
    #     'RMSProp LR probe (15 epochs)',
    #     'epoch',
    #     'training loss'
    # )

    # plot_results(
    #     [(h, f'SGD+Nesterov α={a}') for a, h in sgd_results.items()],
    #     'SGD+Nesterov LR probe (15 epochs)',
    #     'epoch',
    #     'training loss'
    # )
 
#     plot_results(
#         [(h, f'AdaGrad α={a}') for a, h in adagrad_results.items()],
#         'AdaGrad LR probe (15 epochs)',
#         'epoch',
#         'training loss'
#     )

#     plot_results(
#         [(h, f'Adam α={a}') for a, h in adam_results.items()],
#         'Adam LR probe (15 epochs)',
#         'epoch',
#         'training loss'
#     )

#-----Checks---------------------------------------------------------------------------------------

    # Gradients, against central differences.

    # gradcheck(1e-5)                   # MLP, dropout on   (Figure 2a path)
    # gradcheck(1e-5, dropout=False)    # MLP, dropout off  (Figure 2b path)
    # gradcheck_vae(1e-6)               # VAE               (Figure 4 path)

    # Optimizers, against values that fall out of the paper's algorithms.

    # check_adamax()
    # check_bias_correction()

    # The CNN's gradients come from torch, so check the NumPy optimizer still
    # drives the model. Needs about 30 seconds.

    # check_cnn_bridge()

    # Regression: the dropout flag must not have moved Figure 2(a).
    # Re-runs 5 epochs, so about 2 minutes.

    # check_figure_2a(inputs, labels)

#-----Experiment: Bias-Correction Term (Figure 4)------------------------------------------------

    # Effect of removing the /(1-β1^t) and /(1-β2^t) terms from Algorithm 1, measured
    # on a VAE (784 -> 500 softplus -> 50-d Gaussian latent -> 500 softplus -> 784
    # Bernoulli). Paper grid: β1 in {0, 0.9}, β2 in {0.99, 0.999, 0.9999},
    # log10(α) in [-5, -1], 10 epochs each, with and without bias correction.

    # gradcheck_vae(1e-6)

    # vae_inputs, _ = load_MNIST(standardize=False)   # Bernoulli likelihood wants [0,1]
    # fig4_results = []

    # for β1 in [0.0, 0.9]:
    #     for β2 in [0.99, 0.999, 0.9999]:
    #         for log_α in [-5, -4, -3, -2, -1]:
    #             for bias_correction in (True, False):
    #                 α = 10.0 ** log_α
    #                 history = MNIST_VAE(
    #                     lambda w: Adam(α, w, β1, β2, ζ, bias_correction=bias_correction),
    #                     10, vae_inputs
    #                 )
    #                 h = np.array(history)
    #                 finite = h[np.isfinite(h)]
    #                 fig4_results.append({
    #                     'beta1': β1, 'beta2': β2, 'log_alpha': log_α,
    #                     'bias_correction': bias_correction,
    #                     'best_loss': float(finite.min()) if finite.size else float('inf'),
    #                 })

#-----Experiment: Convolutional Neural Networks (Figure 3)---------------------------------------

    # CIFAR-10 ConvNet: c1 5x5x64, s1 3x3 max pool stride 2, c2 5x5x64, s2 pool,
    # c3 5x5x128, s3 pool, 1000 ReLU, 10-way output. 1,475,266 parameters.
    # This is the one model whose backward pass is not hand-coded: conv gradients
    # come from torch autograd, and the NumPy optimizers still do every update.
    # Six curves, each optimizer with and without dropout, 45 epochs, batch 128.

    # cifar_inputs, cifar_labels = load_CIFAR10(whiten=True)

    # for name, make in [('adam',         lambda a, w: Adam(a, w, decay_1, decay_2, ζ)),
    #                    ('adagrad',      lambda a, w: AdaGrad(a, w, ζ)),
    #                    ('sgd_nesterov', lambda a, w: SGD_Nesterov(a, w, decay_1))]:
    #     for use_dropout in (False, True):
    #         history = CIFAR_CNN(
    #             lambda w: make(cifar_alpha_best[(name, use_dropout)], w),
    #             45, cifar_inputs, cifar_labels, dropout=use_dropout
    #         )
    #         tag = name + ('_dropout' if use_dropout else '')
    #         np.save(f'history_cifar_{tag}.npy', np.array(history))

#-----Experiment: MLP, Deterministic Cost (Figure 2b)--------------------------------------------

    # Same MNIST MLP as Figure 2(a) with dropout switched off, which makes the cost
    # function deterministic. Adds AdaMax (Algorithm 2). The paper's panel also has
    # the sum-of-functions optimizer (Sohl-Dickstein et al. 2014); that is a separate
    # method and is not recreated here, so this panel has four curves, not five.
    # Stepsizes are re-probed because the Figure 2(a) grid was tuned with dropout on.

    # gradcheck(1e-5, dropout=False)

    # for name, make, alpha in [
    #     ('adam',         lambda a, w: Adam(a, w, decay_1, decay_2, ζ), adam_alpha_best_no_dropout),
    #     ('adamax',       lambda a, w: AdaMax(a, w, decay_1, decay_2),  adamax_alpha_best_no_dropout),
    #     ('adagrad',      lambda a, w: AdaGrad(a, w, ζ),                adagrad_alpha_best_no_dropout),
    #     ('sgd_nesterov', lambda a, w: SGD_Nesterov(a, w, decay_1),     sgd_alpha_best_no_dropout),
    # ]:
    #     history = MNIST_MLP(lambda w: make(alpha, w), epochs, inputs, labels, dropout=False)
    #     np.save(f'history_mlp_det_{name}.npy', np.array(history))

#-----Experiment: Multi-layer Perceptron (MLP)---------------------------------------------------
    
    gradcheck(1e-5)

    MNIST_MLP_Adam = MNIST_MLP(
        lambda w: Adam(adam_alpha_best, w, decay_1, decay_2, ζ),
        epochs, inputs, labels
    )
    np.save('history_mlp_adam.npy', np.array(MNIST_MLP_Adam))

    MNIST_MLP_AdaGrad = MNIST_MLP(
        lambda w: AdaGrad(adagrad_alpha_best, w, ζ),
        epochs, inputs, labels
    )
    np.save('history_mlp_adagrad.npy', np.array(MNIST_MLP_AdaGrad))

    MNIST_MLP_SGD_Nesterov = MNIST_MLP(
        lambda w: SGD_Nesterov(sgd_alpha_best, w, decay_1),
        epochs, inputs, labels
    )
    np.save('history_mlp_sgd.npy', np.array(MNIST_MLP_SGD_Nesterov))

    MNIST_MLP_RMSProp = MNIST_MLP(
        lambda w: RMSProp(rms_alpha_best, w, decay_1, ζ),
        epochs, inputs, labels
    )
    np.save('history_mlp_rms.npy', np.array(MNIST_MLP_RMSProp))

    MNIST_MLP_AdaDelta = MNIST_MLP(
        lambda w: AdaDelta(w, 0.95, 1e-6, 1.0),
        epochs, inputs, labels
    )
    np.save('history_mlp_adadelta.npy', np.array(MNIST_MLP_AdaDelta))

    plot_results(
        [
            (MNIST_MLP_Adam,         'Adam'),
            (MNIST_MLP_AdaGrad,      'AdaGrad'),
            (MNIST_MLP_SGD_Nesterov, 'SGD + Nesterov'),
            (MNIST_MLP_AdaDelta, 'AdaDelta'),
            (MNIST_MLP_RMSProp, 'RMSProp')
        ],
        'MNIST MLP + Dropout — Figure 2(a) Recreation',
        'iterations over entire dataset',
        'training cost'
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