"""Experiment entry point. Enable the commented run blocks at the bottom.

Importing this module or running it with all blocks commented starts no training.
Model/optimizer math stays in its own modules; run settings and plotting live here.
"""
from pathlib import Path
from decimal import Decimal
import hashlib
import json
import subprocess
import sys

import numpy as np

from optimizers import Adam, AdaGrad, SGD_Nesterov, AdaDelta, RMSProp, AdaMax
from train import train, load_MNIST, load_CIFAR10
from utils import one_hot, pack, unpack, plot_results
from logreg import log_reg_backward
from mlp import reseed, mlp_backward
from vae import init_vae, vae_backward, reseed as vae_reseed
from cnn import build_cnn, cnn_params, make_cnn_backward, pick_device, N_PARAMS

def CIFAR_CNN(optimizer_fn, epochs, inputs, labels, dropout=True, seed=0, record_batches=False, device=None):
    model = build_cnn(dropout=dropout, seed=seed)
    device = pick_device() if device is None else device
    params = cnn_params(model)

    theta = pack(params)
    optimizer = optimizer_fn(theta)

    backward = make_cnn_backward(model, device)

    return train(backward, params, optimizer, inputs, labels, epochs,
                 seed=seed, record_batches=record_batches)

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
    stored = np.load(FIGURE2A_RUN / 'baseline/history_mlp_adam.npy')[:5]
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

    return train(vae_backward, params, optimizer, inputs, placeholder, epochs, seed=seed)

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


#-----Shared experiment paths and output helpers------------------------------------------------

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
RESULTS = ROOT / "results"
FIGURE2A_RUN = RESULTS / "figure2a_search_20261002_v1"


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def new_run(path, configuration):
    """Never overwrite saved experiment evidence."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    save_json(path / "configuration.json", configuration)
    return path


def plot_histories(series, title, output):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for history, label in series:
        ax.plot(np.arange(1, len(history) + 1), history, label=label)
    ax.set(yscale="log", xlabel="iterations over entire dataset",
           ylabel="training loss", title=title)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


#-----Experiment: Logistic Regression (Figure 1; historical baseline)---------------------------


def run_figure1(run_dir, epochs=200, stepsize=0.001):
    """Existing baseline setup, not a claim of exact paper reproduction.

    The paper's L2 penalty and epoch stepsize decay are not implemented by this
    historical block. The original figure's numerical histories were not saved
    in this repository. A new run is saved separately and cannot replace it.
    """
    out = new_run(run_dir, {"figure": "1_baseline", "epochs": epochs,
                           "stepsize": stepsize, "seed": 0,
                           "limitations": "No L2 or epoch stepsize decay"})
    inputs, labels = load_MNIST()
    curves = []
    for name, make in [
        ("Adam", lambda w: Adam(stepsize, w, 0.9, 0.999, 1e-8)),
        ("AdaGrad", lambda w: AdaGrad(stepsize, w, 1e-8)),
        ("SGD + Nesterov", lambda w: SGD_Nesterov(stepsize, w, 0.9)),
    ]:
        history = MNIST_log_reg(make, epochs, inputs, labels)
        np.save(out / (name.lower().replace(" ", "_") + ".npy"), history)
        curves.append((history, name))
    plot_histories(curves, "MNIST logistic regression — baseline", out / "figure_1_baseline.png")


#-----Experiment: MLP with Dropout (Figure 2a; completed)----------------------------------------

# The grid and optimizer settings are here; the checkpoint/worker engine is a helper.
FIGURE2A_MULTIPLIERS = [0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 7, 10]
FIGURE2A_BASE_RATES = {"adam": 0.0003, "adagrad": 0.01}
FIGURE2A_SELECTED_RATES = {"adam": 0.00009, "adagrad": 0.01}


def figure2a_configs():
    order = [1] + [m for m in FIGURE2A_MULTIPLIERS if m != 1]
    return [{"id": f"{name}_lr{float(Decimal(str(base)) * Decimal(str(m))):g}",
             "optimizer": name,
             "learning_rate": float(Decimal(str(base)) * Decimal(str(m))),
             "multiplier": m, "epochs": 200, "seed": 0, "batch_size": 128}
            for m in order for name, base in FIGURE2A_BASE_RATES.items()]


def run_figure2a_search(run_dir):
    """Explicit new search only; frozen completed results never change."""
    import figure2a_search as engine
    out = Path(run_dir).resolve()
    engine.prepare(ROOT, out)
    subprocess.run([sys.executable, "-u", str(out / "source/figure2a_search.py"),
                    "supervise", "--run-dir", str(out)], check=True)


def plot_figure2a():
    """Replot completed search histories without training or re-finalizing the run."""
    import matplotlib.pyplot as plt
    records = json.loads((FIGURE2A_RUN / "search_results.json").read_text())
    manifest = json.loads((FIGURE2A_RUN / "manifest.json").read_text())
    labels = {"adam": "Adam", "adagrad": "AdaGrad", "sgd": "SGD + Nesterov",
              "adadelta": "AdaDelta", "rms": "RMSProp"}
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for name, label in labels.items():
        path = FIGURE2A_RUN / "chart_data" / f"history_mlp_{name}.npy"
        selected = records["selected"].get(name)
        expected = selected["history_sha256"] if selected else manifest["baseline"][name]["sha256"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Chart history changed: {name}")
        if selected:
            label += f" (α={selected['config']['learning_rate']:g})"
        ax.plot(np.load(path, allow_pickle=False), label=label)
    ax.set(yscale="log", xlabel="iterations over entire dataset", ylabel="training loss",
           title="MNIST MLP + Dropout — Figure 2(a) Recreation")
    ax.legend()
    ax.grid(True)
    fig.text(.5, .015, "Adam/AdaGrad: 200-epoch learning-rate search, seed 0. Other curves: original settings.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(ASSETS / "figure_2_recreation_tuned_v1.png", dpi=200)
    plt.close(fig)


#-----Experiment: MLP, Deterministic Cost (Figure 2b; incomplete comparison)----------------------


def run_figure2b(run_dir, epochs=200):
    """Existing dropout-free comparison. SFO and L2 are still missing.

    Retain these four configured runs in one place, but never label the output a
    faithful Figure 2(b) reproduction. Old short-probe loops are removed.
    """
    rates = {"adam": 1e-4, "adamax": 5e-4, "adagrad": 0.01, "sgd_nesterov": 0.03}
    out = new_run(run_dir, {"figure": "2b_partial", "epochs": epochs,
                           "rates": rates, "seed": 0, "missing": ["SFO", "L2"]})
    inputs, labels = load_MNIST()
    curves = []
    for name, make in [
        ("adam", lambda a, w: Adam(a, w, 0.9, 0.999, 1e-8)),
        ("adamax", lambda a, w: AdaMax(a, w, 0.9, 0.999)),
        ("adagrad", lambda a, w: AdaGrad(a, w, 1e-8)),
        ("sgd_nesterov", lambda a, w: SGD_Nesterov(a, w, 0.9)),
    ]:
        history = MNIST_MLP(lambda w: make(rates[name], w), epochs, inputs, labels, dropout=False)
        np.save(out / f"history_mlp_det_{name}.npy", history)
        curves.append((history, name))
    plot_histories(curves, "MNIST MLP without dropout — partial comparison", out / "figure_2b_partial.png")


#-----Experiment: Convolutional Neural Networks (Figure 3; saved chart completed)----------------

# These settings produce the currently selected curves. They are not a completed
# full-horizon hyperparameter search. Adam without dropout uses its saved follow-up.
CIFAR_SELECTED_RATES = {
    ("adam", False): 1e-4, ("adam", True): 3e-4,
    ("adagrad", False): 3e-3, ("adagrad", True): 3e-3,
    ("sgd_nesterov", False): 1e-2, ("sgd_nesterov", True): 1e-2,
}


FIGURE3_SEARCH_RUN = RESULTS / "figure3_search_20261005_v1"
FIGURE3_LR_GRID = {
    "adam": [0.00006, 0.0001, 0.0003, 0.0006],
    "sgd_nesterov": [0.003, 0.01, 0.03, 0.06],
    "adagrad": [0.001, 0.003, 0.01],
}
FIGURE3_EXTRA_MOMENTA = [0.8, 0.95]


def figure3_config(name, dropout, rate, momentum, stage):
    return {"id": f"{name}_d{int(dropout)}_lr{rate:g}_m{momentum}",
            "optimizer": name, "dropout": dropout, "learning_rate": rate,
            "momentum": momentum, "stage": stage, "epochs": 45,
            "seed": 0, "batch_size": 128, "batches_per_epoch": 391}


def figure3_stage1_configs():
    return [figure3_config(name, dropout, rate, None if name == "adagrad" else 0.9, 1)
            for name, rates in FIGURE3_LR_GRID.items() for dropout in [False, True] for rate in rates]


def figure3_stage2_configs(records):
    from figure3_search import rank
    configs = []
    for name in ["adam", "sgd_nesterov"]:
        for dropout in [False, True]:
            candidates = [r for r in records if r["status"] == "completed"
                          and r["config"]["optimizer"] == name and r["config"]["dropout"] == dropout]
            candidates = sorted(candidates, key=rank)
            if len(candidates) < 2:
                raise ValueError(f"Fewer than two finite stage-1 rates for {name}, dropout={dropout}")
            for record in candidates[:2]:
                for momentum in FIGURE3_EXTRA_MOMENTA:
                    configs.append(figure3_config(name, dropout, record["config"]["learning_rate"], momentum, 2))
    return configs


def run_figure3_search(run_dir=FIGURE3_SEARCH_RUN, prepare_only=False):
    """Approved two-stage, 38-trial GPU search. Completed runs are never retrained."""
    import figure3_search as engine
    out = Path(run_dir).resolve()
    if not out.exists():
        engine.prepare(ROOT, out)
    if prepare_only:
        return out
    if json.loads((out / "status.json").read_text())["state"] == "completed":
        return out
    subprocess.run([sys.executable, "-u", str(out / "source/experiments.py"),
                    "--figure3-supervise", "--run-dir", str(out)], check=True)
    return out


def plot_figure3_search(run_dir):
    """Plot selected full-search runs; retain all raw minibatch measurements."""
    import matplotlib.pyplot as plt
    out = Path(run_dir)
    result = json.loads((out / "search_results.json").read_text())
    colors = {"adam": "#2a78d6", "adagrad": "#eb6834", "sgd_nesterov": "#1baf7a"}
    labels = {"adam": "Adam", "adagrad": "AdaGrad", "sgd_nesterov": "SGD + Nesterov"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), facecolor="#fcfcfb")
    for name in colors:
        for dropout in [False, True]:
            key = name + ("_dropout" if dropout else "")
            record = result["selected"][key]
            config = record["config"]
            paths = [out / "chart_data" / key / n for n in ["history.npy", "batch_losses.npy"]]
            for path, expected in zip(paths, [record["history_sha256"], record["batch_sha256"]]):
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise ValueError(f"Selected chart data changed: {key}")
            history, batches = [np.load(path, allow_pickle=False) for path in paths]
            if history.shape != (45,) or batches.shape != (45 * 391,):
                raise ValueError("Expected 45 complete epochs and all minibatch losses")
            label = labels[name] + (" + dropout" if dropout else "")
            label += f" (α={config['learning_rate']:g}"
            if config["momentum"] is not None:
                label += f", {'β1' if name == 'adam' else 'μ'}={config['momentum']:g}"
            label += ")"
            style = dict(color=colors[name], linestyle="--" if dropout else "-", label=label)
            # Identical nonoverlapping blocks of 20 minibatches reduce display noise.
            # Raw losses remain saved. Positions are numbers of examples processed
            # before each batch, including the last short batch of each epoch.
            n = np.arange(3 * 391)
            positions = n // 391 + (n % 391) * 128 / 50000
            xs = [positions[i:i + 20].mean() for i in range(0, len(n), 20)]
            ys = [batches[i:min(i + 20, len(n))].mean() for i in range(0, len(n), 20)]
            axes[0].plot(xs, ys, linewidth=1.8, **style)
            axes[1].plot(np.arange(1, 46), history, linewidth=1.8, **style)
    axes[0].set(xlim=(0, 3), title="(a) first 3 epochs")
    axes[1].set(xlim=(0, 45), yscale="log", title="(b) 45 epochs")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")
        ax.set(xlabel="iterations over entire dataset", ylabel="training cost")
        ax.grid(True, alpha=.3)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    handles, legends = axes[1].get_legend_handles_labels()
    fig.legend(handles, legends, loc="upper center", bbox_to_anchor=(.5, .94), ncol=3, fontsize=8)
    fig.suptitle("CIFAR-10 ConvNet — Figure 3 recreation", fontsize=15)
    fig.text(.5, .035, "Seed 0; selection: mean loss at epochs 41–45. (a) Means of 20 minibatches; (b) epoch means.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .83))
    fig.savefig(out / "figure_3_recreation.png", dpi=150)
    plt.close(fig)


def run_figure3(run_dir, device="cpu"):
    """Explicit 45-epoch runs at current rates, recording within-epoch losses.

    The proposed rate/momentum search is documented separately, not launched here.
    CPU is explicit because the previous MPS full-search attempt failed.
    """
    out = new_run(run_dir, {"figure": "3", "epochs": 45, "seed": 0, "device": str(device),
                           "rates": {f"{n}_{d}": a for (n, d), a in CIFAR_SELECTED_RATES.items()}})
    inputs, labels = load_CIFAR10(whiten=True, cache=str(ROOT.parent / "cifar10_zca.npy"))
    rows = []
    for name, make in [
        ("adam", lambda a, w: Adam(a, w, 0.9, 0.999, 1e-8)),
        ("adagrad", lambda a, w: AdaGrad(a, w, 1e-8)),
        ("sgd_nesterov", lambda a, w: SGD_Nesterov(a, w, 0.9)),
    ]:
        for dropout in (False, True):
            rate = CIFAR_SELECTED_RATES[name, dropout]
            history, batches = CIFAR_CNN(lambda w: make(rate, w), 45, inputs, labels,
                                        dropout=dropout, record_batches=True, device=device)
            tag = name + ("_dropout" if dropout else "")
            np.save(out / f"history_cifar_{tag}.npy", history)
            np.save(out / f"batches_cifar_{tag}.npy", batches)
            rows.append((name, dropout, rate, np.asarray(history), batches))
    plot_figure3_batches(rows, out / "figure_3.png", len(inputs), batch_size=128)


def plot_figure3_batches(rows, output, n_examples, batch_size):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    colors = {"adam": "#2a78d6", "adagrad": "#eb6834", "sgd_nesterov": "#1baf7a"}
    per_epoch = (n_examples + batch_size - 1) // batch_size
    for name, dropout, rate, history, batches in rows:
        label = f"{name}{' + dropout' if dropout else ''} (α={rate:g})"
        style = {"color": colors[name], "linestyle": "--" if dropout else "-", "label": label}
        # Each saved loss is measured before its minibatch update; position it at
        # the number of training examples already processed, including the short last batch.
        positions = (np.arange(len(batches)) // per_epoch +
                     (np.arange(len(batches)) % per_epoch) * batch_size / n_examples)
        axes[0].plot(positions[:3 * per_epoch], batches[:3 * per_epoch], **style)
        axes[1].plot(np.arange(1, len(history) + 1), history, **style)
    axes[0].set(xlim=(0, 3), title="(a) first 3 epochs — minibatch losses")
    axes[1].set(yscale="log", title="(b) 45 epochs — epoch averages")
    for ax in axes:
        ax.set(xlabel="iterations over entire dataset", ylabel="training loss")
        ax.grid(True, alpha=.3)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .85))
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_figure3():
    if (FIGURE3_SEARCH_RUN / "status.json").exists():
        state = json.loads((FIGURE3_SEARCH_RUN / "status.json").read_text())["state"]
        if state == "completed":
            plot_figure3_search(FIGURE3_SEARCH_RUN)
            (ASSETS / "figure_3_recreation.png").write_bytes((FIGURE3_SEARCH_RUN / "figure_3_recreation.png").read_bytes())
            return
    import matplotlib.pyplot as plt
    root = ROOT
    run = root / "results/figure3_saved_update_20261005"
    manifest = json.loads((run / "manifest.json").read_text())
    histories = {}
    for name, record in manifest["series"].items():
        path = run / "chart_data" / f"history_cifar_{name}.npy"
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"History changed: {name}")
        h = np.load(path, allow_pickle=False)
        if h.shape != (45,) or not np.isfinite(h).all() or not (h > 0).all():
            raise ValueError(f"Expected 45 finite positive losses: {name}")
        histories[name] = h

    colors = {"adam": "#2a78d6", "adagrad": "#eb6834", "sgd_nesterov": "#1baf7a"}
    labels = {"adam": "Adam", "adagrad": "AdaGrad", "sgd_nesterov": "SGD + Nesterov"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8), facecolor="#fcfcfb")
    for ax, limit, title in zip(axes, [3, 45], ["(a) first 3 epochs — epoch averages", "(b) 45 epochs"]):
        ax.set_facecolor("#fcfcfb")
        for opt in colors:
            for dropout in (False, True):
                name = opt + ("_dropout" if dropout else "")
                label = labels[opt] + (" + dropout" if dropout else "")
                if opt == "adam":
                    label += f" (α={manifest['series'][name]['learning_rate']:g})"
                ax.plot(np.arange(1, limit + 1), histories[name][:limit],
                        color=colors[opt], linewidth=2,
                        linestyle="--" if dropout else "-", label=label,
                        marker="o" if limit == 3 else None, markersize=4)
        ax.set(xlabel="iterations over entire dataset", ylabel="training cost", title=title)
        if limit == 3:
            ax.set_xticks([1, 2, 3])
        else:
            ax.set_yscale("log")
        ax.grid(True, color="#b8b7b0", linewidth=0.6, alpha=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#b8b7b0")
        ax.tick_params(colors="#52514e", labelsize=9)
    handles, legend_labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, fontsize=9,
               loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=3)
    fig.suptitle("CIFAR-10 ConvNet — Figure 3 recreation", fontsize=15)
    fig.text(0.5, 0.025, "Adam without dropout: saved α=0.0001 run. Other curves unchanged. Panel (a) uses epoch averages.",
             ha="center", fontsize=9, color="#52514e")
    fig.tight_layout(rect=(0, 0.06, 1, 0.84))
    output = root / "assets/figure_3_recreation.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(output)




#-----Benchmark: current CPU/GPU Figure 3 runtime (no search)-------------------------------------


def benchmark_figure3(run_dir, device="cpu", threads=4, screen=False):
    """Bounded timing pilot, using existing whitened data and the real NumPy bridge.

    Screen: Adam+dropout, four warmup batches and 24 measured batches.
    Full: each of six configurations for one full epoch; Adam+dropout gets a
    second epoch to assess warmed-up timing. These histories are not search trials.
    """
    import os
    import platform
    import time
    import traceback
    import torch
    from torchvision.datasets import CIFAR10

    torch.set_num_threads(threads)
    if torch.get_num_threads() != threads:
        raise RuntimeError("Thread count did not change; run each benchmark in a fresh process")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("The MPS GPU is not available in this process")
    out = new_run(run_dir, {"purpose": "runtime benchmark only", "device": device,
                           "threads": torch.get_num_threads(), "screen": screen, "seed": 0,
                           "batch_size": 128, "torch": torch.__version__,
                           "numpy": np.__version__, "python": platform.python_version(),
                           "interop_threads": torch.get_num_interop_threads(),
                           "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
                           "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                              for name in ["experiments.py", "cnn.py", "optimizers.py", "train.py"]}})
    started = time.perf_counter()
    inputs = np.load(ROOT.parent / "cifar10_zca.npy", allow_pickle=False)
    dataset = CIFAR10(root=str(ROOT.parent / "data"), train=True, download=False)
    labels = one_hot(10, np.asarray(dataset.targets))
    if inputs.shape != (50000, 3, 32, 32) or labels.shape != (10, 50000):
        raise ValueError("Unexpected CIFAR-10 data dimensions")
    print(f"BENCHMARK {device}: loaded data in {time.perf_counter() - started:.2f}s", flush=True)
    rows = []
    configs = [(n, d) for n in ["adam", "sgd_nesterov", "adagrad"] for d in [False, True]]
    if screen:
        configs = [("adam", True)]
    try:
        for name, dropout in configs:
            model = build_cnn(dropout=dropout, seed=0)
            params = cnn_params(model)
            shapes = [p.shape for p in params]
            weights = pack(params)
            rate = CIFAR_SELECTED_RATES[name, dropout]
            if name == "adam":
                optimizer = Adam(rate, weights, 0.9, 0.999, 1e-8)
            elif name == "adagrad":
                optimizer = AdaGrad(rate, weights, 1e-8)
            else:
                optimizer = SGD_Nesterov(rate, weights, 0.9)
            backward = make_cnn_backward(model, device)
            shuffle = np.random.default_rng(0)

            def batch(indices):
                loss, grads = backward(unpack(optimizer.weights, shapes), inputs[indices], labels[:, indices], 0.0)
                optimizer.update(pack(grads))
                return loss

            if screen:
                for i in range(4):
                    batch(np.arange(i * 128, (i + 1) * 128))
            epochs = 1 if screen or (name, dropout) != ("adam", True) else 2
            for epoch in range(epochs):
                if device == "mps":
                    torch.mps.synchronize()
                epoch_start = time.perf_counter()
                indices = shuffle.permutation(len(inputs))
                limit = 24 * 128 if screen else len(inputs)
                losses, batch_seconds = [], []
                for i in range(0, limit, 128):
                    start = time.perf_counter()
                    losses.append(batch(indices[i:i + 128]))
                    # Gradients are copied back to CPU every batch by the existing
                    # bridge, so normal operation already waits for GPU work.
                    batch_seconds.append(time.perf_counter() - start)
                    if len(losses) % 100 == 0:
                        print(f"PROGRESS {device} {name} dropout={dropout} epoch={epoch+1} batches={len(losses)}", flush=True)
                if device == "mps":
                    torch.mps.synchronize()
                duration = time.perf_counter() - epoch_start
                if not np.isfinite(losses).all() or not np.isfinite(optimizer.weights).all():
                    raise FloatingPointError("Nonfinite benchmark loss or weights")
                row = {"optimizer": name, "dropout": dropout, "epoch": epoch + 1,
                       "batches": len(losses), "seconds": duration,
                       "mean_loss": float(np.mean(losses)),
                       "steady_batch_median_seconds": float(np.median(batch_seconds[5:])),
                       "batch_seconds": batch_seconds}
                rows.append(row)
                save_json(out / "timings.json", rows)
                print("TIMING " + json.dumps({k: v for k, v in row.items() if k != "batch_seconds"}), flush=True)
            del backward, model, params, optimizer, weights
        save_json(out / "status.json", {"state": "completed", "seconds": time.perf_counter() - started})
    except BaseException as error:
        save_json(out / "status.json", {"state": "failed", "reason": str(error),
                                      "traceback": traceback.format_exc()})
        raise
    return rows


#-----Experiment: Bias-Correction Term (Figure 4; incomplete)------------------------------------


def run_figure4(run_dir, epochs=100):
    """Full beta/rate grid with losses at epochs 10 and 100.

    Removes the obsolete best-finite-loss rule, which hid later divergence.
    Existing partial 10-epoch files are not reused as completed 100-epoch runs.
    """
    if epochs != 100:
        raise ValueError("Figure 4 requires both the 10-epoch and 100-epoch endpoints")
    configs = [(b1, b2, la, bc) for b1 in [0.0, 0.9] for b2 in [0.99, 0.999, 0.9999]
               for la in [-5, -4, -3, -2, -1] for bc in [True, False]]
    out = new_run(run_dir, {"figure": "4", "epochs": epochs, "seed": 0,
                           "selection": "loss at epochs 10 and 100; divergence explicitly marked"})
    inputs, _ = load_MNIST(standardize=False)
    rows = []
    for b1, b2, la, bc in configs:
        row = {"beta1": b1, "beta2": b2, "log_alpha": la, "bias_correction": bc}
        # Keep the completed prefix so a failure after epoch 10 does not erase
        # the valid ten-epoch endpoint. Save all nonfinite outcomes explicitly.
        vae_reseed(0)
        params = init_vae(seed=0)
        shapes = [p.shape for p in params]
        optimizer = Adam(10.0 ** la, pack(params), b1, b2, 1e-8, bias_correction=bc)
        shuffle = np.random.default_rng(0)
        history = []
        try:
            for epoch in range(epochs):
                losses = []
                order = shuffle.permutation(len(inputs))
                for start in range(0, len(inputs), 128):
                    with np.errstate(over="raise", invalid="raise", divide="raise"):
                        loss, grads = vae_backward(unpack(optimizer.weights, shapes), inputs[order[start:start + 128]])
                        if not np.isfinite(loss) or not all(np.isfinite(g).all() for g in grads):
                            raise FloatingPointError("Nonfinite loss or gradient")
                        optimizer.update(pack(grads))
                        if not np.isfinite(optimizer.weights).all():
                            raise FloatingPointError("Nonfinite parameters")
                    losses.append(loss)
                history.append(float(np.mean(losses)))
                print(f"Figure 4 {row}: epoch {epoch + 1}, loss={history[-1]:.5f}", flush=True)
            row["status"] = "completed"
        except FloatingPointError as error:
            row.update(status="failed", reason=str(error), failed_epoch=len(history) + 1)
        row.update(history=history, loss10=history[9] if len(history) >= 10 else None,
                   loss100=history[99] if len(history) >= 100 else None)
        rows.append(row)
        save_json(out / "results.json", rows)
    plot_figure4(out)


def plot_figure4(run_dir):
    """Require a full grid of completed or explicitly failed configurations."""
    import matplotlib.pyplot as plt
    out = Path(run_dir)
    rows = json.loads((out / "results.json").read_text())
    expected = {(b1, b2, la, bc) for b1 in [0.0, 0.9] for b2 in [0.99, 0.999, 0.9999]
                for la in [-5, -4, -3, -2, -1] for bc in [True, False]}
    keys = {(r["beta1"], r["beta2"], r["log_alpha"], r["bias_correction"]) for r in rows}
    if len(rows) != 60 or keys != expected or any(r["status"] not in ["completed", "failed"] for r in rows):
        raise ValueError("Refusing to chart an incomplete or duplicate Figure 4 grid")
    fig, axes = plt.subplots(2, 6, figsize=(20, 7), sharey=True)
    for block, endpoint in enumerate([10, 100]):
        for i, b1 in enumerate([0.0, 0.9]):
            for j, b2 in enumerate([0.99, 0.999, 0.9999]):
                ax = axes[i, block * 3 + j]
                for bc, color in [(False, "#eb6834"), (True, "#2a78d6")]:
                    group = sorted([r for r in rows if r["beta1"] == b1 and r["beta2"] == b2
                                    and r["bias_correction"] == bc], key=lambda r: r["log_alpha"])
                    x = [r["log_alpha"] for r in group]
                    y = [r[f"loss{endpoint}"] if r[f"loss{endpoint}"] is not None else np.nan for r in group]
                    ax.plot(x, y, "o-", color=color, label="with correction" if bc else "without correction")
                    for r in group:
                        if r[f"loss{endpoint}"] is None:
                            ax.plot(r["log_alpha"], .97 if bc else .90, "x", color=color,
                                    transform=ax.get_xaxis_transform())
                ax.set(yscale="log", xlabel="log10(α)", title=f"{endpoint} epochs; β1={b1}, β2={b2}")
                ax.grid(True, alpha=.3)
    axes[0, 0].set_ylabel("training loss")
    axes[1, 0].set_ylabel("training loss")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("VAE bias correction — losses at 10 and 100 epochs; × marks divergence")
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(out / "figure_4.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    # Internal worker/supervisor commands used by the managed, frozen GPU job.
    if len(sys.argv) > 1:
        import argparse
        import figure3_search as engine
        parser = argparse.ArgumentParser()
        commands = parser.add_mutually_exclusive_group(required=True)
        commands.add_argument("--figure3-worker", action="store_true")
        commands.add_argument("--figure3-supervise", action="store_true")
        parser.add_argument("--run-dir", type=Path, required=True)
        args = parser.parse_args()
        if args.figure3_worker:
            engine.run(args.run_dir)
            raise SystemExit(0)
        raise SystemExit(engine.supervise(args.run_dir))

    # Completed blocks are commented out. Uncomment only the run you want.
    # No dataset is loaded and no training starts while these remain commented.

#-----Checks-----------------------------------------------------------------------------------
    # gradcheck(1e-5)
    # gradcheck(1e-5, dropout=False)
    # gradcheck_vae(1e-6)
    # check_adamax()
    # check_bias_correction()
    # check_cnn_bridge()

#-----Experiment: Logistic Regression (Figure 1; historical chart exists)-----------------------
    # Original image retained. Exact original histories unavailable; this is only
    # the historical baseline setup, not a verified paper-faithful rerun.
    # run_figure1(RESULTS / "figure1_new_baseline")

#-----Experiment: MLP with Dropout (Figure 2a; completed — no rerun needed)-----------------------
    # 26/26 trials finished; Adam α=0.00009, AdaGrad α=0.01. Other curves unchanged.
    # plot_figure2a()
    # Only for an explicitly requested NEW search; requires a new output path:
    # run_figure2a_search(RESULTS / "figure2a_new_search")

#-----Experiment: MLP, Deterministic Cost (Figure 2b; incomplete — excluded from current work)----
    # Four-method extension; SFO and L2 still absent. Not a finished paper figure.
    # run_figure2b(RESULTS / "figure2b_new_partial")

#-----Experiment: Convolutional Neural Networks (Figure 3; current saved chart completed)--------
    # Replot the selected saved histories; no training:
    # plot_figure3()
    # New fixed-setting 45-epoch runs are available, but have NOT been requested:
    # run_figure3(RESULTS / "figure3_new_fixed_settings", device="cpu")
    # Approved reduced GPU search: 22 learning-rate trials + 16 momentum trials.
    # Run/resume only when the managed supervisor is not already running:
    # run_figure3_search(FIGURE3_SEARCH_RUN)
    # Saved-data plot selects the completed search automatically when available.

#-----Benchmark: Figure 3 runtime (bounded pilot only; no full search)----------------------------
    # benchmark_figure3(RESULTS / "figure3_cpu_benchmark", device="cpu", threads=4)
    # benchmark_figure3(RESULTS / "figure3_gpu_benchmark", device="mps", threads=4)

#-----Experiment: Bias-Correction Term (Figure 4; incomplete — not run by default)---------------
    # Previous partial 10-epoch/best-finite-loss experiment is obsolete.
    # This replacement uses the complete grid and endpoints at 10 and 100 epochs.
    # run_figure4(RESULTS / "figure4_new_full_grid")
    # plot_figure4(RESULTS / "figure4_new_full_grid")

    print("Experiments are disabled by default. Uncomment the desired block in experiments.py.")
