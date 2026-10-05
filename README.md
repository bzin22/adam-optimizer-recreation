# Adam Optimizer Recreation

NumPy recreation of **"Adam: A Method for Stochastic Optimization"** (Kingma & Ba, ICLR 2015).

## Overview

This project implements Adam, AdaGrad, and SGD with Nesterov momentum from scratch in NumPy and reproduces the experiments from the original paper. No PyTorch autograd — all forward passes, backward passes, and optimizer update rules are hand-coded.

The completed Figure 2(a) learning-rate search evaluates Adam and AdaGrad over
200 epochs per trial. Adam at learning rate **0.00009** achieves a mean training
loss of **0.01106** over the final 20 epochs, compared with **0.01360** for AdaGrad
at **0.01**. This reverses their ordering in the earlier 15-epoch-tuned recreation.

## Architecture

The codebase is organized so optimizers and the training loop are shared across
experiments; only the model changes per figure.

- `optimizers.py` — Adam, AdaGrad, SGD+Nesterov. Each operates on a single
  flattened parameter vector.
- `train.py` — model-agnostic training loop. Handles minibatching, seeded
  shuffling, per-epoch optimizer hooks, and pack/unpack at the optimizer boundary.
  Also contains `load_MNIST()`.
- `logreg.py` — logistic regression `forward` and `loss_and_grads`.
- `utils.py` — `one_hot`, `pack`/`unpack`, `plot_results`.
- `experiments.py` — thin runners (`run_figure1`, later `run_figure2`) that wire
  a model + optimizer into `train()`.

Regarding pack/unpack functions in utils.py: optimizers see one 1D vector regardless of the model. `pack([W, b, ...])` concatenates a list of arrays; `unpack(vector, shapes)` inverts it. This means
the same `Adam` instance handles logreg's 2 arrays (7,850 params) or the MLP's
6 arrays (1,796,010 params) with no code change.

## Experiments

### Figure 1: MNIST Logistic Regression

L2-regularized multi-class logistic regression on MNIST (784-dim image vectors, minibatch size 128). Adam's stepsize is annealed by 1/√t per epoch, matching the paper's Section 4 theoretical prediction.

![Figure 1 Recreation](assets/figure_1_recreation.png)

**Result:** Adam and SGD+Nesterov converge together and both outperform AdaGrad, consistent with the paper's findings.

### Figure 2: MNIST MLP with Dropout

This experiment recreates Figure 2(a) from the Adam paper: training a multilayer perceptron on MNIST with dropout stochastic regularization. The original paper uses a neural network with two fully connected hidden layers of 1000 ReLU units each and minibatch size 128. It compares Adam, AdaGrad, RMSProp, SGD with Nesterov momentum, and AdaDelta on training cost over 200 passes through the full dataset.

Implemented optimizers:

- Adam
- AdaGrad
- SGD + Nesterov momentum
- RMSProp
- AdaDelta

#### Original 15-epoch hyperparameter probes

Before running the full 200-epoch experiment, I ran 15-epoch probes to select learning rates:

```text
SGD+Nesterov alpha=0.003: 0.1343
SGD+Nesterov alpha=0.01:  0.0790
SGD+Nesterov alpha=0.03:  0.0725
SGD+Nesterov alpha=0.1:   1.0884

AdaGrad alpha=0.001: 0.2049
AdaGrad alpha=0.003: 0.1050
AdaGrad alpha=0.01:  0.0679
AdaGrad alpha=0.03:  0.1167

Adam alpha=0.0001: 0.0922
Adam alpha=0.0003: 0.0641
Adam alpha=0.001:  0.1016
Adam alpha=0.003:  0.2679

RMSProp alpha=0.0001: 0.1013
RMSProp alpha=0.0003: 0.0862
RMSProp alpha=0.001:  0.1899
RMSProp alpha=0.003:  0.4719
```

#### Original settings

Adam: alpha = 0.0003, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8
AdaGrad: alpha = 0.01, epsilon = 1e-8
SGD+Nesterov: alpha = 0.03, momentum = 0.9
RMSProp: alpha = 0.0003, decay = 0.9, epsilon = 1e-8
AdaDelta: rho = 0.95, epsilon = 1e-6, alpha = 1.0

#### Completed 200-epoch learning-rate search

Completed October 4, 2026: **26 of 26 trials**, each trained for **200 epochs**,
with no failed trials. Only Adam and AdaGrad were searched, using 13 learning
rates each and seed 0. Every trial resets the model, optimizer, and random state.

The grid multiplies Adam's original learning rate of `0.0003` and AdaGrad's
`0.01` by `[0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 7, 10]`.
Each optimizer is selected independently by its lowest **mean training loss over
epochs 181–200**, with epoch-200 loss and then lower learning rate breaking ties.
Neither selected rate is at a grid boundary.

The architecture remains two 1000-unit ReLU hidden layers, with batch size 128,
20% input dropout, and 50% hidden dropout. Adam keeps `beta1=0.9`, `beta2=0.999`,
and `epsilon=1e-8`; AdaGrad keeps `epsilon=1e-8`. Only learning rates change.

| Optimizer | Selected learning rate | Mean loss, epochs 181–200 | Epoch-200 loss |
|---|---:|---:|---:|
| Adam | **0.00009** | **0.011057** | **0.011676** |
| AdaGrad | **0.01** | **0.013596** | **0.013831** |

![Figure 2(a), completed Adam and AdaGrad search](assets/figure_2_recreation_tuned_v1.png)

Adam's selected trial has **18.7% lower final-20-epoch mean loss than AdaGrad**
and 38.0% lower than the original Adam trial at `0.0003` (mean loss `0.017839`).
AdaGrad's original rate remains its best. The SGD+Nesterov, RMSProp, and AdaDelta
curves use their original 200-epoch histories, verified byte-for-byte unchanged;
those optimizers were not searched again.

This supports the conclusion that selecting Adam's learning rate using only
15 epochs was insufficient for this 200-epoch objective. The result is closer
to the paper's late-training ordering, but does not establish an exact
reproduction: Adam is still slower early in this run, the search uses one seed,
and only learning rates for two optimizers were retuned. The retained dropout
rates are implementation choices, not verified settings from the Adam paper.

The [original chart](assets/figure_2_recreation.png), where AdaGrad finishes below
Adam, is preserved. See the [search records](results/figure2a_search_20261002_v1/search_results.json)
for all 26 trials and [artifact guide](FIGURE2A_SEARCH.md) for histories, frozen
source, and verification details. Running `experiments.py` uses the existing
experiment defaults; this chart comes from the archived search.

### Figure 3: CIFAR-10 CNN

_In progress_

## Implementation Notes

- **Variable naming** follows the paper exactly: `α=stepsize`, `β1=decay_1`, `β2=decay_2`, `ε=epsilon`, `θ=weights`, `g=grad`, `t=timestep`
- The `1/√t` stepsize decay applies **per epoch**, not per minibatch step, in experiment 1 (only).
- AdaGrad's learning rate already decays naturally via its accumulator; adding `1/√t` on top causes double decay and severely degrades performance

## File Structure

```

adam-recreation/
├── optimizers.py # Adam, AdaGrad, SGD_Nesterov, RMSProp, and AdaDelta classes
├── mlp.py # main model for multi-layer neural network in experiment 2
├── logreg.py # main model for logistic regression in experiment 1
├── utils.py # Helper functions like one-hot, pack, unpack
├── train.py # Training loop and data loading
├── experiments.py # Experiment calls and plots
└── assets/ # Output figures

```

## Usage

```bash
pip install numpy matplotlib torch torchvision
python experiments.py
```

## Reference

Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. _ICLR 2015_. https://arxiv.org/abs/1412.6980
