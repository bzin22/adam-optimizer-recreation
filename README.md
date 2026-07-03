# Adam Optimizer Recreation

NumPy recreation of **"Adam: A Method for Stochastic Optimization"** (Kingma & Ba, ICLR 2015).

## Overview

This project implements Adam, AdaGrad, and SGD with Nesterov momentum from scratch in NumPy and reproduces the experiments from the original paper. No PyTorch autograd — all forward passes, backward passes, and optimizer update rules are hand-coded.

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

<img width="991" height="486" alt="Screenshot 2026-07-01 at 5 09 50 PM" src="https://github.com/user-attachments/assets/925dc01f-0df5-48ec-8693-b2f93b85fd73" />

**Result:** Adam and SGD+Nesterov converge together and both outperform AdaGrad, consistent with the paper's findings.

### Figure 2: MNIST MLP with Dropout

Two hidden layers of 1000 units, ReLU, dropout. Same optimizer classes as
Figure 1 — the only change is the model in `mlp.py`.
_In progress_

### Figure 3: CIFAR-10 CNN

_In progress_

## Implementation Notes

- **Variable naming** follows the paper exactly: `α=stepsize`, `β1=decay_1`, `β2=decay_2`, `ε=epsilon`, `θ=weights`, `g=grad`, `t=timestep`
- The `1/√t` stepsize decay applies **per epoch**, not per minibatch step — an ambiguity in the paper that required empirical debugging to resolve
- AdaGrad's learning rate already decays naturally via its accumulator; adding `1/√t` on top causes double decay and severely degrades performance

## File Structure

```
adam-recreation/
├── optimizers.py     # Adam, AdaGrad, SGD_Nesterov classes
├── train.py          # Training loop, data loading, helpers
├── experiments.py    # Experiment calls and plots
└── assets/           # Output figures
```

## Usage

```bash
pip install numpy matplotlib torch torchvision
python experiments.py
```

## Reference

Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. _ICLR 2015_. https://arxiv.org/abs/1412.6980
