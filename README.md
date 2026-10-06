# Adam Optimizer Recreation

NumPy recreation of **"Adam: A Method for Stochastic Optimization"** (Kingma & Ba, ICLR 2015).

## Overview

This project implements optimizer updates in NumPy and recreates selected experiments from the original paper. The logistic regression, MLP, and VAE backward passes are hand-coded; the CNN uses PyTorch autograd for gradients and the same NumPy optimizer updates.

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
- `experiments.py` — the single user-facing entry point for experiment settings,
  runs, and plots. Its bottom section uses commented experiment blocks.
- `figure2a_search.py` — internal checkpoint/worker helper for the completed search;
  its grid is defined in `experiments.py`. Frozen run snapshots remain unchanged.
- `figure3_search.py` — internal checkpoint and supervisor helper for the
  38-trial CNN search; grids, entry points, and plotting live in `experiments.py`.

Regarding pack/unpack functions in utils.py: optimizers see one 1D vector regardless of the model. `pack([W, b, ...])` concatenates a list of arrays; `unpack(vector, shapes)` inverts it. This means
the same `Adam` instance handles logreg's 2 arrays (7,850 params) or the MLP's
6 arrays (1,796,010 params) with no code change.

## Experiments

### Figure 1: MNIST Logistic Regression

Historical MNIST logistic-regression chart. The retained baseline runner does not apply the paper's L2 regularization or epoch stepsize decay, and the original numerical histories are unavailable; this chart is not verified as an exact reproduction.

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

The [original chart](results/figure2a_search_20261002_v1/baseline/figure_2_recreation.png), where AdaGrad finishes below
Adam, is preserved. See the [search records](results/figure2a_search_20261002_v1/search_results.json)
for all 26 trials and [artifact guide](FIGURE2A_SEARCH.md) for histories, frozen
source, and verification details. The completed chart comes from the archived search. Its settings and plot entry
point now live in `experiments.py`; completed run calls are commented out.

### Figure 3: CIFAR-10 CNN

The approved **38-trial GPU search started October 5, 2026**: 22 learning-rate
trials followed by 16 targeted Adam/SGD momentum trials, including AdaGrad
retuning, with and without dropout. Every stable trial runs 45 epochs under
master seed 0. Selection uses mean training loss over epochs 41–45.
See the [protocol](FIGURE3_TUNING_PLAN.md). The in-progress run and its live
status remain on the training machine; they are not part of this publication.
On completion, both panels will be replaced with the six selected runs;
the chart below uses the earlier saved histories until then.

![Figure 3 recreation using the saved Adam follow-up](assets/figure_3_recreation.png)

Both panels use the saved Adam **without dropout** run at learning rate `0.0001`.
Its epoch-45 loss is **0.005508**, compared with **0.089495** for the previous
Adam curve at `0.001`. Adam with dropout remains at `0.0003`; all five other
curves retain their existing histories. Producing this saved-data chart required
no new training.

Panel (a) shows the first three **epoch averages** on a linear loss axis, not the
paper's within-epoch trajectory. Panel (b) shows all 45 epoch averages on a
logarithmic loss axis. The selected Adam run starts more slowly but ends below
the existing SGD curve; this is not an exact reproduction of the paper.

The [saved chart data and provenance](results/figure3_saved_update_20261005/manifest.json)
record the six histories used. Regenerate both panels with:

```bash
# In experiments.py, uncomment plot_figure3(), then:
python experiments.py
```

## Implementation Notes

- **Variable naming** follows the paper exactly: `α=stepsize`, `β1=decay_1`, `β2=decay_2`, `ε=epsilon`, `θ=weights`, `g=grad`, `t=timestep`
- The paper uses epoch stepsize decay for logistic regression; the retained historical baseline runner does not implement it.
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

All experiment calls are disabled by default. Uncomment the desired block at the
bottom of `experiments.py`, then run it. No data is loaded and no training begins
with every block commented. Figure 2(a) and the current saved Figure 3 chart are
complete; Figure 2(b) and the full Figure 4 comparison remain incomplete.
Figure 3's approved full-horizon search is described in [FIGURE3_TUNING_PLAN.md](FIGURE3_TUNING_PLAN.md).

The Figure 2(a) and Figure 3 search engines resume saved checkpoints. The other
training functions require a fresh output directory and do not resume partial
trials. The Figure 3 search preserves baseline assets and automatically replaces
its canonical chart after all planned trials have been accounted for and six
winners selected. Do not launch a second copy while the managed job is running.

## Reference

Kingma, D. P., & Ba, J. (2015). Adam: A Method for Stochastic Optimization. _ICLR 2015_. https://arxiv.org/abs/1412.6980
