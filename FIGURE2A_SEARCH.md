# Completed Figure 2(a) search

The Adam/AdaGrad search completed on October 4, 2026 at 04:27 PDT. All 26 trials
completed 200 epochs; none failed. The selected rates are Adam `0.00009` and
AdaGrad `0.01`, minimizing mean training loss over epochs 181–200.

## Published records

The archive is `results/figure2a_search_20261002_v1/`:

- `search_results.json`: all trial configurations, scores, and selected rates.
- `trials/<trial>/history.npy` and `result.json`: each 200-epoch loss history and its hash.
- `chart_data/`: the five exact histories plotted in the revised figure.
- `baseline/`: preserved original histories and chart.
- `source/`: immutable source snapshot used for the run, including the search runner.
- `manifest.json`: source/data/baseline hashes, grid, dropout, environment, and selection rule.
- `status.json`: completed status and timestamp.
- `validation.json`: pre-run validation evidence.
- `selected_diagnostics.json`: compact diagnostics for the two selected trials.
- `figure_2_recreation_tuned_v1.png`: the chart also published under `assets/`.

The source snapshot captures the local implementation at training time; it is
separate from the repository's top-level experiment defaults. The manifest and
status retain original machine paths as provenance. The frozen runner's resume
and plotting commands require those paths; they are not portable fresh-checkout
commands. MNIST data is not included. Saved NumPy histories can be read directly
with `numpy.load(path, allow_pickle=False)` without rerunning training.

## Validation and limits

Before training, nine tests covered original-loop/preprocessing parity, exact
serialized resume for both optimizers, diagnostic neutrality, nonfinite and stop
handling, selection, interruption/stall reporting, and original-chart/series
preservation. Baseline trials reproduced the original histories. At publication,
all 26 histories and their recorded scores and hashes were checked, as were the
selected histories, frozen source hashes, and unchanged comparison curves.

Only learning rates for Adam and AdaGrad were searched. Each trial used seed 0,
200 epochs, batch size 128, two 1000-unit ReLU hidden layers, and dropout of 20%
at the inputs and 50% in hidden layers. No pruning, additional seeds, or automatic
grid expansion was used. Neither winner is on a grid boundary. A single seed
does not establish robustness or a global optimum.

Checkpoints, temporary detailed diagnostics, launcher files, and per-epoch logs
are excluded from publication. The original figure remains unchanged.
