# Figure 3: approved 38-trial GPU search

Approved October 5, 2026. Entry point: `run_figure3_search()` in `experiments.py`.
The checkpoint/worker helper is `figure3_search.py`. No Figure 2 training is part
of this run. The original 154-trial proposal has been replaced by this plan.

## Stage 1: 22 learning-rate trials

Every configuration trains for 45 epochs, separately with and without dropout.
Adam beta1 and SGD momentum are initially 0.9.

| Optimizer | Learning rates | Trials |
|---|---|---:|
| Adam | 0.00006, 0.0001, 0.0003, 0.0006 | 8 |
| SGD + Nesterov | 0.003, 0.01, 0.03, 0.06 | 8 |
| AdaGrad | 0.001, 0.003, 0.01 | 6 |

## Stage 2: 16 targeted momentum trials

For each Adam/SGD dropout condition, take the two best completed stage-1 rates
by mean training loss over epochs 41–45. Test each rate at momentum 0.8 and 0.95.
The stage-1 trials at 0.9 remain eligible for final selection. AdaGrad has no
momentum parameter. Freeze the stage-2 shortlist and its stage-1 result hashes.
If a condition has fewer than two finite completed rates, stop and report it;
do not invent replacement trials or silently shrink the search.

Final selection is independent for all six optimizer/dropout conditions, by
mean loss over epochs 41–45, then epoch-45 loss, lower learning rate, and lower
momentum. Boundary winners are flagged without automatic expansion. This is a
narrowed search: rates outside the top two at momentum 0.9 may behave better at
other momenta and are not all tested.

## Fixed setup and reproducibility

- Current M2 Max GPU through MPS; NumPy optimizer updates remain on CPU.
- CIFAR-10 existing whitening; c64–c64–c128–1000 CNN; batch size 128.
- Seed 0 for every trial; fresh parameters, optimizer, and shuffle RNG each time.
- Dropout 20% at inputs and 50% at the hidden layer when enabled.
- Adam beta2 0.999; epsilon 1e-8; no learning-rate schedule.
- All numerically stable trials run the full 45 epochs. No short-run pruning.
- Source, data, configurations, baseline assets, and environment are recorded.

Torch 2.2.2 MPS RNG-state restoration alone did not reproduce dropout losses
across processes. The validated fix starts each epoch with
`torch.manual_seed(master_seed + zero_based_epoch)`. All trials share this same
epoch-substream policy under master seed 0. The real-CNN, fresh-process GPU test
then reproduced losses, weights, optimizer moments, and CPU/MPS RNG exactly.
This differs from the older continuous dropout RNG stream, so all 38 trials
are run afresh; old trial losses are not mixed into selection.

## Checkpoints, alerts, and charts

Save weights, optimizer state, shuffle RNG, CPU/MPS RNG, complete epoch and batch
losses, timings, and configuration hashes atomically after each epoch. Resume
from the last complete epoch. Remove completed-trial weight checkpoints only
after their durable histories and result hashes have been verified.

A separate supervisor sends local macOS notifications on startup, stage-1
completion, interruption, prolonged stale progress, and completion. Locks stop
duplicate workers. Numerical failure is recorded for that trial; process or GPU
errors stop the search. The managed job uses KeepAlive=false, so it does not
silently restart. A stopped/offline supervisor cannot deliver an alert while
offline, and macOS notification settings may suppress banners.

Keep the original Figure 3 in the run's `baseline/` directory. On completion,
select six winners and generate both panels from the same runs. Panel (a)
displays nonoverlapping means of 20 minibatches at their mean sub-epoch positions
on a linear axis; all raw losses are retained. Panel (b) uses epoch means on a
log axis. Verify the original asset has not changed before replacing
`assets/figure_3_recreation.png`. Figure 2 remains untouched.

## Timing and run records

Fresh GPU timings project approximately 5.46 training hours for 38 trials.
Budget approximately **6–8 hours** for checkpointing and sustained-run variation.
The run directory is `results/figure3_search_20261005_v1/`; inspect `status.json`
and `events.jsonl` for current state. Temporary checkpoints/logs are in
`.local-runs/figure3_search_20261005_v1/` and are excluded from Git.

Launched October 5 at approximately 17:09 PDT under
`gui/501/com.bzin.adam-figure3.20261005-v1`. `launch.json` records the explicit
launchd configuration. `/usr/bin/caffeinate -i` prevents idle sleep while the
job runs; the job does not automatically restart after failure or reboot.
Inspect it with:

```bash
launchctl print gui/501/com.bzin.adam-figure3.20261005-v1
cat results/figure3_search_20261005_v1/status.json
```

After verifying the worker has stopped, resume through `run_figure3_search()`
in `experiments.py`, or restart the existing loaded launchd job with
`launchctl kickstart gui/501/com.bzin.adam-figure3.20261005-v1` (without `-k`).
Do not restart a healthy job. If the job is no longer loaded, bootstrap the
recorded plist. Frozen source is used for every resumed trial.

The first startup banner failed because AppleScript rejected an escaped Unicode
dash. An ASCII startup notification was successfully retried and recorded in
`events.jsonl`; stage, interruption, stall, and completion messages are ASCII.
The training worker was unaffected.

The rates, momentum values, and selection metric are our explicit reproduction
protocol; the paper describes dense tuning without publishing this exact grid.
One seed does not establish robustness or a global optimum.
