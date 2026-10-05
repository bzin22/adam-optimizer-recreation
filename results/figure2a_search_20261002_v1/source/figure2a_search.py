"""Resumable, single-seed Adam/AdaGrad search for the existing Figure 2(a).

Prepare once from the working tree, then execute the frozen copy printed by
prepare. Results/source/baselines are retained; checkpoints are local-only.
No network access, other optimizer training, or automatic grid expansion.
"""
import argparse
import copy
from datetime import datetime, timezone
from decimal import Decimal
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import struct
import subprocess
import sys
import time
import traceback

import numpy as np
import mlp
from optimizers import Adam, AdaGrad
from utils import one_hot, pack, unpack

MULTIPLIERS = [0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5, 7, 10]
BASE_RATES = {"adam": 0.0003, "adagrad": 0.01}
BASELINES = ["adam", "adagrad", "sgd", "rms", "adadelta"]
SOURCE_FILES = ["figure2a_search.py", "mlp.py", "optimizers.py", "utils.py",
                "train.py", "experiments.py", "logreg.py", "cnn.py", "vae.py"]
STOP = False


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic(path, payload, kind="json"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with open(temp, "wb") as f:
        if kind == "pickle":
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        elif kind == "npy":
            np.save(f, payload, allow_pickle=False)
        elif kind == "bytes":
            f.write(payload)
        else:
            f.write((json.dumps(payload, indent=2, allow_nan=False) + "\n").encode())
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text())


def notify(message):
    """Local macOS notification, explicitly requested by the user. No messages to others."""
    try:
        result = subprocess.run(
            ["/usr/bin/osascript", "-e",
             "display notification " + json.dumps(message) +
             ' with title "Adam recreation"'],
            capture_output=True, text=True, timeout=10)
        return {"attempted": True, "returncode": result.returncode,
                "error": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"attempted": True, "error": str(e)}


def emit(root, event, **fields):
    record = {"time": utc(), "event": event, **fields}
    with open(root / "events.jsonl", "a") as f:
        f.write(json.dumps(record, allow_nan=False) + "\n")
        f.flush()
    if event != "epoch" or fields.get("epoch", 0) % 10 == 0:
        print(json.dumps(record, allow_nan=False), flush=True)


def configs():
    # Finish the existing-rate pair first; order does not change independently seeded trials.
    order = [1] + [m for m in MULTIPLIERS if m != 1]
    return [{"id": f"{name}_lr{float(Decimal(str(base)) * Decimal(str(m))):g}",
             "optimizer": name,
             "learning_rate": float(Decimal(str(base)) * Decimal(str(m))),
             "multiplier": m, "epochs": 200, "seed": 0, "batch_size": 128}
            for m in order for name, base in BASE_RATES.items()]


def init_params(seed=0, input_dim=784, hidden=1000, classes=10):
    rng = np.random.RandomState(seed)  # matches experiments.MNIST_MLP exactly
    return [rng.randn(hidden, input_dim) / np.sqrt(input_dim), np.zeros((hidden, 1)),
            rng.randn(hidden, hidden) / np.sqrt(hidden), np.zeros((hidden, 1)),
            rng.randn(classes, hidden) / np.sqrt(hidden), np.zeros((classes, 1))]


def make_optimizer(config, weights):
    alpha = config["learning_rate"]
    if config["optimizer"] == "adam":
        return Adam(alpha, weights, 0.9, 0.999, 1e-8)
    if config["optimizer"] == "adagrad":
        return AdaGrad(alpha, weights, 1e-8)
    raise ValueError("Only Adam and AdaGrad are authorized")


def start_trial(config, params=None):
    params = init_params(config["seed"]) if params is None else params
    mlp.reseed(config["seed"])
    return (make_optimizer(config, pack(params)), [p.shape for p in params],
            np.random.default_rng(config["seed"]))


def checkpoint(config, optimizer, shapes, shuffle, history, timings, manifest_sha):
    return {"config": config, "manifest_sha256": manifest_sha,
            "optimizer_state": copy.deepcopy(optimizer.__dict__), "shapes": shapes,
            "shuffle_rng_state": copy.deepcopy(shuffle.bit_generator.state),
            "dropout_rng_state": copy.deepcopy(mlp._rng.bit_generator.state),
            "history": list(history), "epoch_seconds": list(timings)}


def restore(saved, config, manifest_sha):
    if saved["config"] != config or saved["manifest_sha256"] != manifest_sha:
        raise ValueError("Checkpoint configuration/source manifest mismatch")
    optimizer = make_optimizer(config, saved["optimizer_state"]["weights"].copy())
    optimizer.__dict__.update(copy.deepcopy(saved["optimizer_state"]))
    shuffle = np.random.default_rng()
    shuffle.bit_generator.state = copy.deepcopy(saved["shuffle_rng_state"])
    mlp.reseed(config["seed"])
    mlp._rng.bit_generator.state = copy.deepcopy(saved["dropout_rng_state"])
    return optimizer, saved["shapes"], shuffle, list(saved["history"]), list(saved["epoch_seconds"])


def assert_finite_state(optimizer):
    for name, value in optimizer.__dict__.items():
        if isinstance(value, np.ndarray) and not np.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite optimizer state: {name}")


def rms(values):
    scale = float(np.max(np.abs(values)))
    return scale * float(np.sqrt(np.mean((values / scale) ** 2))) if scale else 0.0


def run_epoch(optimizer, shapes, shuffle, inputs, labels, batch_size=128, diagnostic=False):
    losses = []
    order = shuffle.permutation(len(inputs))
    detail = None
    with np.errstate(over="raise", divide="raise", invalid="raise", under="ignore"):
        for i in range(0, len(inputs), batch_size):
            if STOP:
                raise InterruptedError("Stop requested; resume from the last completed epoch")
            idx = order[i:i + batch_size]
            params = unpack(optimizer.weights, shapes)
            loss, grads = mlp.mlp_backward(params, inputs[idx], labels[:, idx], 0.0, dropout=True)
            grad = pack(grads)
            if not np.isfinite(loss) or not np.isfinite(grad).all():
                raise FloatingPointError("Nonfinite minibatch loss or gradient")
            last = i + batch_size >= len(inputs)
            before = optimizer.weights.copy() if diagnostic and last else None
            optimizer.update(grad)
            losses.append(loss)
            if before is not None:
                if isinstance(optimizer, Adam):
                    denominator = np.sqrt(optimizer.v_t / (1 - optimizer.decay_2 ** optimizer.t)) + optimizer.epsilon
                else:
                    denominator = np.sqrt(optimizer.v_t) + optimizer.epsilon
                # Summaries only. No full gradient/update arrays are persisted.
                detail = {"gradient_rms": [rms(g) for g in grads],
                          "update_rms": [rms(u)
                                         for u in unpack(optimizer.weights - before, shapes)],
                          "denominator_mean": [float(d.mean()) for d in unpack(denominator, shapes)]}
    assert_finite_state(optimizer)
    return float(np.mean(losses)), detail  # same unweighted minibatch mean as train.py


def load_data(root):
    image_path = root / "train-images-idx3-ubyte"
    label_path = root / "train-labels-idx1-ubyte"
    with open(image_path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        pixels = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, rows * cols)
    with open(label_path, "rb") as f:
        label_magic, label_n = struct.unpack(">II", f.read(8))
        targets = np.frombuffer(f.read(), dtype=np.uint8)
    if (magic, n, rows, cols, label_magic, label_n, len(targets)) != (2051, 60000, 28, 28, 2049, 60000, 60000):
        raise ValueError("Unexpected MNIST training data")
    inputs = pixels.astype(np.float32) / 255.0
    inputs = (inputs - 0.1307) / 0.3081
    return inputs, one_hot(10, targets)


def prepare(repo, root):
    repo, root = repo.resolve(), root.resolve()
    if root.exists():
        raise FileExistsError(f"Refusing to replace an existing search: {root}")
    source = root / "source"
    baseline = root / "baseline"
    source.mkdir(parents=True)
    baseline.mkdir()
    sources, originals = {}, {}
    for name in SOURCE_FILES:
        original = repo / name
        atomic(source / name, original.read_bytes(), "bytes")
        sources[name] = digest(source / name)
    for name in BASELINES:
        original = repo.parent / f"history_mlp_{name}.npy"
        values = np.load(original, allow_pickle=False)
        if values.shape != (200,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid original history: {original}")
        atomic(baseline / original.name, original.read_bytes(), "bytes")
        originals[name] = {"path": str(original), "sha256": digest(original)}
    original_figure = repo / "assets/figure_2_recreation.png"
    atomic(baseline / original_figure.name, original_figure.read_bytes(), "bytes")
    data_root = repo / "data/MNIST/raw"
    data = {name: digest(data_root / name) for name in
            ["train-images-idx3-ubyte", "train-labels-idx1-ubyte"]}
    scratch = repo / ".local-runs" / root.name
    scratch.mkdir(parents=True, exist_ok=False)
    manifest = {"created_at": utc(), "repo": str(repo), "run_dir": str(root),
                "scratch": str(scratch), "data_root": str(data_root), "data_sha256": data,
                "source_sha256": sources, "baseline": originals,
                "original_figure": {"path": str(original_figure), "sha256": digest(original_figure)},
                "python": sys.version, "numpy": np.__version__, "threads": 4,
                "dropout": {"input_drop_probability": 0.2, "hidden_drop_probability": 0.5},
                "selection": "lowest mean loss at epochs 181-200; ties: epoch 200, then lower rate",
                "trials": configs(), "original_state": subprocess.run(
                    ["git", "status", "--short"], cwd=repo, capture_output=True, text=True).stdout}
    atomic(root / "manifest.json", manifest)
    atomic(root / "status.json", {"state": "prepared", "updated_at": utc(), "completed": 0, "failed": 0})
    print(f"Prepared {len(manifest['trials'])} trials in {root}")
    print(f"Run frozen code: {source / 'figure2a_search.py'} supervise --run-dir {root}")


def verify(root, manifest):
    if sys.version != manifest["python"] or np.__version__ != manifest["numpy"]:
        raise ValueError("Python/NumPy environment differs from the frozen manifest")
    if os.environ.get("OPENBLAS_NUM_THREADS") != str(manifest["threads"]):
        raise ValueError("Run with OPENBLAS_NUM_THREADS=4")
    for name, expected in manifest["source_sha256"].items():
        if digest(root / "source" / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    for module in (mlp, sys.modules[Adam.__module__], sys.modules[pack.__module__]):
        if Path(module.__file__).resolve().parent != (root / "source").resolve():
            raise ValueError("Training must import the frozen source modules")
    for name, expected in manifest["data_sha256"].items():
        if digest(Path(manifest["data_root"]) / name) != expected:
            raise ValueError(f"Training data changed: {name}")
    for name, record in manifest["baseline"].items():
        if digest(root / "baseline" / f"history_mlp_{name}.npy") != record["sha256"]:
            raise ValueError(f"Preserved baseline changed: {name}")


def trial_record(root, config):
    path = root / "trials" / config["id"] / "result.json"
    return read_json(path) if path.exists() else None


def counts(root, manifest):
    records = [trial_record(root, c) for c in manifest["trials"]]
    return {s: sum(r is not None and r["status"] == s for r in records)
            for s in ["completed", "failed"]}


def write_status(root, manifest, state, **extra):
    atomic(root / "status.json", {"state": state, "pid": os.getpid(), "updated_at": utc(),
                                   **counts(root, manifest), **extra})


def run_trial(root, manifest, config, inputs, labels):
    out = root / "trials" / config["id"]
    scratch = Path(manifest["scratch"])
    cp = scratch / (config["id"] + ".pkl")
    manifest_sha = digest(root / "manifest.json")
    if cp.exists():
        # Only checkpoints produced by this local run are read.
        with open(cp, "rb") as f:
            optimizer, shapes, shuffle, history, timings = restore(pickle.load(f), config, manifest_sha)
    else:
        optimizer, shapes, shuffle = start_trial(config)
        history, timings = [], []
        atomic(cp, checkpoint(config, optimizer, shapes, shuffle, history, timings, manifest_sha), "pickle")
    emit(root, "trial_start", trial=config["id"], resume_epoch=len(history))
    write_status(root, manifest, "running", trial=config["id"], epoch=len(history))
    for epoch in range(len(history) + 1, config["epochs"] + 1):
        if STOP:
            raise InterruptedError("Stop requested")
        start = time.perf_counter()
        loss, detail = run_epoch(optimizer, shapes, shuffle, inputs, labels,
                                 config["batch_size"], diagnostic=epoch % 10 == 0)
        # Independent parity check for the original-rate trials, using existing saved evidence.
        if config["multiplier"] == 1 and epoch <= 5:
            reference = np.load(root / "baseline" / f"history_mlp_{config['optimizer']}.npy")
            if not np.isclose(loss, reference[epoch - 1], rtol=0, atol=1e-12):
                raise RuntimeError("Original-rate history parity failed; stop the search")
        history.append(loss)
        timings.append(time.perf_counter() - start)
        atomic(cp, checkpoint(config, optimizer, shapes, shuffle, history, timings, manifest_sha), "pickle")
        atomic(out / "history.npy", np.asarray(history), "npy")
        atomic(out / "progress.json", {"config": config, "epoch": epoch,
                                       "loss": loss, "epoch_seconds": timings, "updated_at": utc()})
        if detail is not None:
            with open(scratch / (config["id"] + "_diagnostics.jsonl"), "a") as f:
                f.write(json.dumps({"epoch": epoch, **detail}, allow_nan=False) + "\n")
        emit(root, "epoch", trial=config["id"], epoch=epoch, loss=loss, seconds=timings[-1])
        write_status(root, manifest, "running", trial=config["id"], epoch=epoch, loss=loss)
    # Recover an exit between the last checkpoint commit and history/result publication.
    atomic(out / "history.npy", np.asarray(history), "npy")
    record = {"config": config, "status": "completed", "epochs": len(history),
              "mean_last20": float(np.mean(history[-20:])), "final_loss": history[-1],
              "training_seconds": sum(timings), "history_sha256": digest(out / "history.npy")}
    atomic(out / "result.json", record)
    emit(root, "trial_completed", trial=config["id"], mean_last20=record["mean_last20"], final_loss=record["final_loss"])


def choose_best(records, name):
    valid = [r for r in records if r["status"] == "completed"
             and r["config"]["optimizer"] == name and r["epochs"] == 200
             and np.isfinite(r["mean_last20"]) and np.isfinite(r["final_loss"])]
    if not valid:
        raise ValueError(f"No complete finite trial for {name}")
    return min(valid, key=lambda r: (r["mean_last20"], r["final_loss"], r["config"]["learning_rate"]))


def finalize(root, manifest):
    records = [trial_record(root, c) for c in manifest["trials"]]
    if any(r is None or r["status"] not in ("completed", "failed") for r in records):
        raise ValueError("Cannot finalize an incomplete search")
    for original in list(manifest["baseline"].values()) + [manifest["original_figure"]]:
        if digest(original["path"]) != original["sha256"]:
            raise ValueError("An original input changed during the run; inspect before publishing")
    for record in records:
        if record["status"] == "completed":
            p = root / "trials" / record["config"]["id"] / "history.npy"
            values = np.load(p, allow_pickle=False)
            if (values.shape != (200,) or not np.isfinite(values).all()
                    or digest(p) != record["history_sha256"]
                    or float(np.mean(values[-20:])) != record["mean_last20"]
                    or float(values[-1]) != record["final_loss"]):
                raise ValueError("Completed history and recorded selection metrics disagree")
    selected = {name: choose_best(records, name) for name in BASE_RATES}
    chart_data = root / "chart_data"
    for name in BASELINES:
        if name in selected:
            src = root / "trials" / selected[name]["config"]["id"] / "history.npy"
        else:
            src = root / "baseline" / f"history_mlp_{name}.npy"
        atomic(chart_data / f"history_mlp_{name}.npy", src.read_bytes(), "bytes")
        if name not in selected and digest(chart_data / src.name) != manifest["baseline"][name]["sha256"]:
            raise ValueError("An unchanged chart series changed")
    for name, chosen in selected.items():
        chosen["at_grid_boundary"] = chosen["config"]["multiplier"] in (0.1, 10)
    atomic(root / "search_results.json", {"selection": manifest["selection"],
                                         "selected": selected, "trials": records})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5.4))
    labels = {"adam": "Adam", "adagrad": "AdaGrad", "sgd": "SGD + Nesterov",
              "adadelta": "AdaDelta", "rms": "RMSProp"}
    for name in ["adam", "adagrad", "sgd", "adadelta", "rms"]:
        values = np.load(chart_data / f"history_mlp_{name}.npy", allow_pickle=False)
        if values.shape != (200,) or not np.isfinite(values).all():
            raise ValueError("Chart must contain complete finite histories")
        label = labels[name]
        if name in selected:
            label += f" (α={selected[name]['config']['learning_rate']:g})"
        ax.plot(values, label=label)
    ax.set(yscale="log", xlabel="iterations over entire dataset", ylabel="training loss",
           title="MNIST MLP + Dropout — Figure 2(a) Recreation")
    ax.legend()
    ax.grid(True)
    fig.text(.5, .015, "Adam/AdaGrad: 200-epoch learning-rate search, seed 0. Other curves: original settings.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    figure = root / "figure_2_recreation_tuned_v1.png"
    fig.savefig(figure, dpi=200)
    plt.close(fig)
    # Preserve the original asset. Publish only the new version after all checks pass.
    asset = Path(manifest["repo"]) / "assets" / figure.name
    if asset.exists() and digest(asset) != digest(figure):
        raise FileExistsError(f"Refusing to replace a different figure: {asset}")
    atomic(asset, figure.read_bytes(), "bytes")
    diagnostic_summary = {}
    for name in selected:
        p = Path(manifest["scratch"]) / (selected[name]["config"]["id"] + "_diagnostics.jsonl")
        if p.exists():
            diagnostic_summary[name] = [json.loads(line) for line in p.read_text().splitlines()]
    if diagnostic_summary or not (root / "selected_diagnostics.json").exists():
        atomic(root / "selected_diagnostics.json", diagnostic_summary)
    write_status(root, manifest, "completed", figure=str(asset), selected={
        n: r["config"]["learning_rate"] for n, r in selected.items()})
    emit(root, "search_completed", figure=str(asset))
    # Delete only this search's own disposable checkpoints/diagnostics, after verification.
    for c in manifest["trials"]:
        for suffix in (".pkl", "_diagnostics.jsonl"):
            (Path(manifest["scratch"]) / (c["id"] + suffix)).unlink(missing_ok=True)


def request_stop(signum, frame):
    global STOP
    STOP = True


def run(root):
    manifest = read_json(root / "manifest.json")
    lock = open(Path(manifest["scratch"]) / "run.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        verify(root, manifest)
        inputs, labels = load_data(Path(manifest["data_root"]))
        for config in manifest["trials"]:
            existing = trial_record(root, config)
            if existing and existing["status"] in ("completed", "failed"):
                continue
            try:
                run_trial(root, manifest, config, inputs, labels)
            except FloatingPointError as e:
                record = {"config": config, "status": "failed", "reason": str(e), "time": utc()}
                atomic(root / "trials" / config["id"] / "result.json", record)
                emit(root, "trial_failed", trial=config["id"], reason=str(e))
        if STOP:
            raise InterruptedError("Stop requested")
        finalize(root, manifest)
    except BaseException as e:
        state = "interrupted" if isinstance(e, (InterruptedError, KeyboardInterrupt)) else "error"
        previous = read_json(root / "status.json")
        write_status(root, manifest, state, trial=previous.get("trial"), epoch=previous.get("epoch"),
                     error=str(e), traceback=traceback.format_exc())
        emit(root, state, error=str(e))
        raise
    finally:
        lock.close()


def supervise(root):
    manifest = read_json(root / "manifest.json")
    # A separate lock prevents duplicate supervisors from misreporting a healthy run as interrupted.
    lock = open(Path(manifest["scratch"]) / "supervisor.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    emit(root, "supervisor_start", notification=notify("Adam/AdaGrad search starting. Estimated duration: 24-30 hours."))
    child = subprocess.Popen([sys.executable, "-u", str(root / "source/figure2a_search.py"),
                              "run", "--run-dir", str(root)])
    def stop_child(signum, frame):
        child.send_signal(signum)
    signal.signal(signal.SIGTERM, stop_child)
    signal.signal(signal.SIGINT, stop_child)
    stalled_notified = False
    while True:
        try:
            code = child.wait(timeout=30)
            break
        except subprocess.TimeoutExpired:
            progress = read_json(root / "status.json")
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(progress["updated_at"])).total_seconds()
            if progress["state"] == "running" and age > 180 and not stalled_notified:
                emit(root, "heartbeat_stale", seconds=age, notification=notify(
                    "Adam/AdaGrad has not saved an epoch for over 3 minutes. Training may be paused or stalled."))
                stalled_notified = True
            elif age <= 180:
                stalled_notified = False
    status = read_json(root / "status.json")
    if code != 0 or status["state"] != "completed":
        if status["state"] not in ("error", "interrupted"):
            atomic(root / "status.json", {**status, "state": "interrupted", "updated_at": utc(),
                                           "exit_code": code})
        message = "Adam/AdaGrad search interrupted. Progress is saved; check the run status before resuming."
    else:
        message = "Adam/AdaGrad search finished. The revised Figure 2(a) and selected learning rates are saved."
    emit(root, "supervisor_exit", exit_code=code, notification=notify(message))
    lock.close()
    return code


def status(root):
    record = read_json(root / "status.json")
    if record.get("state") == "running":
        try:
            os.kill(record["pid"], 0)
            record["process_alive"] = True
        except ProcessLookupError:
            record["process_alive"] = False
            record["state"] = "interrupted (process missing)"
        except PermissionError:
            record["process_alive"] = "unknown (permission denied)"
    print(json.dumps(record, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "supervise", "status", "plot"])
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if args.command == "prepare":
        prepare(args.repo, root)
    elif args.command == "run":
        run(root)
    elif args.command == "supervise":
        return supervise(root)
    elif args.command == "status":
        status(root)
    else:
        finalize(root, read_json(root / "manifest.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
