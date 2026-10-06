"""Internal resumable CNN search engine. User entry points live in experiments.py."""
import copy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torchvision.datasets import CIFAR10
import cnn
import experiments
import optimizers
from utils import pack, unpack, one_hot
from figure2a_search import atomic, digest, notify

STOP = False
SOURCES = ['experiments.py', 'figure3_search.py', 'figure2a_search.py', 'cnn.py',
           'optimizers.py', 'train.py', 'utils.py', 'logreg.py', 'mlp.py', 'vae.py']


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def emit(root, event, **fields):
    row = dict(time=utc(), event=event, **fields)
    with open(root / 'events.jsonl', 'a') as f:
        f.write(json.dumps(row, allow_nan=False) + '\n')
    print(json.dumps(row, allow_nan=False), flush=True)


def status(root, state, **fields):
    previous = read(root / 'status.json') if (root / 'status.json').exists() else {}
    atomic(root / 'status.json', {**previous, 'state': state, 'pid': os.getpid(),
                                  'updated_at': utc(), **fields})


def stop(signum, frame):
    global STOP
    STOP = True


def prepare(repo, root):
    repo, root = Path(repo).resolve(), Path(root).resolve()
    if root.exists():
        raise FileExistsError(root)
    source = root / 'source'
    source.mkdir(parents=True)
    for name in SOURCES:
        atomic(source / name, (repo / name).read_bytes(), 'bytes')
    baseline = root / 'baseline'
    baseline.mkdir()
    assets = {}
    for name in ['figure_3_recreation.png', 'figure_2_recreation_tuned_v1.png']:
        original = repo / 'assets' / name
        atomic(baseline / name, original.read_bytes(), 'bytes')
        assets[name] = {'path': str(original), 'sha256': digest(original)}
    cache = repo.parent / 'cifar10_zca.npy'
    data_root = repo.parent / 'data'
    data_files = [cache] + [data_root / 'cifar-10-batches-py' / f'data_batch_{i}' for i in range(1, 6)]
    scratch = repo / '.local-runs' / root.name
    scratch.mkdir(parents=True, exist_ok=False)
    manifest = {'created_at': utc(), 'repo': str(repo), 'run_dir': str(root),
                'scratch': str(scratch), 'cache': str(cache), 'data_root': str(data_root),
                'data_sha256': {str(p): digest(p) for p in data_files},
                'source_sha256': {n: digest(source / n) for n in SOURCES},
                'original_assets': assets, 'python': sys.version, 'numpy': np.__version__,
                'torch': torch.__version__, 'device': 'mps', 'threads': 4,
                'stage1': experiments.figure3_stage1_configs(),
                'stage2_policy': {'top_rates_per_condition': 2, 'momenta': [0.8, 0.95]},
                'planned_trials': 38, 'seed': 0, 'epochs': 45, 'batch_size': 128,
                'dropout_rng_policy': 'torch.manual_seed(master_seed + zero_based_epoch) before every epoch; verified MPS cross-process resume',
                'selection': 'lowest mean loss at epochs 41-45; ties: epoch 45, lower rate, lower momentum',
                'model': 'c64-c64-c128-1000; input dropout 0.2, hidden dropout 0.5 when enabled',
                'expected_examples': 50000, 'expected_batches': 391}
    atomic(root / 'manifest.json', manifest)
    status(root, 'prepared', completed=0, failed=0, planned_trials=38, stage=1)
    return manifest


def verify(root, manifest):
    if (sys.version, np.__version__, torch.__version__) != (manifest['python'], manifest['numpy'], manifest['torch']):
        raise RuntimeError('Python/NumPy/Torch environment differs from frozen run')
    if os.environ.get('OPENBLAS_NUM_THREADS') != '4':
        raise RuntimeError('Use OPENBLAS_NUM_THREADS=4')
    torch.set_num_threads(manifest['threads'])
    if torch.get_num_threads() != manifest['threads']:
        raise RuntimeError('Torch thread count mismatch')
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS GPU unavailable; refusing silent CPU fallback')
    for name, expected in manifest['source_sha256'].items():
        if digest(root / 'source' / name) != expected:
            raise RuntimeError(f'Frozen source changed: {name}')
    for module in [cnn, experiments, optimizers]:
        if Path(module.__file__).resolve().parent != root / 'source':
            raise RuntimeError('Worker must import frozen modules')
    for name, expected in manifest['data_sha256'].items():
        if digest(name) != expected:
            raise RuntimeError(f'Input changed: {name}')


def make_optimizer(config, weights):
    rate, momentum = config['learning_rate'], config['momentum']
    if config['optimizer'] == 'adam':
        return optimizers.Adam(rate, weights, momentum, 0.999, 1e-8)
    if config['optimizer'] == 'sgd_nesterov':
        return optimizers.SGD_Nesterov(rate, weights, momentum)
    if config['optimizer'] == 'adagrad':
        return optimizers.AdaGrad(rate, weights, 1e-8)
    raise ValueError(config['optimizer'])


def start_trial(config, device='mps', model_factory=None):
    model = (model_factory or cnn.build_cnn)(dropout=config['dropout'], seed=config['seed'])
    params = cnn.cnn_params(model)
    shapes = [p.shape for p in params]
    optimizer = make_optimizer(config, pack(params))
    backward = cnn.make_cnn_backward(model, device)
    shuffle = np.random.default_rng(config['seed'])
    return model, optimizer, shapes, backward, shuffle


def checkpoint(config, optimizer, shapes, shuffle, history, batches, timings, manifest_sha, device):
    if device == 'mps':
        torch.mps.synchronize()
    return {'config': config, 'manifest_sha': manifest_sha,
            'optimizer': copy.deepcopy(optimizer.__dict__), 'shapes': shapes,
            'shuffle_rng': copy.deepcopy(shuffle.bit_generator.state),
            'torch_rng': torch.get_rng_state(),
            'mps_rng': torch.mps.get_rng_state() if device == 'mps' else None,
            'history': list(history), 'batch_losses': list(batches), 'timings': list(timings)}


def restore(saved, config, manifest_sha, device, model_factory=None):
    if saved['config'] != config or saved['manifest_sha'] != manifest_sha:
        raise ValueError('Checkpoint/configuration mismatch')
    model, optimizer, shapes, backward, shuffle = start_trial(config, device, model_factory)
    if shapes != saved['shapes']:
        raise ValueError('Checkpoint architecture mismatch')
    optimizer.__dict__ = copy.deepcopy(saved['optimizer'])
    shuffle.bit_generator.state = copy.deepcopy(saved['shuffle_rng'])
    torch.set_rng_state(saved['torch_rng'])
    if device == 'mps':
        torch.mps.set_rng_state(saved['mps_rng'])
    return model, optimizer, shapes, backward, shuffle


def run_epoch(optimizer, shapes, backward, shuffle, inputs, labels, batch_size=128, heartbeat=None):
    if STOP:
        raise InterruptedError('Stop requested')
    order = shuffle.permutation(len(inputs))
    losses = []
    for i in range(0, len(inputs), batch_size):
        if STOP:
            raise InterruptedError('Stop requested')
        idx = order[i:i + batch_size]
        loss, grads = backward(unpack(optimizer.weights, shapes), inputs[idx], labels[:, idx], 0.0)
        if not np.isfinite(loss) or not all(np.isfinite(g).all() for g in grads):
            raise FloatingPointError('Nonfinite loss or gradient')
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            optimizer.update(pack(grads))
        if not np.isfinite(optimizer.weights).all():
            raise FloatingPointError('Nonfinite parameters')
        losses.append(loss)
        if heartbeat and len(losses) % 100 == 0:
            heartbeat(len(losses))
    return float(np.mean(losses)), losses


def load_record(root, config):
    path = root / 'trials' / config['id'] / 'result.json'
    if not path.exists():
        return None
    record = read(path)
    if record['config'] != config or record['status'] not in ('completed', 'failed'):
        raise ValueError('Invalid saved trial record')
    if record['status'] == 'completed':
        history = np.load(path.parent / 'history.npy', allow_pickle=False)
        batches = np.load(path.parent / 'batch_losses.npy', allow_pickle=False)
        expected_batches = config['epochs'] * config.get('batches_per_epoch', 391)
        if (history.shape != (config['epochs'],) or batches.shape != (expected_batches,)
                or not np.isfinite(history).all() or not np.isfinite(batches).all()
                or digest(path.parent / 'history.npy') != record['history_sha256']
                or digest(path.parent / 'batch_losses.npy') != record['batch_sha256']
                or float(history[-5:].mean()) != record['mean_last5']
                or float(history[-1]) != record['final_loss']):
            raise ValueError('Saved histories do not match trial metrics/hashes')
        np.testing.assert_array_equal(history, batches.reshape(config['epochs'], -1).mean(axis=1))
    return record


def run_trial(root, manifest, config, inputs, labels, device='mps', model_factory=None):
    out = root / 'trials' / config['id']
    cp = Path(manifest['scratch']) / (config['id'] + '.pkl')
    signature = digest(root / 'manifest.json')
    if cp.exists():
        with open(cp, 'rb') as f:
            saved = pickle.load(f)
        model, optimizer, shapes, backward, shuffle = restore(saved, config, signature, device, model_factory)
        history, batches, timings = saved['history'], saved['batch_losses'], saved['timings']
    else:
        model, optimizer, shapes, backward, shuffle = start_trial(config, device, model_factory)
        history, batches, timings = [], [], []
        atomic(cp, checkpoint(config, optimizer, shapes, shuffle, history, batches, timings, signature, device), 'pickle')
    emit(root, 'trial_start', trial=config['id'], resume_epoch=len(history), config=config)
    for epoch in range(len(history), config['epochs']):
        # Torch 2.2.2 MPS RNG-state restore alone failed the cross-process dropout
        # check. A deterministic epoch substream reproduces restart trajectories.
        # Every trial uses the same master seed and the same epoch substreams.
        torch.manual_seed(config['seed'] + epoch)
        status(root, 'running', trial=config['id'], epoch=epoch, batch=0, stage=config['stage'])
        started = time.perf_counter()
        loss, epoch_batches = run_epoch(optimizer, shapes, backward, shuffle, inputs, labels,
            config['batch_size'], lambda batch: status(root, 'running', batch=batch))
        if device == 'mps':
            torch.mps.synchronize()
        history.append(loss)
        batches.extend(epoch_batches)
        timings.append(time.perf_counter() - started)
        atomic(cp, checkpoint(config, optimizer, shapes, shuffle, history, batches, timings, signature, device), 'pickle')
        atomic(out / 'history.npy', np.asarray(history), 'npy')
        atomic(out / 'batch_losses.npy', np.asarray(batches), 'npy')
        status(root, 'running', epoch=epoch + 1, batch=0, loss=loss)
        emit(root, 'epoch', trial=config['id'], epoch=epoch + 1, loss=loss, seconds=timings[-1])
    # Recover a stop after the final checkpoint but before result publication.
    atomic(out / 'history.npy', np.asarray(history), 'npy')
    atomic(out / 'batch_losses.npy', np.asarray(batches), 'npy')
    record = {'config': config, 'status': 'completed', 'epochs': len(history),
              'mean_last5': float(np.mean(history[-5:])), 'final_loss': history[-1],
              'epoch_seconds': timings, 'training_seconds': sum(timings),
              'history_sha256': digest(out / 'history.npy'), 'batch_sha256': digest(out / 'batch_losses.npy')}
    atomic(out / 'result.json', record)
    load_record(root, config)
    cp.unlink(missing_ok=True)
    emit(root, 'trial_completed', trial=config['id'], mean_last5=record['mean_last5'])
    return record


def rank(record):
    c = record['config']
    return (record['mean_last5'], record['final_loss'], c['learning_rate'], c['momentum'] or 0)


def choose(records, name, dropout):
    valid = [r for r in records if r and r['status'] == 'completed'
             and r['config']['optimizer'] == name and r['config']['dropout'] == dropout]
    if not valid:
        raise ValueError(f'No complete finite trial for {name}, dropout={dropout}')
    return min(valid, key=rank)


def stage2(root, records):
    configs = experiments.figure3_stage2_configs(records)
    payload = {'configs': configs,
               'stage1_result_sha256': {r['config']['id']: digest(root / 'trials' / r['config']['id'] / 'result.json') for r in records}}
    path = root / 'stage2.json'
    if path.exists() and read(path) != payload:
        raise ValueError('Stage 2 selection changed after it was frozen')
    atomic(path, payload)
    return configs


def finalize(root, manifest):
    configs = manifest['stage1'] + read(root / 'stage2.json')['configs']
    records = [load_record(root, c) for c in configs]
    if len(configs) != 38 or any(r is None for r in records):
        raise ValueError('Cannot finalize an incomplete search')
    selected = {}
    for name in ['adam', 'adagrad', 'sgd_nesterov']:
        for dropout in [False, True]:
            key = name + ('_dropout' if dropout else '')
            record = copy.deepcopy(choose(records, name, dropout))
            rates = experiments.FIGURE3_LR_GRID[name]
            record['rate_at_stage1_boundary'] = record['config']['learning_rate'] in (min(rates), max(rates))
            record['momentum_at_tested_boundary'] = (record['config']['momentum'] in [0.8, 0.95]) if name != 'adagrad' else False
            selected[key] = record
            trial = root / 'trials' / record['config']['id']
            for filename in ['history.npy', 'batch_losses.npy']:
                atomic(root / 'chart_data' / key / filename, (trial / filename).read_bytes(), 'bytes')
    atomic(root / 'search_results.json', {'selection': manifest['selection'], 'selected': selected, 'trials': records})
    experiments.plot_figure3_search(root)
    figure = root / 'figure_3_recreation.png'
    original = manifest['original_assets']['figure_3_recreation.png']
    asset = Path(original['path'])
    if digest(asset) not in (original['sha256'], digest(figure)):
        raise ValueError('Current Figure 3 was edited during search; not replacing it')
    other = manifest['original_assets']['figure_2_recreation_tuned_v1.png']
    if digest(other['path']) != other['sha256']:
        raise ValueError('Figure 2 changed during search; inspect before publishing')
    atomic(asset, figure.read_bytes(), 'bytes')
    status(root, 'completed', completed=sum(r['status'] == 'completed' for r in records),
           failed=sum(r['status'] == 'failed' for r in records), selected={k: v['config'] for k, v in selected.items()},
           figure=str(asset), trial=None, epoch=None, batch=None)
    emit(root, 'search_completed', figure=str(asset))


def run(root):
    global STOP
    STOP = False
    root = Path(root).resolve()
    manifest = read(root / 'manifest.json')
    with open(Path(manifest['scratch']) / 'run.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            if read(root / 'status.json')['state'] == 'completed':
                return
            verify(root, manifest)
            inputs = np.load(manifest['cache'], allow_pickle=False)
            dataset = CIFAR10(root=manifest['data_root'], train=True, download=False)
            labels = one_hot(10, np.asarray(dataset.targets))
            if inputs.shape != (50000, 3, 32, 32):
                raise ValueError('Unexpected CIFAR inputs')
            records = []
            for stage in [1, 2]:
                configs = manifest['stage1'] if stage == 1 else stage2(root, records)
                for config in configs:
                    if STOP:
                        raise InterruptedError('Stop requested')
                    record = load_record(root, config)
                    if record is None:
                        try:
                            record = run_trial(root, manifest, config, inputs, labels)
                        except FloatingPointError as error:
                            record = {'config': config, 'status': 'failed', 'reason': str(error), 'time': utc()}
                            atomic(root / 'trials' / config['id'] / 'result.json', record)
                            emit(root, 'numerical_failure', trial=config['id'], reason=str(error))
                    records.append(record)
                    status(root, 'running', completed=sum(r['status'] == 'completed' for r in records),
                           failed=sum(r['status'] == 'failed' for r in records))
                if stage == 1:
                    emit(root, 'stage1_completed', notification=notify('Figure 3: learning-rate stage finished; starting targeted momentum trials.'))
            finalize(root, manifest)
        except BaseException as error:
            state = 'interrupted' if isinstance(error, (KeyboardInterrupt, InterruptedError)) else 'error'
            status(root, state, error=str(error), traceback=traceback.format_exc())
            emit(root, state, error=str(error))
            raise


def supervise(root):
    root = Path(root).resolve()
    manifest = read(root / 'manifest.json')
    with open(Path(manifest['scratch']) / 'supervisor.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        emit(root, 'supervisor_start', notification=notify('Figure 3 GPU search starting: 38 trials, estimated 6–8 hours.'))
        child = subprocess.Popen([sys.executable, '-u', str(root / 'source/experiments.py'),
                                  '--figure3-worker', '--run-dir', str(root)])
        def forward(signum, frame):
            child.send_signal(signum)
        signal.signal(signal.SIGTERM, forward)
        signal.signal(signal.SIGINT, forward)
        stale = False
        while True:
            try:
                code = child.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                current = read(root / 'status.json')
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(current['updated_at'])).total_seconds()
                if current['state'] == 'running' and age > 180 and not stale:
                    emit(root, 'heartbeat_stale', age_seconds=age,
                         notification=notify('Figure 3 has not saved progress for three minutes. It may be stalled.'))
                    stale = True
                elif age <= 180:
                    stale = False
        completed = code == 0 and read(root / 'status.json')['state'] == 'completed'
        if not completed:
            status(root, 'interrupted', exit_code=code)
        message = ('Figure 3 search complete. Updated charts and selected settings are saved.' if completed else
                   'Figure 3 search interrupted. Completed epochs are checkpointed; check status before resuming.')
        emit(root, 'supervisor_exit', exit_code=code, notification=notify(message))
        return code if code else (0 if completed else 1)
