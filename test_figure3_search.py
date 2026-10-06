"""Search-control and checkpoint tests; no full experiment training."""
import copy
import json
import os
from pathlib import Path
import pickle
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
import experiments as exp
import figure3_search as search


def tiny_model(dropout, seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(torch.nn.Dropout(.2 if dropout else 0),
        torch.nn.Linear(7, 9), torch.nn.ReLU(), torch.nn.Dropout(.5 if dropout else 0), torch.nn.Linear(9, 3))


class SearchTests(unittest.TestCase):
    def setUp(self):
        search.STOP = False
        rng = np.random.default_rng(5)
        self.x = rng.normal(size=(19, 7)).astype(np.float32)
        self.y = exp.one_hot(3, rng.integers(0, 3, size=19))

    def config(self, name='adam', dropout=True):
        return {**exp.figure3_config(name, dropout, .0001, None if name == 'adagrad' else .9, 1),
                'epochs': 3, 'batch_size': 8, 'batches_per_epoch': 3}

    def test_grid_and_shortlist(self):
        configs = exp.figure3_stage1_configs()
        self.assertEqual(len(configs), 22)
        records = [{'config': c, 'status': 'completed', 'mean_last5': float(i), 'final_loss': float(i)} for i, c in enumerate(configs)]
        second = exp.figure3_stage2_configs(records)
        self.assertEqual(len(second), 16)
        self.assertEqual(len({c['id'] for c in configs + second}), 38)
        self.assertEqual({c['momentum'] for c in second}, {.8, .95})
        self.assertTrue(all(c['optimizer'] != 'adagrad' for c in second))
        for name in ['adam', 'sgd_nesterov']:
            self.assertEqual({c['learning_rate'] for c in second if c['optimizer'] == name}, set(exp.FIGURE3_LR_GRID[name][:2]))
        records[0]['status'] = 'failed'
        records[1]['status'] = 'failed'
        records[2]['status'] = 'failed'
        with self.assertRaises(ValueError):
            exp.figure3_stage2_configs(records)

    def test_exact_resume_all_optimizers_with_dropout(self):
        for name in ['adam', 'adagrad', 'sgd_nesterov']:
            c = self.config(name)
            model, optimizer, shapes, backward, shuffle = search.start_trial(c, 'cpu', tiny_model)
            h, b = search.run_epoch(optimizer, shapes, backward, shuffle, self.x, self.y, 8)
            saved = pickle.loads(pickle.dumps(search.checkpoint(c, optimizer, shapes, shuffle, [h], b, [1.], 'hash', 'cpu')))
            expected = search.run_epoch(optimizer, shapes, backward, shuffle, self.x, self.y, 8)
            expected_weights = optimizer.weights.copy()
            model, optimizer, shapes, backward, shuffle = search.restore(saved, c, 'hash', 'cpu', tiny_model)
            actual = search.run_epoch(optimizer, shapes, backward, shuffle, self.x, self.y, 8)
            self.assertEqual(actual, expected)
            np.testing.assert_array_equal(optimizer.weights, expected_weights)
            with self.assertRaises(ValueError):
                search.restore(saved, c, 'other-hash', 'cpu', tiny_model)

    def test_loop_matches_existing_training(self):
        c = self.config()
        model, optimizer, shapes, backward, shuffle = search.start_trial(c, 'cpu', tiny_model)
        params = exp.unpack(optimizer.weights.copy(), shapes)
        original, batches = exp.train(backward, params, optimizer, self.x, self.y, 2, batch_size=8, seed=0, record_batches=True)
        expected = optimizer.weights.copy()
        model, optimizer, shapes, backward, shuffle = search.start_trial(c, 'cpu', tiny_model)
        records = [search.run_epoch(optimizer, shapes, backward, shuffle, self.x, self.y, 8) for _ in range(2)]
        np.testing.assert_array_equal(original, [h for h, b in records])
        np.testing.assert_array_equal(batches, np.concatenate([b for h, b in records]))
        np.testing.assert_array_equal(expected, optimizer.weights)

    def test_stop_nonfinite_and_final_checkpoint_recovery(self):
        c = self.config()
        model, opt, shapes, backward, shuffle = search.start_trial(c, 'cpu', tiny_model)
        search.STOP = True
        with self.assertRaises(InterruptedError):
            search.run_epoch(opt, shapes, backward, shuffle, self.x, self.y, 8)
        search.STOP = False
        with self.assertRaises(FloatingPointError):
            search.run_epoch(opt, shapes, lambda *a: (float('nan'), []), shuffle, self.x, self.y, 8)
        history, batches = [], []
        for _ in range(3):
            h, b = search.run_epoch(opt, shapes, backward, shuffle, self.x, self.y, 8)
            history.append(h); batches.extend(b)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); scratch = root / 'scratch'; scratch.mkdir()
            manifest = {'scratch': str(scratch)}
            search.atomic(root / 'manifest.json', manifest)
            saved = search.checkpoint(c, opt, shapes, shuffle, history, batches, [1.] * 3, search.digest(root / 'manifest.json'), 'cpu')
            search.atomic(scratch / (c['id'] + '.pkl'), saved, 'pickle')
            with patch.object(search, 'run_epoch', side_effect=AssertionError('must not retrain')):
                result = search.run_trial(root, manifest, c, self.x, self.y, 'cpu', tiny_model)
            self.assertEqual(result['epochs'], 3)
            self.assertFalse((scratch / (c['id'] + '.pkl')).exists())
            search.load_record(root, c)

    def test_supervisor_detects_stall_and_abrupt_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); scratch = root / 'scratch'; scratch.mkdir()
            search.atomic(root / 'manifest.json', {'scratch': str(scratch)})
            search.atomic(root / 'status.json', {'state': 'running', 'updated_at': '2000-01-01T00:00:00+00:00'})
            child = Mock(); child.wait.side_effect = [subprocess.TimeoutExpired('worker', 30), -9]
            with patch.object(search.subprocess, 'Popen', return_value=child), patch.object(search, 'notify', return_value={'returncode': 0}) as notice, patch.object(search.signal, 'signal'):
                self.assertEqual(search.supervise(root), -9)
                self.assertEqual(notice.call_count, 3)
            self.assertEqual(search.read(root / 'status.json')['state'], 'interrupted')

    def test_epoch_seeded_trial_resume_matches_uninterrupted(self):
        c = self.config()
        with tempfile.TemporaryDirectory() as tmp:
            def setup(name):
                root = Path(tmp) / name; scratch = root / 'scratch'; scratch.mkdir(parents=True)
                manifest = {'scratch': str(scratch)}
                search.atomic(root / 'manifest.json', manifest)
                return root, manifest
            original, manifest = setup('original')
            expected = search.run_trial(original, manifest, c, self.x, self.y, 'cpu', tiny_model)
            resumed, manifest = setup('resumed')
            run_epoch = search.run_epoch
            count = [0]
            def interrupt(*args, **kwargs):
                count[0] += 1
                if count[0] == 2:
                    raise InterruptedError('simulated process interruption')
                return run_epoch(*args, **kwargs)
            with patch.object(search, 'run_epoch', side_effect=interrupt):
                with self.assertRaises(InterruptedError):
                    search.run_trial(resumed, manifest, c, self.x, self.y, 'cpu', tiny_model)
            actual = search.run_trial(resumed, manifest, c, self.x, self.y, 'cpu', tiny_model)
            self.assertEqual(actual['history_sha256'], expected['history_sha256'])
            self.assertEqual(actual['batch_sha256'], expected['batch_sha256'])

    def test_finalization_and_immutable_shortlist(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); asset = root / 'assets'; asset.mkdir()
            originals = {}
            for name in ['figure_3_recreation.png', 'figure_2_recreation_tuned_v1.png']:
                path = asset / name; path.write_bytes(b'original')
                originals[name] = {'path': str(path), 'sha256': search.digest(path)}
            manifest = {'stage1': exp.figure3_stage1_configs(), 'original_assets': originals, 'selection': 'test'}
            records = []
            def create(configs):
                for c in configs:
                    out = root / 'trials' / c['id']
                    batches = np.repeat(np.linspace(2, .1 + len(records) * .005, 45), 391)
                    history = batches.reshape(45, 391).mean(axis=1)
                    search.atomic(out / 'history.npy', history, 'npy')
                    search.atomic(out / 'batch_losses.npy', batches, 'npy')
                    r = {'config': c, 'status': 'completed', 'mean_last5': float(history[-5:].mean()), 'final_loss': float(history[-1]),
                         'history_sha256': search.digest(out / 'history.npy'), 'batch_sha256': search.digest(out / 'batch_losses.npy')}
                    search.atomic(out / 'result.json', r); records.append(r)
            create(manifest['stage1'])
            second = search.stage2(root, records)
            self.assertEqual(second, search.stage2(root, records))
            with patch.object(exp, 'figure3_stage2_configs', return_value=[]):
                with self.assertRaises(ValueError):search.stage2(root, records)
            create(second)
            search.finalize(root, manifest)
            self.assertEqual((asset / 'figure_2_recreation_tuned_v1.png').read_bytes(), b'original')
            self.assertEqual(search.read(root / 'status.json')['state'], 'completed')
            self.assertEqual(len(search.read(root / 'search_results.json')['selected']), 6)
            Path('/tmp/figure3-search-layout-check.png').write_bytes((root / 'figure_3_recreation.png').read_bytes())
            (root / 'trials' / second[-1]['id'] / 'result.json').unlink()
            with self.assertRaises(ValueError):search.finalize(root, manifest)


if __name__ == '__main__':
    unittest.main()
