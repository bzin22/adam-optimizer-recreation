"""Checks for resume, original-loop parity, selection, and preserved curves."""
import copy
from pathlib import Path
import pickle
import subprocess
import tempfile
import unittest
from unittest.mock import patch, Mock

import numpy as np
import figure2a_search as search
import mlp
from train import train, load_MNIST
from utils import one_hot


class SearchTests(unittest.TestCase):
    def setUp(self):
        search.STOP = False
        rng = np.random.default_rng(22)
        self.x = rng.normal(size=(19, 7)).astype(np.float32)
        self.y = one_hot(3, rng.integers(0, 3, size=19))
        self.params = search.init_params(0, input_dim=7, hidden=9, classes=3)

    def config(self, name):
        return {"optimizer": name, "learning_rate": search.BASE_RATES[name],
                "seed": 0, "batch_size": 8, "epochs": 200}

    def test_new_loop_matches_original(self):
        for name in search.BASE_RATES:
            with self.subTest(optimizer=name):
                config = self.config(name)
                original = search.make_optimizer(config, search.pack(self.params))
                mlp.reseed(0)
                reference = train(mlp.mlp_backward, self.params, original, self.x, self.y,
                                  3, batch_size=8, seed=0)
                optimizer, shapes, shuffle = search.start_trial(config, self.params)
                history = [search.run_epoch(optimizer, shapes, shuffle, self.x, self.y,
                                            batch_size=8)[0] for _ in range(3)]
                np.testing.assert_array_equal(history, reference)
                np.testing.assert_array_equal(optimizer.weights, original.weights)

    def test_serialized_resume_is_exact_for_both_optimizers(self):
        for name in search.BASE_RATES:
            with self.subTest(optimizer=name):
                config = self.config(name)
                optimizer, shapes, shuffle = search.start_trial(config, self.params)
                loss, _ = search.run_epoch(optimizer, shapes, shuffle, self.x, self.y, 8)
                saved = pickle.loads(pickle.dumps(search.checkpoint(
                    config, optimizer, shapes, shuffle, [loss], [1.0], "source-hash")))
                uninterrupted = [search.run_epoch(optimizer, shapes, shuffle, self.x, self.y, 8)[0]
                                 for _ in range(2)]
                expected_state = copy.deepcopy(optimizer.__dict__)
                expected_rng = copy.deepcopy(mlp._rng.bit_generator.state)
                resumed, shapes, shuffle, history, timings = search.restore(saved, config, "source-hash")
                continued = [search.run_epoch(resumed, shapes, shuffle, self.x, self.y, 8)[0]
                             for _ in range(2)]
                np.testing.assert_array_equal(continued, uninterrupted)
                self.assertEqual(history, [loss])
                self.assertEqual(timings, [1.0])
                for key, value in expected_state.items():
                    np.testing.assert_array_equal(resumed.__dict__[key], value)
                self.assertEqual(mlp._rng.bit_generator.state, expected_rng)
                with self.assertRaises(ValueError):
                    search.restore(saved, config, "different-source")

    def test_diagnostics_do_not_change_training(self):
        config = self.config("adam")
        results = []
        for enabled in (False, True):
            opt, shapes, shuffle = search.start_trial(config, self.params)
            loss, detail = search.run_epoch(opt, shapes, shuffle, self.x, self.y, 8, diagnostic=enabled)
            results.append((loss, opt.weights.copy()))
            if enabled:
                self.assertEqual(len(detail["update_rms"]), 6)
        self.assertEqual(results[0][0], results[1][0])
        np.testing.assert_array_equal(results[0][1], results[1][1])

    def test_resume_recovers_last_checkpoint_before_result_publication(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scratch = root / "scratch"
            scratch.mkdir()
            config = {**self.config("adam"), "id": "test", "epochs": 3, "multiplier": 0.3}
            manifest = {"scratch": str(scratch), "trials": [config]}
            search.atomic(root / "manifest.json", manifest)
            optimizer, shapes, shuffle = search.start_trial(config, self.params)
            history = [search.run_epoch(optimizer, shapes, shuffle, self.x, self.y, 8)[0]
                       for _ in range(3)]
            saved = search.checkpoint(config, optimizer, shapes, shuffle, history, [1., 1., 1.],
                                      search.digest(root / "manifest.json"))
            search.atomic(scratch / "test.pkl", saved, "pickle")
            search.atomic(root / "trials/test/history.npy", np.asarray(history[:2]), "npy")
            search.run_trial(root, manifest, config, self.x, self.y)
            np.testing.assert_array_equal(np.load(root / "trials/test/history.npy"), history)
            self.assertEqual(search.read_json(root / "trials/test/result.json")["status"], "completed")

    def test_stop_and_nonfinite_are_not_completed_trials(self):
        opt, shapes, shuffle = search.start_trial(self.config("adam"), self.params)
        before = opt.weights.copy()
        search.STOP = True
        with self.assertRaises(InterruptedError):
            search.run_epoch(opt, shapes, shuffle, self.x, self.y, 8)
        np.testing.assert_array_equal(opt.weights, before)
        search.STOP = False
        with patch.object(mlp, "mlp_backward", return_value=(float("nan"), [np.zeros_like(p) for p in self.params])):
            with self.assertRaises(FloatingPointError):
                search.run_epoch(opt, shapes, shuffle, self.x, self.y, 8)

    def test_grid_and_selection(self):
        configs = search.configs()
        self.assertEqual(len(configs), 26)
        self.assertEqual(len({c["id"] for c in configs}), 26)
        self.assertEqual([c["optimizer"] for c in configs[:2]], ["adam", "adagrad"])
        records = []
        for i, c in enumerate(configs):
            records.append({"config": c, "status": "completed", "epochs": 200,
                            "mean_last20": 2.0 + i, "final_loss": 1.0})
        records[0]["final_loss"] = 0.001
        records[2]["mean_last20"] = 1.0
        records[2]["final_loss"] = 0.9
        self.assertEqual(search.choose_best(records, "adam"), records[2])
        records[2]["status"] = "failed"
        self.assertEqual(search.choose_best(records, "adam"), records[0])

    def test_supervisor_records_stall_and_abrupt_exit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scratch = root / "scratch"
            scratch.mkdir()
            search.atomic(root / "manifest.json", {"scratch": str(scratch)})
            search.atomic(root / "status.json", {"state": "running", "pid": 123,
                "updated_at": "2000-01-01T00:00:00+00:00", "epoch": 4})
            child = Mock()
            child.wait.side_effect = [subprocess.TimeoutExpired("training", 30), -9]
            with patch.object(search.subprocess, "Popen", return_value=child), \
                    patch.object(search, "notify", return_value={"returncode": 0}) as notice, \
                    patch.object(search.signal, "signal"):
                self.assertEqual(search.supervise(root), -9)
                self.assertEqual(notice.call_count, 3)
            result = search.read_json(root / "status.json")
            self.assertEqual(result["state"], "interrupted")
            self.assertEqual(result["epoch"], 4)

    @unittest.skipUnless((Path(__file__).resolve().parent / "data/MNIST/raw/train-images-idx3-ubyte").exists(),
                         "Requires the local MNIST training data")
    def test_data_loader_matches_existing_preprocessing(self):
        inputs, labels = search.load_data(Path(__file__).resolve().parent / "data/MNIST/raw")
        original_inputs, original_labels = load_MNIST()
        np.testing.assert_array_equal(inputs, original_inputs)
        np.testing.assert_array_equal(labels, original_labels)

    def test_finalize_preserves_other_series_and_original_figure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "results"
            root.mkdir()
            repo = Path(temp) / "repo"
            scratch = Path(temp) / "scratch"
            scratch.mkdir()
            originals = {}
            parent = Path(__file__).resolve().parent / "results/figure2a_search_20261002_v1/baseline"
            for name in search.BASELINES:
                p = root / "baseline" / f"history_mlp_{name}.npy"
                search.atomic(p, (parent / p.name).read_bytes(), "bytes")
                originals[name] = {"path": str(p), "sha256": search.digest(p)}
            figure = root / "original.png"
            figure.write_bytes(b"preserve-original-file")
            manifest = {"trials": search.configs(), "baseline": originals, "repo": str(repo),
                        "scratch": str(scratch), "selection": "mean of final 20 epochs",
                        "original_figure": {"path": str(figure), "sha256": search.digest(figure)}}
            for c in manifest["trials"]:
                out = root / "trials" / c["id"]
                values = np.load(root / "baseline" / f"history_mlp_{c['optimizer']}.npy")
                search.atomic(out / "history.npy", values, "npy")
                search.atomic(out / "result.json", {"config": c, "status": "completed", "epochs": 200,
                    "mean_last20": float(values[-20:].mean()), "final_loss": float(values[-1]),
                    "history_sha256": search.digest(out / "history.npy")})
            search.finalize(root, manifest)
            self.assertEqual(figure.read_bytes(), b"preserve-original-file")
            for name in ["sgd", "rms", "adadelta"]:
                self.assertEqual(search.digest(root / "chart_data" / f"history_mlp_{name}.npy"), originals[name]["sha256"])
            self.assertEqual(search.read_json(root / "status.json")["state"], "completed")
            check_image = Path('/tmp/figure2a-search-plot-layout-check.png')
            check_image.write_bytes((root / 'figure_2_recreation_tuned_v1.png').read_bytes())
            # An incomplete search must never silently produce a final chart.
            (root / "trials" / manifest["trials"][0]["id"] / "result.json").unlink()
            with self.assertRaises(ValueError):
                search.finalize(root, manifest)


if __name__ == "__main__":
    unittest.main(verbosity=2)
