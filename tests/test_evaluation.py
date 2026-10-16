"""Predictive scoring must describe generation, not the learning objective."""

from dataclasses import asdict
import importlib.util
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from minagi import store
from minagi.plasticity import Plasticity
from minagi.recur import RecurCoder, RecurConfig, PREDICTIVE_METRIC
from minagi.stream import FileReader, Reader, FolderEvaluator, Evaluator, _score_windows
import test_causality as fixtures


class EvaluationTests(unittest.TestCase):
    model = fixtures.CausalityTests.model

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.original_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    @classmethod
    def tearDownClass(cls):
        torch.backends.cuda.matmul.allow_tf32 = cls.original_tf32

    def scripted_model(self):
        cfg = RecurConfig(vocab_size=3, d_model=8, n_head=2, d_ff=16,
                          block=128, n_prelude=0, n_recur=1, n_coda=0,
                          max_steps=2, halt_thresh=0.4, ponder_beta=0.7,
                          halt_prior=0.4, train_steps_mean=0)
        model = RecurCoder(cfg).eval()
        with torch.no_grad():
            model.halt.weight.zero_()
            model.halt.bias.zero_()  # p_1 = p_2 = 1/2
        calls = []

        def head(_module, _args, output):
            values = [4., 0., 0.] if len(calls) % 2 == 0 else [0., 8., 0.]
            result = output.new_tensor(values).expand_as(output)
            calls.append(result)
            return result

        model.head.register_forward_hook(head)
        return model

    def test_manual_training_objective_is_preserved(self):
        m = self.scripted_model()
        x = torch.zeros(1, 7, dtype=torch.long)
        with torch.no_grad():
            logits, objective = m(x, x)
        ce1 = F.cross_entropy(torch.tensor([[4., 0., 0.]]), torch.tensor([0]))
        ce2 = F.cross_entropy(torch.tensor([[0., 8., 0.]]), torch.tensor([0]))
        prior = torch.tensor([0.4, 0.24]) / 0.64
        kl = (0.5 * (torch.tensor(0.5).log() - prior.log())).sum()
        torch.testing.assert_close(objective, 0.5 * (ce1 + ce2) + 0.7 * kl)
        torch.testing.assert_close(logits[0, 0], torch.tensor([2., 4., 0.]))
        predictive = m.predictive_loss(x, x)
        torch.testing.assert_close(predictive, ce1)
        self.assertGreater(abs(float(objective - predictive)), 3)

    def test_file_reader_uses_generation_predictions(self):
        m = self.scripted_model()
        data = np.zeros(8, dtype=np.uint16)
        x = torch.from_numpy(data[:-1].astype(np.int64)).unsqueeze(0)
        with torch.no_grad():
            logits, _ = m(x)
        expected = F.cross_entropy(logits.flatten(0, 1), x.flatten())
        reader = FileReader(m, data, "fixture", 7, 128, "cpu")
        torch.testing.assert_close(reader.step(learn=False), expected)

    def test_prediction_matches_generation_with_guards_disabled(self):
        m = self.model()
        text = torch.randint(0, 265, (1, 18))
        captured = []

        def forced_next(logits, history, *_args, **_kwargs):
            i = len(captured)
            captured.append(logits.detach().clone())
            return text[:, 5+i:6+i]

        with torch.no_grad(), patch("minagi.recur.pick_next", side_effect=forced_next):
            m.generate(text[:, :5], max_new_tokens=12, adapt_strength=0)
        expected = F.cross_entropy(torch.cat(captured), text[:, 5:17].flatten())
        actual = m.predictive_loss(text[:, :16], text[:, 1:17], reduction="none")
        torch.testing.assert_close(actual[4:].mean(), expected, atol=2e-6, rtol=1e-6)

    def test_full_chunk_and_token_scoring_match(self):
        for freeze, capacity in itertools.product((False, True), (0, 1.5)):
            m = self.model(freeze=freeze, capacity=capacity)
            with torch.no_grad():
                m.halt.weight.normal_(0, 2)
                m.halt.bias.zero_()
            m.cfg.halt_thresh = 0.6
            x, y = torch.randint(0, 265, (2, 1, 29))
            expected = m.predictive_loss(x, y, reduction="none")
            for width in (1, 5, 29):
                caches = m.empty_caches()
                actual = torch.cat([
                    m.predictive_loss(x[:, s:s+width], y[:, s:s+width],
                                      caches, s, reduction="none")
                    for s in range(0, x.shape[1], width)])
                with self.subTest(freeze=freeze, capacity=capacity, width=width):
                    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-6)

    def test_predictive_scoring_ignores_training_penalties_and_sampling(self):
        m = self.model()
        x = torch.randint(0, 265, (1, 11))
        expected = m.predictive_loss(x, x)
        m.train()
        m.cfg.train_steps_mean = 0.01
        m.cfg.ponder_beta = 10000
        with patch.object(m, "pool_balance", return_value=torch.tensor(1000.)):
            r = Reader(m, np.arange(60, dtype=np.uint16), chunk=11, context=32)
            r.pos = 0
            loss = r.step(learn=False, aux_weight=1000)
            manual = m.predictive_loss(torch.arange(11).unsqueeze(0),
                                       torch.arange(1, 12).unsqueeze(0))
            torch.testing.assert_close(loss, manual)
        torch.testing.assert_close(m.predictive_loss(x, x), expected)
        self.assertTrue(m.training)
        self.assertFalse(loss.requires_grad)
        self.assertTrue(all(p.grad is None for p in m.parameters()))
        with self.assertRaises(ValueError):
            m.predictive_loss(torch.zeros(1, 129, dtype=torch.long),
                              torch.zeros(1, 129, dtype=torch.long))
        self.assertTrue(m.training)

    def test_file_boundaries_and_last_single_token_are_scored(self):
        m = self.model()
        data = np.arange(84, dtype=np.uint16)
        total = 0.
        for s in range(0, len(data) - 1, 31):
            a = torch.tensor(data[s:min(s + 32, len(data))].astype(np.int64)).unsqueeze(0)
            total += float(m.predictive_loss(a[:, :-1], a[:, 1:], reduction="sum"))
        for width in (1, 7, 31, 40):
            r = FileReader(m, data, "fixture", width, 31, "cpu")
            score, count = 0., 0
            while not r.done():
                loss = r.step(learn=False)
                self.assertIsNotNone(loss)
                score += float(loss) * r.last_count
                count += r.last_count
            self.assertEqual(count, 83)
            self.assertAlmostEqual(score / count, total / 83, places=6)

    def test_folder_weights_by_predictions_and_preserves_mode(self):
        m = self.scripted_model()
        # Use byte ids 0..2 directly via an injected text encoder.
        arrays = {"a": np.zeros(33, dtype=np.uint16),
                  "b": np.ones(8, dtype=np.uint16)}
        with tempfile.TemporaryDirectory() as d:
            for name in arrays:
                Path(d, name).write_text("fixture\n")
            expected = sum(float(m.predictive_loss(
                torch.tensor(a[:-1].astype(np.int64)).unsqueeze(0),
                torch.tensor(a[1:].astype(np.int64)).unsqueeze(0), reduction="sum"))
                for a in arrays.values()) / 39
            for training, width in itertools.product((False, True), (1, 7, 32)):
                m.train(training)
                ev = FolderEvaluator(m, d, width, 32, "cpu")
                ev._as_stream = lambda p: arrays[Path(p).name]
                result = ev.run(chunks=100)
                self.assertAlmostEqual(result["all"], expected, places=6)
                self.assertEqual(m.training, training)

    def test_packed_evaluator_scores_predictive_ce(self):
        m = self.scripted_model()
        with tempfile.TemporaryDirectory() as d:
            np.zeros(300, dtype=np.uint16).tofile(Path(d, "val.bin"))
            m.eval()
            result = Evaluator(m, [d], 7, 31, "cpu").run(chunks=12)
            expected = float(m.predictive_loss(torch.zeros(1, 7, dtype=torch.long),
                                              torch.zeros(1, 7, dtype=torch.long)))
            self.assertAlmostEqual(result[d], expected, places=6)
            self.assertFalse(m.training)

    def test_window_error_is_invariant_to_chunk_partition(self):
        a = [(1., 10, 0), (3., 10, 1), (5., 10, 2)]
        b = [(l, n, w) for l, _, w in a for n in (3, 7)]
        self.assertEqual(_score_windows(a), _score_windows(b))
        mean, se = _score_windows(a)
        self.assertEqual(mean, 3.)
        self.assertAlmostEqual(se, 2 / math.sqrt(3))
        self.assertEqual(_score_windows([(1., 9, 0), (11., 1, 1)])[0], 2.)

    def test_checkpoint_metric_prevents_historical_score_comparison(self):
        for paged in (False, True):
            with tempfile.TemporaryDirectory() as parent:
                d = str(Path(parent, "weights"))
                if paged:
                    from minagi.create import create
                    from train import build_paged
                    create(d, d_model=32, n_head=4, trunk_d_ff=64, experts=16,
                           resident=4, d_ff=64, top_k=2, block=128, max_steps=3,
                           verbose=False)
                    m, _, _, _ = build_paged(d, torch.device("cpu"), resident=4,
                                             ram_capacity=16, read_only=True)
                else:
                    m = self.scripted_model()
                store.save(m, d, cfg=asdict(m.cfg), val=0.1)
                self.assertTrue(math.isinf(store.best_val(d, metric=PREDICTIVE_METRIC)))
                store.save(m, d, cfg=asdict(m.cfg), val=2., val_metric=PREDICTIVE_METRIC)
                self.assertEqual(store.best_val(d, metric=PREDICTIVE_METRIC), 2.)
                self.assertEqual(json.loads(Path(d, "manifest.json").read_text())["val_metric"],
                                 PREDICTIVE_METRIC)

    def test_controller_seeding_excludes_historical_objective(self):
        old = "train loss 0.5 evidence t +3 over 10\nheld-out loss 0.8\n"
        new = "train objective 0.5 evidence t +3 over 10\nheld-out predictive CE 1.2\n"
        self.assertEqual(Plasticity.log_readings(old + new), [1.2])

    def test_metric_migration_preserves_lr_scale_but_discards_fits(self):
        controller = Plasticity()
        for i in range(30):
            controller.observe(2. - i * 0.001, 0.01)
        controller.scale = 0.125
        state = controller.state()
        self.assertTrue(Plasticity.restore(state).has_fits())
        restored = Plasticity.restore(state, reset_fits=True)
        self.assertEqual(restored.scale, 0.125)
        self.assertFalse(restored.has_fits())

    def test_invalid_reader_bounds_and_empty_evaluation_fail_clearly(self):
        m = self.model()
        for cls in (Reader, FileReader):
            for chunk, context in ((0, 32), (7, 129), (7, -1)):
                with self.assertRaises(ValueError):
                    cls(m, np.arange(12, dtype=np.uint16), "fixture", chunk,
                        context, "cpu")
        with tempfile.TemporaryDirectory() as d:
            m.eval()
            with self.assertRaisesRegex(ValueError, "no next-token predictions"):
                FolderEvaluator(m, d, 7, 32, "cpu").run()
            self.assertFalse(m.training)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "plotting dependency unavailable")
    def test_plots_parse_new_scores_and_keep_metric_eras_separate(self):
        from tools.plot_progress import parse
        import train
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, "samples.txt")
            p.write_text("step 0   1.0M of 10M characters   1 min   16 experts\n"
                         "train loss 0.5\nheld-out loss 0.8 +/-0.01\n")
            self.assertEqual(parse(p)[0]["val"], 0.8)
            with patch.object(train, "refresh_plots"):
                train.write_samples(str(p), 1, 2e6, 2, [], val=1.2, se=0,
                                    train=0.6, corpus=10e6, experts=16)
            rows = parse(p)
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0]["predictive"])
            self.assertEqual(rows[0]["val"], 1.2)
            self.assertEqual(rows[0]["train"], 0.6)
            self.assertNotIn("gap", p.read_text())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_bf16_predictive_ce_matches_generation(self):
        m = self.model(device="cuda")
        x, y = torch.randint(0, 265, (2, 1, 17), device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = m(x)
            loss = m.predictive_loss(x, y)
        expected = F.cross_entropy(logits.flatten(0, 1).float(), y.flatten())
        torch.testing.assert_close(loss, expected)
        self.assertEqual(loss.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
