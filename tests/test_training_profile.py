"""Ensure trace summaries do not double-count overlapping GPU activity."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from replication.profile_training import trace_summary, training_state_digests, union_duration


class TrainingProfileTests(unittest.TestCase):
    def test_state_digests_distinguish_weights_gradients_and_moments(self):
        model = torch.nn.Linear(2, 2)
        model.pool = SimpleNamespace(slots=[1, 0], _f=lambda e: e + 100)
        opt = torch.optim.AdamW(model.parameters())
        model(torch.ones(1, 2)).sum().backward()
        opt.step()
        before = training_state_digests(model, opt)
        self.assertEqual(before, training_state_digests(model, opt))
        self.assertEqual(before["slot_expert_uids"], [101, 100])
        model.weight.grad.add_(1)
        after = training_state_digests(model, opt)
        self.assertNotEqual(before["clipped_gradients"], after["clipped_gradients"])
        self.assertEqual(before["parameters"], after["parameters"])
        self.assertEqual(before["optimizer"], after["optimizer"])
        with torch.no_grad():
            model.weight.add_(1)
        self.assertNotEqual(after["parameters"], training_state_digests(model, opt)["parameters"])
        opt.state[model.weight]["exp_avg"].add_(1)
        self.assertNotEqual(after["optimizer"], training_state_digests(model, opt)["optimizer"])

    def test_overlapping_and_nested_intervals_count_once(self):
        self.assertEqual(union_duration([]), 0)
        self.assertEqual(union_duration([(10, 20), (0, 15), (2, 3), (21, 25)]), 24)

    def test_trace_clips_to_step_and_separates_kernels_from_copies(self):
        events = [
            {"name": "measured_training_step", "ph": "X", "ts": 100, "dur": 100},
            {"name": "k1", "cat": "kernel", "ph": "X", "ts": 90, "dur": 30},
            {"name": "k1", "cat": "kernel", "ph": "X", "ts": 110, "dur": 40},
            {"name": "k2", "cat": "kernel", "ph": "X", "ts": 155, "dur": 10},
            {"name": "copy", "cat": "gpu_memcpy", "ph": "X", "ts": 140, "dur": 40},
            {"name": "outside", "cat": "kernel", "ph": "X", "ts": 210, "dur": 100},
            {"name": "metadata", "ph": "M"},
        ]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, "trace.json")
            p.write_text(json.dumps({"traceEvents": events}))
            result = trace_summary(p)
        self.assertEqual(result["kernel_count"], 3)
        self.assertEqual(result["kernel_union_ms"], 0.06)
        self.assertEqual(result["kernel_or_copy_union_ms"], 0.08)
        self.assertEqual(result["top_kernels_ms"], [("k1", 0.06), ("k2", 0.01)])

    def test_missing_cuda_capture_is_not_reported_as_idle_gpu(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, "trace.json")
            p.write_text(json.dumps({"traceEvents": [
                {"name": "measured_training_step", "ph": "X", "ts": 0, "dur": 100}]}))
            with self.assertRaisesRegex(RuntimeError, "no kernels"):
                trace_summary(p)


if __name__ == "__main__":
    unittest.main()
