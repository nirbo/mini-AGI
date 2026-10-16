import unittest

from replication.gpu_monitor import summarize_samples


class PowerSummaryTests(unittest.TestCase):
    def test_integration_clips_to_workload_excludes_loading(self):
        samples = [{"monotonic": t, "power_w": p} for t, p in ((0, 600), (1, 100), (2, 200), (3, 600))]
        result = summarize_samples(samples, 1.25, 1.75, new_tokens=10)
        self.assertEqual(result["average_board_power_w"], 150)
        self.assertEqual(result["peak_sampled_board_power_w"], 175)
        self.assertEqual(result["estimated_board_energy_j"], 75)
        self.assertEqual(result["estimated_board_joules_per_new_token"], 7.5)
        self.assertEqual(result["sample_count_inside"], 0)

    def test_variable_sampling_intervals_use_time_weights(self):
        samples = [{"monotonic": t, "power_w": p} for t, p in ((0, 100), (1, 300), (4, 300))]
        result = summarize_samples(samples, 0, 4)
        self.assertEqual(result["estimated_board_energy_j"], 1100)
        self.assertEqual(result["average_board_power_w"], 275)
        self.assertEqual(result["max_sample_gap_seconds"], 3)

    def test_insufficient_samples_and_invalid_windows_rejected(self):
        samples = [{"monotonic": 1, "power_w": 200}, {"monotonic": 2, "power_w": 200}]
        for start, end in ((0, 2), (1, 3), (2, 1)):
            with self.assertRaises(ValueError):
                summarize_samples(samples, start, end)
        with self.assertRaises(ValueError):
            summarize_samples(samples, 1, 2, new_tokens=0)


if __name__ == "__main__":
    unittest.main()
