"""CPU-only guards against mixing incompatible optimizer experiment results."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from replication import run_muon_screen as protocol


class StudyProtocolTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name)
        replacement=patch.object(protocol,'ROOT',self.root)
        replacement.start()
        self.addCleanup(replacement.stop)
        self.source=self.root/'measured.py'
        self.source.write_text('measured source\n')
        trace=[dict(objective=1./(i+1),gradient_norm_before_clip=2.+i,routing_seed=100+i) for i in range(8)]
        self.report=dict(status='passed',external_training_power={},deterministic_algorithms_enabled=True,
                         fixed_sonic_reduction=dict(kwargs={'BLOCK_H':256,'BLOCK_K':32},num_warps=4,num_stages=4),
                         sources={str(self.source):hashlib.sha256(self.source.read_bytes()).hexdigest()},
                         configuration=dict(optimizer='muon',lr=3e-5,seed=42,steps=256,fingerprint=False),
                         training_window_indices=list(range(8)),trace=trace)
        oracle=self.root/'runs/muon-deterministic-repro-fixed/a/muon-lr3e-05-seed42-steps8.json'
        oracle.parent.mkdir(parents=True)
        oracle.write_text(json.dumps(self.report))

    def test_matching_protocol_and_prefix_accepted(self):
        protocol.validate_learning_protocol(self.report,False)

    def test_uncontrolled_or_different_kernel_results_rejected(self):
        for field,value in (('deterministic_algorithms_enabled',False),
                            ('fixed_sonic_reduction',dict(kwargs={'BLOCK_H':512,'BLOCK_K':32},num_warps=8,num_stages=4))):
            with self.subTest(field=field):
                report=copy.deepcopy(self.report)
                report[field]=value
                with self.assertRaises(RuntimeError):
                    protocol.validate_learning_protocol(report,False)

    def test_prefix_changes_are_not_hidden_by_tolerance(self):
        for field in ('objective','gradient_norm_before_clip','routing_seed'):
            with self.subTest(field=field):
                report=copy.deepcopy(self.report)
                report['trace'][4][field]+=1e-6
                with self.assertRaisesRegex(AssertionError,'prefix diverged'):
                    protocol.validate_learning_protocol(report,False)
        report=copy.deepcopy(self.report)
        report['training_window_indices'][0]=99
        with self.assertRaisesRegex(AssertionError,'data prefix'):
            protocol.validate_learning_protocol(report,False)

    def test_cached_trial_cannot_survive_source_change(self):
        path=self.root/'muon-lr3e-05-seed42-steps256.json'
        path.write_text(json.dumps(self.report))
        # The valid cache returns without loading a model or starting NVML.
        self.assertEqual(protocol.trial(self.root,'muon',3e-5,steps=256),self.report)
        self.source.write_text('different implementation\n')
        with self.assertRaisesRegex(RuntimeError,'source differs'):
            protocol.trial(self.root,'muon',3e-5,steps=256)

    def test_truncated_prefix_cannot_pass_zip_comparison(self):
        for field in ('trace','training_window_indices'):
            with self.subTest(field=field):
                report=copy.deepcopy(self.report)
                report[field]=report[field][:7]
                with self.assertRaisesRegex(AssertionError,'incomplete training prefix'):
                    protocol.validate_learning_protocol(report,False)


if __name__=='__main__':
    unittest.main()
