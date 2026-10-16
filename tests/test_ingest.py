"""Regression coverage for upstream ce31731's UTF-8 probe and verdict cache."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from minagi import ingest
from minagi.tokenizer import ByteTokenizer


class IngestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="minagi-ingest-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.corpus = self.root / "corpus"
        self.corpus.mkdir()
        self.cache = str(self.root / "index.json")

    def collect(self, paths=None, **kwargs):
        return ingest.collect(paths or [str(self.corpus)], cache=self.cache,
                              quiet=True, **kwargs)

    def test_utf8_characters_cut_by_probe_are_kept(self):
        p = self.corpus / "unicode.txt"
        for char in ("é", "●", "漢", "🙂"):
            for cut in range(1, len(char.encode())):
                p.write_bytes(b"a" * (4096 - cut) + char.encode() + b"tail")
                with self.subTest(char=char, cut=cut):
                    self.assertIsNone(ingest.why_not_text(p))
                    self.assertTrue(ingest.looks_like_text(p))

    def test_actual_truncated_and_interior_invalid_utf8_are_rejected(self):
        p = self.corpus / "invalid.txt"
        for raw in (b"a" * 4095 + b"\xc3", b"a" * 4093 + b"\xf0\x9f\x99",
                    b"a" * 40 + b"\xff" + b"a" * 5000):
            p.write_bytes(raw)
            self.assertEqual(ingest.why_not_text(p), "not UTF-8")

    def test_rejection_reasons_and_ascii_boundary(self):
        for name, raw, reason in (("empty.txt", b"", "empty"),
                                  ("nul.txt", b"text\0text", "with a NUL byte"),
                                  ("binary.PNG", b"text", "of a binary type"),
                                  ("control.txt", b"\x01" * 40, "mostly unprintable")):
            p = self.corpus / name
            p.write_bytes(raw)
            self.assertEqual(ingest.why_not_text(p), reason)
        self.assertEqual(ingest.why_not_text(self.corpus / "missing"), "unreadable")
        for size in (4095, 4096, 4097):
            p = self.corpus / "ascii.txt"
            p.write_bytes(b"a" * size)
            self.assertTrue(ingest.looks_like_text(p))

    def test_same_name_same_size_changes_are_rechecked_both_directions(self):
        p = self.corpus / "changing.txt"
        p.write_bytes(b"ordinary text")
        self.assertEqual(self.collect(), [str(p)])
        for raw, expected in ((b"\0" + b"x" * 12, []),
                              (b"ordinary text", [str(p)])):
            previous = p.stat().st_mtime_ns
            p.write_bytes(raw)
            os.utime(p, ns=(previous + 1, previous + 1))
            self.assertEqual(self.collect(), expected)

    def test_unchanged_verdicts_reused_from_memory_and_disk(self):
        p = self.corpus / "stable.txt"
        p.write_text("ordinary text\n")
        self.assertEqual(self.collect(), [str(p)])
        with patch.object(ingest, "why_not_text", side_effect=AssertionError("reopened unchanged text")):
            self.assertEqual(self.collect(), [str(p)])
            ingest._HELD.pop(self.cache, None)
            self.assertEqual(self.collect(), [str(p)])

    def test_old_and_malformed_caches_are_resurveyed(self):
        p = self.corpus / "unicode.txt"
        p.write_bytes(b"a" * 4095 + "é".encode() + b"tail")
        stamp = [p.stat().st_size, p.stat().st_mtime_ns, "not UTF-8"]
        for data in ({"entries": {"old": []}},
                     {"version": 1, "files": {str(p): stamp}},
                     {"version": 2, "files": []}, []):
            Path(self.cache).write_text(json.dumps(data))
            ingest._HELD.pop(self.cache, None)
            with patch.object(ingest, "why_not_text", wraps=ingest.why_not_text) as probe:
                self.assertEqual(self.collect(), [str(p)])
                probe.assert_called_once_with(str(p))

    def test_deleted_files_pruned_without_evicting_other_roots(self):
        a, b = self.corpus / "a.txt", self.root / "other" / "b.txt"
        a.write_text("first document")
        b.parent.mkdir()
        b.write_text("second document")
        self.collect()
        self.collect([str(b.parent)])
        a.unlink()
        self.assertEqual(self.collect(), [])
        files = json.loads(Path(self.cache).read_text())["files"]
        self.assertNotIn(str(a), files)
        self.assertIn(str(b), files)

    def test_overlapping_roots_do_not_duplicate_files(self):
        p = self.corpus / "a.txt"
        p.write_text("ordinary text")
        self.assertEqual(self.collect([str(self.corpus), str(p), str(p)]), [str(p)])

    def test_skip_reporting_can_be_silenced(self):
        p = self.corpus / "bad.txt"
        p.write_bytes(b"hello\0world")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(ingest.collect([str(self.corpus)], cache=None), [])
        self.assertIn("with a NUL byte", stdout.getvalue())
        self.assertIn(str(p), stdout.getvalue())
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(ingest.collect([str(self.corpus)], cache=None, quiet=True), [])
        self.assertEqual(stdout.getvalue(), "")

    def test_failed_cache_save_keeps_scan_result_and_removes_temporary_file(self):
        p = self.corpus / "a.txt"
        p.write_text("ordinary text")
        with patch.object(ingest.os, "replace", side_effect=OSError("simulated full disk")):
            self.assertEqual(self.collect(), [str(p)])
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_accepted_unicode_text_reaches_tokenization(self):
        p = self.corpus / "unicode.txt"
        text = "a" * 4095 + "é🙂\n<user>\nHello!\n</user>"
        p.write_text(text)
        self.assertEqual(self.collect(), [str(p)])
        self.assertEqual(ingest.as_stream(str(p)).tolist(), ByteTokenizer().encode(text).ids)


if __name__ == "__main__":
    unittest.main()
