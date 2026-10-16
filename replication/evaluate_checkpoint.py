"""Read-only predictive evaluation of a weights directory on a text file.

Scores a specified prefix, resetting at fixed context boundaries. Results are
conditional on the supplied corpus and are not a general capability benchmark.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from minagi.ingest import as_stream
from minagi.precision import amp, set_compute_dtype
from minagi.recur import load_any, PREDICTIVE_METRIC
from minagi.stream import FileReader, _score_windows


@torch.no_grad()
def evaluate(weights, text, *, chunk=512, context=4096, tokens=32768,
             device="cuda", precision="bf16", compare_objective=False):
    if min(chunk, context, tokens) < 1:
        raise ValueError("chunk, context and tokens must be positive")
    set_compute_dtype(precision)
    model, _ = load_any(str(weights), torch.device(device), read_only=True)
    context = min(context, model.cfg.block)
    data = as_stream(str(text))[:tokens + 1]
    reader = FileReader(model, data, str(text), chunk, context, device)
    samples = []
    start = time.perf_counter()
    while not reader.done():
        loss = reader.step(learn=False)
        samples.append((float(loss), reader.last_count, reader.resets))
    ce, stderr = _score_windows(samples)
    count = sum(n for _, n, _ in samples)
    result = {
        "metric": PREDICTIVE_METRIC, "nats_per_token": ce,
        "bits_per_token": ce / math.log(2), "perplexity": math.exp(ce),
        "stderr": stderr, "scored_tokens": count,
        "windows": len({w for _, _, w in samples}),
        "seconds": time.perf_counter() - start,
        "weights": str(weights), "text": str(text),
        "text_sha256": hashlib.sha256(text.read_bytes()).hexdigest(),
        "scored_token_ids_sha256": hashlib.sha256(data.astype("<u2").tobytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256((weights / "manifest.json").read_bytes()).hexdigest(),
        "chunk": chunk, "context": context, "device": device, "precision": precision,
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else None,
        "halting": {k: getattr(model.cfg, k) for k in
                    ("max_steps", "min_steps", "halt_thresh", "halt_freeze")},
        "policy": "threshold-selected logits; no decoding guards; fresh context at fixed boundaries",
        "limitations": "Single corpus prefix; checkpoint training overlap is unverified; not comparable to historical scores.",
    }
    # Structural markers have multi-byte spellings but occupy one token.
    # Avoid calling bits/token bits/byte in their presence.
    result["bits_per_byte"] = result["bits_per_token"] if np.all(data[1:] < 256) else None
    provenance = weights.parent / "download-provenance.json"
    if provenance.exists():
        p = json.loads(provenance.read_text())
        result["checkpoint_source"] = {k: p[k] for k in ("repo", "revision")}
    if compare_objective:
        # Same weights, inputs, context and inference depth ceiling, but the
        # target-bearing forward reports expected depth CE + halting KL.
        # This diagnostic excludes reader-level expert auxiliary penalties.
        objective, n = 0., 0
        caches, seen = None, 0
        while n < len(data) - 1:
            if caches is None or seen == context:
                caches, seen = model.empty_caches(), 0
            width = min(chunk, context - seen, len(data) - 1 - n)
            a = torch.tensor(data[n:n+width+1].astype(np.int64), device=device).unsqueeze(0)
            with amp(device):
                _, loss = model(a[:, :-1], a[:, 1:], caches=caches, pos_offset=seen)
            objective += float(loss) * width
            n += width
            seen += width
        result["diagnostic_expected_ce_plus_ponder_kl"] = objective / n
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--tokens", type=int, default=32768)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--compare-objective", action="store_true")
    args = parser.parse_args()
    result = evaluate(args.weights, args.text, chunk=args.chunk, context=args.context,
                      tokens=args.tokens, device=args.device, precision=args.precision,
                      compare_objective=args.compare_objective)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
