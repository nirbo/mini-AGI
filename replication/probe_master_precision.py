"""CPU numerical evidence for BF16 update retention, not an optimizer backend."""

import json

import torch


def exhaustive_sr_values(value):
    """Enumerate 16-bit rounding offsets; numerical oracle, not a backend."""
    value = torch.tensor(value, dtype=torch.float32)
    bits = int(value.view(torch.int32)) & 0xffffffff
    offsets = torch.arange(65536, dtype=torch.int64)
    rounded = ((bits + offsets) & 0xffff0000).to(torch.int32).view(torch.float32)
    return value, rounded.to(torch.bfloat16)


def evidence():
    initial = torch.tensor(.013, dtype=torch.bfloat16)
    bf16, fp32 = initial.clone(), initial.float()
    delta = 1.8e-5
    for _ in range(32):
        bf16 = (bf16.float() - delta).to(torch.bfloat16)
        fp32 -= delta
    target, stochastic = exhaustive_sr_values(float(initial.float() - delta))
    next_value = torch.nextafter(initial, torch.tensor(float("inf"), dtype=torch.bfloat16))
    return {"initial_bf16_value": float(initial), "bf16_spacing_here": float(next_value - initial),
            "update": delta, "updates": 32, "bf16_final": float(bf16), "fp32_final": float(fp32),
            "one_update_fp32_target": float(target), "one_update_round_nearest": float(target.bfloat16()),
            "one_update_exhaustive_stochastic_mean": float(stochastic.double().mean()),
            "limits": "Numerical example at historical repo weight/update scales, not current-model update statistics or convergence."}


if __name__ == "__main__":
    torch.set_num_threads(1)
    print(json.dumps(evidence(), indent=2))
