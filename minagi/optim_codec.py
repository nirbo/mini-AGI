"""Lossless, pickle-free optimizer trees inside ordinary NPZ bundles."""

import json

import numpy as np
import torch


def encode(tree, prefix="__optim_"):
    arrays = {}

    def visit(value):
        if torch.is_tensor(value):
            value = value.detach().cpu().contiguous()
            key = prefix + str(len(arrays))
            dtype = str(value.dtype)
            arrays[key] = (value.view(torch.uint16) if value.dtype == torch.bfloat16
                           else value).numpy().copy()
            return {"tensor": key, "dtype": dtype}
        if isinstance(value, dict):
            return {"dict": [[visit(k), visit(v)] for k, v in value.items()]}
        if isinstance(value, (tuple, list)):
            return {"tuple" if isinstance(value, tuple) else "list": [visit(v) for v in value]}
        if value is None or type(value) in (str, int, float, bool):
            return value
        raise TypeError(f"unsupported optimizer metadata: {type(value).__name__}")

    metadata = json.dumps({"version": 1, "tree": visit(tree)}, allow_nan=False).encode()
    arrays[prefix + "json"] = np.frombuffer(metadata, dtype=np.uint8).copy()
    return arrays


def decode(arrays, prefix="__optim_"):
    def numpy(value):
        return value.detach().cpu().numpy() if torch.is_tensor(value) else value

    metadata = json.loads(numpy(arrays[prefix + "json"]).tobytes())
    if metadata.get("version") != 1:
        raise ValueError("unsupported optimizer codec version")

    def visit(value):
        if not isinstance(value, dict):
            return value
        if "tensor" in value:
            tensor = torch.from_numpy(numpy(arrays[value["tensor"]]).copy())
            if value["dtype"] == "torch.bfloat16":
                if tensor.dtype != torch.uint16:
                    raise ValueError("invalid BF16 optimizer storage")
                tensor = tensor.view(torch.bfloat16)
            if str(tensor.dtype) != value["dtype"]:
                raise ValueError("optimizer tensor dtype disagrees with metadata")
            return tensor
        if "dict" in value:
            return {visit(k): visit(v) for k, v in value["dict"]}
        if "tuple" in value:
            return tuple(visit(v) for v in value["tuple"])
        if "list" in value:
            return [visit(v) for v in value["list"]]
        raise ValueError("invalid optimizer metadata")

    return visit(metadata["tree"])
