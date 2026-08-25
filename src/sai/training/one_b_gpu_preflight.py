"""Measure Sai 1B H100 forward/backward readiness without an optimizer update."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from sai.data.agent_labeling import _atomic_create
from sai.data.token_stream import canonical_sha256, sha256_file
from sai.training.one_b_environment_receipt import SCHEMA as ENVIRONMENT_SCHEMA
from sai.training.one_b_olmo_config import SCHEMA as CONFIG_SCHEMA
from sai.training.one_b_production_contract import build_contract

SCHEMA = "sai-1b-h100-no-step-preflight-v1"


class OneBGpuPreflightError(RuntimeError):
    """The environment, config, data, model, or H100 measurement differs."""


def _load_signed(path: Path, schema: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise OneBGpuPreflightError("signed preflight input differs") from error
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or value.get("schema") != schema
        or value.get("receipt_sha256") != canonical_sha256(unsigned)
    ):
        raise OneBGpuPreflightError("signed preflight input differs")
    return value


def run(
    environment_path: Path,
    config_receipt_path: Path,
    config_root: Path,
) -> dict[str, Any]:
    """Run two one-sequence gradient measurements and never construct an optimizer."""

    environment = _load_signed(environment_path, ENVIRONMENT_SCHEMA)
    config_receipt = _load_signed(config_receipt_path, CONFIG_SCHEMA)
    descriptors = [
        row
        for row in config_receipt.get("configs", [])
        if row.get("stage") == 0 and row.get("phase") == "body"
    ]
    if len(descriptors) != 1:
        raise OneBGpuPreflightError("foundation body config differs")
    descriptor = descriptors[0]
    config_path = config_root / descriptor["path"]
    if (
        not config_path.is_file()
        or config_path.is_symlink()
        or config_path.stat().st_size != descriptor["bytes"]
        or sha256_file(config_path) != descriptor["sha256"]
    ):
        raise OneBGpuPreflightError("foundation body config bytes differ")
    try:
        import torch
        import torch.nn.functional as functional
        from olmo.config import TrainConfig
        from olmo.data import build_memmap_dataset
        from olmo.model import OLMo
    except ImportError as error:
        raise OneBGpuPreflightError("production GPU imports differ") from error
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise OneBGpuPreflightError("exactly one H100 is required")
    device = torch.device("cuda", 0)
    properties = torch.cuda.get_device_properties(device)
    if "H100" not in properties.name or properties.total_memory < 79_000_000_000:
        raise OneBGpuPreflightError("H100 device differs")
    config = TrainConfig.load(config_path)
    config.model.init_device = "cuda"
    dataset = build_memmap_dataset(config, config.data, include_instance_metadata=False)
    if len(dataset) != 244_140_544:
        raise OneBGpuPreflightError("foundation body dataset length differs")
    item = dataset[0]
    input_ids = item["input_ids"].unsqueeze(0).to(device)
    doc_lens = item["doc_lens"].unsqueeze(0).to(device)
    if (
        input_ids.shape != (1, 4_096)
        or int(input_ids.min()) < 0
        or int(input_ids.max()) >= 48_000
        or int(doc_lens.sum()) != 4_096
    ):
        raise OneBGpuPreflightError("preflight token or document geometry differs")
    input_sha256 = hashlib.sha256(
        input_ids.detach().cpu().numpy().astype("<u2", copy=False).tobytes()
    ).hexdigest()
    torch.cuda.reset_peak_memory_stats(device)
    model = OLMo(config.model)
    model.train()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    expected = build_contract()["parameter_ledger"]["total"]
    if parameter_count != expected:
        raise OneBGpuPreflightError("instantiated parameter count differs")
    elapsed = []
    losses = []
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(
                input_ids,
                doc_lens=doc_lens,
                max_doc_lens=[int(doc_lens.max())],
            )
            logits = output.logits
            loss = functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                input_ids[:, 1:].reshape(-1),
            )
        loss.backward()
        torch.cuda.synchronize(device)
        duration = time.perf_counter() - started
        if not math.isfinite(float(loss)) or duration <= 0:
            raise OneBGpuPreflightError("preflight loss or timing differs")
        elapsed.append(duration)
        losses.append(float(loss))
    measured_seconds = elapsed[-1]
    tokens_per_second = 4_096 / measured_seconds
    payload = {
        "schema": SCHEMA,
        "status": "passed_nontraining_h100_forward_backward_no_step",
        "environment_receipt_sha256": environment["receipt_sha256"],
        "config_bundle_receipt_sha256": config_receipt["receipt_sha256"],
        "config_sha256": descriptor["sha256"],
        "data_input_sha256": input_sha256,
        "dataset_sequences": len(dataset),
        "sequence_length": 4_096,
        "document_count_in_sample": int((doc_lens > 0).sum()),
        "parameter_count": parameter_count,
        "device": {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
        },
        "measurement": {
            "forward_backward_seconds": elapsed,
            "losses": losses,
            "steady_tokens_per_second_single_h100": tokens_per_second,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "optimizer_constructed": False,
        "optimizer_update_performed": False,
        "checkpoint_written": False,
        "model_training_started": False,
        "one_b_training_authorized": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--config-receipt", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise OneBGpuPreflightError("preflight output exists")
    value = run(args.environment, args.config_receipt, args.config_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.partial.{os.getpid()}")
    _atomic_create(temporary, value)
    os.replace(temporary, args.output)
    print(json.dumps({"receipt_sha256": value["receipt_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
