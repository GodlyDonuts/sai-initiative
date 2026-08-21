"""Deterministic, scale-aware initialization for every Sai model parameter."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import torch
from torch import nn

from sai.model.reference import CausalDepthwiseConv1d, RMSNorm, SaiCausalLM

POLICY = {
    "schema": "sai-model-initialization-policy-v1",
    "base_std": 0.02,
    "residual_projection_std": "0.02/sqrt(2*num_hidden_layers)",
    "residual_projection_modules": ["down_proj", "o_proj"],
    "embedding_std": 0.02,
    "causal_conv_std": 0.02,
    "rms_norm_weight": 1.0,
    "alpha_log_scale": 0.0,
    "alpha_bias": 0.0,
    "device": "cpu_before_device_cast",
    "distribution": "torch_normal_generator_seeded",
}


class SaiInitializationError(RuntimeError):
    """A parameter, seed, device, or initialization identity differs."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


POLICY_SHA256 = canonical_sha256(POLICY)


def initialize_sai_model(model: SaiCausalLM, *, seed: int) -> dict[str, Any]:
    """Initialize every parameter exactly once on CPU and return its receipt."""

    if not isinstance(model, SaiCausalLM):
        raise SaiInitializationError("initialization requires SaiCausalLM")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise SaiInitializationError("initialization seed differs")
    parameters = list(model.parameters())
    if not parameters or any(
        parameter.device.type != "cpu" for parameter in parameters
    ):
        raise SaiInitializationError("Sai initialization must occur on CPU")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initialized: set[int] = set()
    residual_std = POLICY["base_std"] / math.sqrt(2 * model.config.num_hidden_layers)
    with torch.no_grad():
        for module_name, module in model.named_modules():
            if isinstance(module, RMSNorm):
                module.weight.fill_(POLICY["rms_norm_weight"])
                initialized.add(id(module.weight))
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=POLICY["embedding_std"],
                    generator=generator,
                )
                initialized.add(id(module.weight))
            elif isinstance(module, nn.Linear):
                terminal = module_name.rsplit(".", 1)[-1]
                standard_deviation = (
                    residual_std
                    if terminal in POLICY["residual_projection_modules"]
                    else POLICY["base_std"]
                )
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=standard_deviation,
                    generator=generator,
                )
                initialized.add(id(module.weight))
                if module.bias is not None:
                    module.bias.zero_()
                    initialized.add(id(module.bias))
            elif isinstance(module, CausalDepthwiseConv1d):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=POLICY["causal_conv_std"],
                    generator=generator,
                )
                initialized.add(id(module.weight))
        for name, parameter in model.named_parameters():
            if id(parameter) in initialized:
                continue
            if name.endswith("alpha_log_scale"):
                parameter.fill_(POLICY["alpha_log_scale"])
            elif name.endswith("alpha_bias"):
                parameter.fill_(POLICY["alpha_bias"])
            else:
                raise SaiInitializationError(
                    f"parameter initialization is undefined: {name}"
                )
            initialized.add(id(parameter))
    if initialized != {id(parameter) for parameter in parameters}:
        raise SaiInitializationError("not every Sai parameter was initialized exactly")
    return {
        "schema": POLICY["schema"],
        "policy_sha256": POLICY_SHA256,
        "seed": seed,
        "parameters_initialized": len(parameters),
        "parameter_elements_initialized": sum(
            parameter.numel() for parameter in parameters
        ),
        "residual_projection_std": residual_std,
    }
