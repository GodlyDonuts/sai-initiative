# Sai Workspace Performance Contract

Status: implemented no-training CPU mechanics receipt. Production CUDA/H100
performance remains explicitly unqualified.

## Measurement scopes

`sai-workspace-performance` measures the incremental `LatentWorkspace` reference
under `eval()` and `torch.inference_mode()`. It hashes deterministic inputs,
mask, outputs, module state, parameter versions, and RNG state; records raw
latency samples and exact module call counts; and proves zero gradients, zero
backward calls, zero optimizer steps, zero GPU submissions, and unchanged state.

CPU timing is diagnostic only. The Python backbone has no decode cache and is
not a production decoder. The reference wrapper now integrates the workspace
delta into the final hidden state before one shared LM-head projection, avoiding
an unaccounted second vocabulary projection, but full-model timing is still not
qualified by this harness.

## Production matrix

Before the workspace is training-ready, a frozen CUDA runner must measure batch
one, BF16 incremental workspace cases at prompt lengths 128, 512, and 2,048 and
recurrence horizons 1, 2, 4, 8, and 16. The 2,048-token cases at every horizon
are mandatory. Each fresh process must pin the exact source/runtime/kernel,
device UUID, backend, determinism settings, input/state hashes, warmups, raw
timing blocks, and allocator peaks. A production threshold must be declared
before results exist.

True HBM traffic requires hardware counters such as Nsight Compute DRAM read and
write bytes. Allocator peak or profiler memory is not a substitute. Until those
counters, production kernels, and the complete case matrix exist, receipts must
retain:

- `production_qualified=false`;
- `design_performance_gate_pass=null`;
- `dram_traffic_measured=false`; and
- `end_to_end_production_latency_qualified=false`.

Infrastructure absence or OOM makes qualification incomplete; it cannot become
a scientific failure or a performance pass.
