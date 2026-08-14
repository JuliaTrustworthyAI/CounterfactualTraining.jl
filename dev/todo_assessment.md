# Per-Epoch vs Per-Batch Counterfactual Generation

## Correctness

The per-epoch design is correct. CFs are generated against the model at epoch start, then the model updates batch-by-batch. By the last batch, CFs are ~39 updates stale. This is a known, accepted pattern (analogous to target networks in deep RL, or off-policy correction). It actually helps stability by preventing CFs from aggressively chasing a moving model. The research branch deliberately chose this design (see `src/training.jl:89–94`, "PAPER REF" comment about generating CFs "in one sweep").

CFs need not be paired with batch factuals. The CFs contribute to the implausibility/regularization/adversarial terms — a separate signal from the classification loss on the batch's factuals. `split_obs(1:nsamples, length(train_set))` partitions CFs across batches without pairing them to specific training samples. This is correct for the objective being optimized (the CF terms are aggregated via `mean`, not paired with specific factuals).

No correctness issue is introduced by the per-epoch design. The alternative (per-batch) would be marginally "fresher" but not more correct — it would just be a different point on the freshness/stability tradeoff.

## Simplicity

The current design is simpler. Clean separation of concerns: `generate_native!` handles CF search (train/eval mode, chunking, convergence, mutability, domain). The training loop handles only the forward/backward/update. Mixing them would entangle two distinct algorithms.

The per-batch alternative would add complexity:
- Per-batch train/eval mode switching (CF search needs eval mode per PERFORMANCE_PLAN.md Step 5)
- Per-batch neighbour finding
- Per-batch target encoding
- Careful handling of the `nce < batchsize` case

Minor wart: the `counterfactual_dl` as a `Vector{NTuple{5}}` is slightly awkward, and `split_obs` adds indirection — but these are minor and don't justify restructuring.

## Performance

### Remaining performance concerns with per-batch generation

1. **GPU utilization / kernel-launch overhead.** Generating ~3 CFs per batch means each of the 30 search iterations runs a pullback on 3 columns through the model. For a launch-bound workload like ResNet-18 on 28×28 MNIST (which PERFORMANCE_PLAN.md identifies as the bottleneck regime), tiny pulls are dominated by kernel-launch and Zygote overhead, not compute. The current design batches 32–128 columns per pullback, which amortizes that overhead. Same total FLOPs, but ~1200 tiny pullbacks vs ~30–120 larger ones.

2. **Fixed per-call overhead.** `generate_counterfactuals!` has setup cost (opt state, buffer allocation, mask/bounds prep). 40 calls/epoch vs 1 multiplies this.

3. **Marginal freshness benefit.** The CFs would be ~0–39 gradient steps fresher. But the current design's staleness is already accepted (and arguably a stability feature, like target networks in RL). The objective doesn't pair CFs with specific batch factuals anyway — CFs contribute to implausibility/reg terms aggregated via `mean`.

### Workload dependence

| Workload | Recommended design | Rationale |
|---|---|---|
| Launch-bound (ResNet, small `nce`) | **Per-epoch** (current) | GPU kernel-launch overhead dominates; larger batches amortize better |
| Compute-bound (large `nce`, GPU saturated) | Per-batch *could* be competitive | Might give slightly better convergence; empirical question needing a benchmark |

