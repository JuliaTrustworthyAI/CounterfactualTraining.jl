# Performance Plan: Native GPU Training

> **Audience.** This plan is written to be executed by less-capable coding agents starting in a fresh session. It assumes NO prior context. Each step is self-contained: it names the exact file/function, shows the current code (with `~line` citations), gives the after-state sketch, explains *why* it helps, and states a verification command plus an acceptance criterion. Line numbers drift — always re-read the source before editing.

> **Repo rule that overrides everything.** Do not modify existing tests. If a step makes a parity test fail, STOP and report — fix the *code*, never the test. If in doubt, discuss.

---

## Context

The `Native` submodule (`src/native/training.jl`) provides batched, GPU-compatible counterfactual (CF) training. Measured per-epoch times for the `docs/src/gpu.qmd` example (ResNet-18 on a 5000-sample MNIST subset, batchsize 128, `nce=128`, `cf_batchsize=32`, `maxiter=30`, 10 epochs, AMD GPU; CPU numbers will be comparable as a ratio):

- Full objective: ~3.6s, 2.0s during burn-in (epochs 1-2), then ~14.5-19s per epoch (epochs 3-10).
- Vanilla objective: ~1.7-2.0s per epoch.

So CF generation adds ~13s per epoch.

Workload flavour: vanilla epoch = 40 batches × (forward+backward on 128 samples) ≈ 1.8s → ~45ms per batch step. For a ResNet-18 on single-channel 28×28 this is **kernel-launch/overhead-bound, not compute-bound**. The CF path multiplies exactly the overheads that dominate at this scale (many small GPU kernels + many CPU↔GPU synchronizations).

## Cost model

Full-objective overhead ≈ 13s/epoch breaks down as:

- CF search: 30 iterations × 4 chunks (`nce=128`, `cf_batchsize=32`) = **120 Zygote pullbacks**, each ≈ a launch-bound forward+backward (~45ms) → ≈ 5.4s of raw passes.
- ~120 convergence-check GPU→CPU syncs (`|> Flux.cpu` per chunk per iteration).
- ~2-3 boolean-mask-indexing syncs per iteration in gradient-zeroing and AE-tracking (masking a GPU array with a `BitVector` forces a `findall`/`count` sync).
- Per-batch scalar syncs in the training loop (see Step 6).

Phases 1-2 remove the factor-4 chunking and ~90% of the syncs, targeting a full-objective epoch of **~4-6s** (down from ~15s) at unchanged convergence.

---

## Bottleneck Table

1. **CF search chunking.** `cf_batchsize=32` with `nce=128` → 4 chunks × 30 iterations = 120 pullbacks on 32-column slices instead of 30 pullbacks on the full 128-column matrices. Chunk loop in `generate_counterfactuals!` (src/native/training.jl, ~lines 466-502).
2. **GPU→CPU sync per chunk per iteration.** `target_probs = probs_chunk[linear_idx] |> Flux.cpu` in the convergence check (~line 499) → 120 syncs/epoch, each stalling the GPU pipeline.
3. **Boolean-mask indexing of GPU arrays.** `ΔX[:, converged] .= zero(...)` (~line 518), `update[:, converged] .= zero(...)` (~line 539), and `last_valid_ae[:, valid_ae] .= X′[:, valid_ae]` in `track_adversarial_examples!` (~line 272). Masked indexing on GPU requires `findall` (sync) + gather/scatter kernels.
4. **Per-batch scalar syncs in the training loop.** The three `push!` calls inside `ChainRulesCore.ignore_derivatives()` in `counterfactual_training` (~lines 1006-1010) each call `sum`/`mean` on GPU vectors → ~120 avoidable syncs/epoch. (`push!(losses, val)` and `isfinite(val)` syncs are inherent and stay.)
5. **Three tiny launch-bound forwards per training batch.** `m(perturbed_input)`, `m(neighbours)`, `m(advexms)` (~lines 996-999) each run on only ~3 columns (128 CEs split over ~40 batches via `split_obs`) → ~120 tiny ResNet forwards/epoch.
6. **Docs example defaults.** Training data stays on CPU (`Flux.DataLoader((X, y_onehot); ...)` on CPU arrays in docs/src/gpu.qmd), so `input |> device` copies ~400KB/batch inside the loop; `cf_batchsize = 32` is the demonstrated value.
7. **BatchNorm correctness.** The CF search loop and `accuracy()` (src/utils.jl) run the model in **train mode**, so ResNet BatchNorm running statistics get updated ~120×/epoch on counterfactual inputs, and accuracy is measured with batch (not running) statistics. (No BatchNorm exists in `src/` or `test/` — this only affects BN models like the docs' ResNet-18.)

---

## Phase 0 — Benchmark Harness (new file `dev/bench_native.jl`)

**File/Function.** New file `dev/bench_native.jl`.

**Change.** A self-contained script that reproduces the gpu.qmd setup and reports before/after per-epoch times plus correctness anchors. It must run cleanly on CPU (CI) and on a GPU when present.

**Code sketch** (skeleton — structure it like this):

```julia
using CounterfactualTraining
using CounterfactualTraining.Native
using CounterfactualExplanations
using Flux
using Metalhead
using Random
using MLDatasets
using Printf, Statistics

Random.seed!(42)

# --- GPU detection (copy verbatim from docs/src/gpu.qmd lines 37-68) ---
has_gpu = false
device = identity
try using CUDA;   CUDA.functional()   && (has_gpu = true; device = Flux.gpu) catch end
try using AMDGPU; AMDGPU.functional() && (has_gpu = true; device = Flux.gpu) catch end

# --- Data (MNIST subset 5000, flattened 784, normalized [-1,1]) ---
train_data = MLDatasets.MNIST(; split=:train)
X = Float32.(reshape(train_data.features, 784, :)) .* 2f0 .- 1f0
y = train_data.targets .+ 1
idx = shuffle(1:size(X, 2))[1:5000]
X, y = X[:, idx], y[idx]
domain = [(-1.0f0, 1.0f0) for _ in 1:784]
y_onehot = Flux.onehotbatch(y, 1:10)
train_set = Flux.DataLoader((X, y_onehot); batchsize=128, shuffle=true)

gen = NativeGenerator()
opt = Flux.Adam(1e-3)
nepochs = 3          # short run; burnin rounds to 1 epoch, leaving 2 CF epochs

make_model() = Chain(x -> reshape(x, 28, 28, 1, :), ResNet(18; inchannels=1, nclasses=10))

function run_training(objective)
    Random.seed!(42)
    m = make_model()
    o = Flux.setup(opt, m)
    _, log = counterfactual_training(objective, m, gen, train_set, o;
        device, nepochs, domain, verbose=0, nce=128, cf_batchsize=32,
        maxiter=30, burnin=0.2f0, accuracy_every=nepochs)
    return [l.time_taken for l in log], [l.percent_valid for l in log]
end

# Standalone timing of CF generation (no training):
function time_generate_native()
    m = make_model() |> device
    t = @timed generate_native!(m, train_set, gen;
        nsamples=128, domain=domain, maxiter=30, verbose=0,
        device=device, cf_batchsize=32)
    return t.time
end

# Warmup (compilation) run, discard:
run_training(FullObjective())
time_generate_native()

# Measured runs:
for rep in 1:3
    t_full, pv = run_training(FullObjective())
    @printf("full[%d]:    %s  (percent_valid=%s)\n", rep, t_full, pv)
end
for rep in 1:3
    t_van, _ = run_training(VanillaObjective(; needs_ce=false))
    @printf("vanilla[%d]: %s\n", rep, t_van)
end
println("generate_native! standalone: ", time_generate_native(), " s")
```

**Why it helps.** Without a reproducible baseline you cannot prove any optimization is a win. The standalone `generate_native!` timing isolates CF-generation cost from training cost (this is the number that should collapse after Phase 1).

**Verification.** Run it once now, BEFORE any Phase 1 change, discarding the warmup (first call includes compilation):

```bash
julia --project=. dev/bench_native.jl | tee dev/bench_baseline.txt
```

**Acceptance criterion.** Runs end-to-end (CPU or GPU), prints per-epoch times for full and vanilla objectives plus a standalone `generate_native!` time. Output saved to `dev/bench_baseline.txt` — the comparison target for Phase 5. Record in the file header which device was used (CPU/CUDA/AMDGPU); all later comparisons must use the same device.

---

## Phase 1 — CF Search Loop (`src/native/training.jl`, `generate_counterfactuals!`)

Steps 1-4 are **numeric-parity-preserving** for BN-free models (same math, different execution order; all loss terms are column-local, so gradients are bitwise identical whether computed per-chunk or on the full matrix). Step 5 is an **intentional numeric change**. Do the steps in order, running the parity tests after each. Re-read the function before each edit.

> **BatchNorm caveat for Step 1 ordering.** In train mode, BN batch statistics depend on chunk size — this is already true in the status quo (`cf_batchsize=32` vs `128` give different BN stats today). So on BN models the Step 1 fast path is *not* bitwise-identical to the chunked baseline until Step 5 (testmode) lands, after which BN behaviour is chunk-size-independent. If you want bitwise before/after comparisons on the ResNet benchmark, land **Step 5 first**; the test suite itself uses only Dense models, so either order keeps it green.

Relevant current structure of `generate_counterfactuals!` (src/native/training.jl, ~lines 410-553). Reused pre-allocations already exist: `X′_old`, `update`, `ΔX`, `perturbations`, `norms`; `mutability_masks` and `domain_bounds` are precomputed. The iteration loop (~lines 466-550):

```julia
for iter in 1:maxiter
    for start in 1:cf_batchsize:N
        stop = min(start + cf_batchsize - 1, N)
        local logits_chunk
        y, back = Flux.pullback(X′[:, start:stop]) do xc
            logits_chunk = model(xc)
            return generator_loss_from_logits(
                generator, logits_chunk, xc, X[:, start:stop],
                targets_onehot[:, start:stop], @view(targets[start:stop]),
                iter, reg_strength, decay, maxiter)
        end
        ΔX[:, start:stop] .= back(one(y))[1]
        if iter < maxiter
            probs_chunk = Flux.softmax(logits_chunk; dims=1)
            target_idx_chunk = targets[start:stop]
            C = size(probs_chunk, 1); n_chunk = length(target_idx_chunk)
            linear_idx = (0:(n_chunk - 1)) .* C .+ target_idx_chunk
            target_probs = probs_chunk[linear_idx] |> Flux.cpu
            converged[start:stop] .= target_probs .>= decision_threshold
        end
    end
    if iter >= maxiter
        converged = trues(N)
    end
    if all(converged)
        break
    end
    if any(converged)
        ΔX[:, converged] .= zero(eltype(ΔX))
    end
    copyto!(X′_old, X′)
    Flux.update!(opt_state, X′, ΔX)
    update .= X′ .- X′_old
    if any(converged)
        update[:, converged] .= zero(eltype(update))
    end
    batched_apply_mutability!(update, mutability_masks)
    X′ .= X′_old .+ update
    batched_apply_domain_constraints!(X′, domain_bounds)
    track_adversarial_examples!(last_valid_ae, X, X′, epsilon, p, perturbations, norms)
end
```

### Step 1: Single-chunk fast path

**File/Function.** `generate_counterfactuals!` (src/native/training.jl).

**Change.** When `cf_batchsize >= N`, the chunk loop runs exactly one chunk covering `1:N`, yet still slices arrays (`X′[:, 1:N]`, `X[:, 1:N]`, `targets_onehot[:, 1:N]`) — each slice copies a `D×N` GPU array, and every chunk boundary costs an extra pullback. Precompute the chunk ranges once before the iteration loop, take a no-slice fast path when a chunk covers all columns, and use `@views` in the remaining chunked path.

**Code sketch** (restructure the chunk loop; the shared update/constraints/tracking body after the inner loop stays unchanged):

```julia
# Before the `for iter` loop, precompute chunk column ranges once:
chunk_ranges = collect(start:min(start + cf_batchsize - 1, N)
                       for start in 1:cf_batchsize:N)

for iter in 1:maxiter
    for cols in chunk_ranges
        local logits_chunk
        if length(cols) == N          # single chunk covers all columns: no slicing
            y, back = Flux.pullback(X′) do xc
                logits_chunk = model(xc)
                return generator_loss_from_logits(
                    generator, logits_chunk, xc, X, targets_onehot,
                    targets, iter, reg_strength, decay, maxiter)
            end
            copyto!(ΔX, back(one(y))[1])
        else                          # chunked path: views instead of copies
            y, back = Flux.pullback(@view(X′[:, cols])) do xc
                logits_chunk = model(xc)
                return generator_loss_from_logits(
                    generator, logits_chunk, xc, @view(X[:, cols]),
                    @view(targets_onehot[:, cols]), @view(targets[cols]),
                    iter, reg_strength, decay, maxiter)
            end
            ΔX[:, cols] .= back(one(y))[1]
        end
        # ... convergence check (see Step 2) ...
    end
    # ... unchanged shared body (early exit, masking, update, constraints, tracking) ...
end
```

Notes for the implementer:
- `length(cols) == N` is true exactly when there is a single chunk `1:N` (chunks are consecutive partitions starting at 1), so the fast path triggers precisely when `cf_batchsize >= N`.
- A `Flux.pullback` over a `@view` returns the gradient as a dense array of the view's shape, so `ΔX[:, cols] .= grad` still works. If the AD/backend combination ever errors on a `SubArray` primal, fall back to plain slicing (`X′[:, cols]`) — the copies are cheap relative to a pullback.
- Preserve the existing ordering: convergence check only when `iter < maxiter`; after the chunk loop `if iter >= maxiter; converged = trues(N); end`; then `if all(converged); break; end` before any update work.

**Why it helps.** 4× fewer pullbacks, 4× fewer GPU slice copies; the biggest single per-epoch win.

**Verification.**
```bash
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. dev/bench_native.jl
```

**Acceptance criterion.** *Parity-preserved.* `parity_generator_tests.jl`, `parity_training_tests.jl`, `native_helpers_tests.jl`, `native_edge_tests.jl` pass unmodified. Standalone `generate_native!` time drops (expect roughly 3-4×). On the Dense test models, results are bitwise identical (chunked vs whole are column-local computations). On BN models in train mode, expect small drift until Step 5 (see ordering caveat above).

### Step 2: One convergence sync per iteration

**File/Function.** `generate_counterfactuals!` (src/native/training.jl).

**Change.** Replace the per-chunk `target_probs = probs_chunk[linear_idx] |> Flux.cpu` (a GPU→CPU sync per chunk) with a GPU→GPU write into a preallocated buffer, then a single `Flux.cpu` per iteration.

**Code sketch.** Before the `for iter` loop (next to the other preallocations, ~line 460):

```julia
target_probs_buf = similar(X′, N)   # N-vector on device
```

Inside each chunk, replace the convergence block with (fast path: use `1:N` as `cols` and the full `probs_chunk`/`targets`):

```julia
if iter < maxiter
    probs_chunk = Flux.softmax(logits_chunk; dims=1)
    target_idx_chunk = targets[cols]
    C = size(probs_chunk, 1); n_chunk = length(target_idx_chunk)
    linear_idx = (0:(n_chunk - 1)) .* C .+ target_idx_chunk
    target_probs_buf[cols] .= probs_chunk[linear_idx]   # GPU→GPU gather+copy, no sync
end
```

After the chunk loop, once per iteration (replacing the per-chunk `converged[start:stop] .= ...` writes):

```julia
if iter < maxiter
    converged_dev = target_probs_buf .>= decision_threshold   # device Bool vector
    converged = Flux.cpu(converged_dev)                       # ONE sync per iteration
else
    converged = trues(N)
end
```

Keep `converged_dev` around — Step 3 reuses it. (At `iter == maxiter` it is stale, but the loop breaks immediately after `all(trues(N))`, so it is never read.)

**Why it helps.** 120 syncs/epoch → 30 syncs/epoch.

**Verification.** As Step 1.

**Acceptance criterion.** *Parity-preserved.* Tests pass unmodified; identical convergence decisions (the threshold comparison is applied to identical values, just materialized once per iteration).

### Step 3: GPU-side converged masking

**File/Function.** `generate_counterfactuals!` (src/native/training.jl).

**Change.** `ΔX[:, converged] .= 0` and `update[:, converged] .= 0` mask GPU arrays with a CPU `BitVector`, forcing a `findall`/sync each. Replace with a broadcast multiply by a device `1×N` mask derived from `converged_dev` (Step 2).

**Code sketch:**

```julia
# After the early-exit check, replacing both `if any(converged) ... end` blocks:
mask = reshape(.!converged_dev, 1, N)   # 1×N device mask; nothing converged → all-ones, multiply is exact
ΔX .*= mask
# ...
update .= X′ .- X′_old   # existing line
update .*= mask
```

**Why it helps.** Removes per-iteration `findall` syncs and gather/scatter; one fused elementwise kernel each.

**Verification.** As Step 1.

**Acceptance criterion.** *Parity-preserved.* Tests pass unmodified. The `any(converged)` guards may be dropped: multiplying by an all-ones mask is numerically exact and cheaper than the guard's sync. `converged` (CPU, from Step 2) is still needed for `all(converged)` early exit and the function's return value — Step 2 already materializes it once per iteration.

### Step 4: Broadcast adversarial-example tracking

**File/Function.** `track_adversarial_examples!` (src/native/training.jl, 6-argument method ~lines 254-274; the 5-argument convenience wrapper delegates to it and needs no change).

**Current code** (~lines 271-273):

```julia
valid_ae = norms .<= epsilon
last_valid_ae[:, valid_ae] .= X′[:, valid_ae]
```

**Code sketch:**

```julia
valid_mask = reshape(norms .<= epsilon, 1, :)
last_valid_ae .= ifelse.(valid_mask, X′, last_valid_ae)
```

**Why it helps.** Boolean-mask indexing on GPUs requires `findall` (a sync) to size the gather plus a scatter; broadcast `ifelse` is a single fused elementwise kernel with no sync.

**Verification.** As Step 1.

**Acceptance criterion.** *Bitwise-parity* (pure selection, no arithmetic). `native_helpers_tests.jl` passes unmodified (it exercises `track_adversarial_examples!`).

### Step 5: BatchNorm fix during CF search (REQUIRED, intentional numeric change)

**File/Function.** `generate_native!` (src/native/training.jl, ~lines 693-815).

**Change.** The factual-prediction chunk loop (~lines 752-757) and the `generate_counterfactuals!` call (~lines 766-780) run the model in **train mode**, so BN layers recompute batch statistics and update running stats on counterfactual inputs (120×/epoch with the docs settings). Wrap both in `testmode!`/`trainmode!` with `try/finally` so train mode is always restored.

**Code sketch** (spanning ~line 751 through the `generate_counterfactuals!` call):

```julia
Flux.testmode!(model)
try
    # ... existing factual_preds chunk loop ...
    # counterfactuals, last_valid_ae, converged_mask, maxiter = generate_counterfactuals!(...)
finally
    Flux.trainmode!(model)
end
```

Note: `generate_counterfactuals!` is exported and may also be called standalone; standalone callers keep the previous (train-mode) behaviour — only the training path via `generate_native!` is switched to testmode. Mention this in the step's commit message.

**Why it helps.** Correctness: keeps BN running statistics clean of adversarial inputs and makes CF search deterministic w.r.t. chunk size. Also slightly faster for BN models (eval mode skips batch-stat computation and running-stat updates). For BN-free models `testmode!` is a no-op, so the Dense-based test suite is unaffected.

**Verification & caveat.**

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

If any parity test uses a BatchNorm model and now fails, STOP and report; do not change the test (repo rule).

**Acceptance criterion.** *Intentional numeric change* (documented above and in the function docstring). Full test suite stays green.

---

## Phase 2 — Training Loop (`counterfactual_training` in `src/native/training.jl`)

### Step 6: Defer logging syncs

**File/Function.** `counterfactual_training` (src/native/training.jl: per-epoch state ~lines 949-952, batch loop ~lines 986-1023, logging ~lines 1036-1044).

**Change.** The three `push!` calls inside `ChainRulesCore.ignore_derivatives()` (~lines 1006-1010) each reduce a GPU vector to a host scalar → a sync per call, ~120/epoch. Accumulate on device and sync once per epoch.

> **Important implementation detail.** `sum(::GPUArray)` returns a *host* scalar (it syncs). To keep the accumulation sync-free you must use a dims-preserving reduction — `sum(x; dims=1)` returns a 1-element *device* array. Sketches below use that form.

**Current code** (~lines 1006-1010):

```julia
ChainRulesCore.ignore_derivatives() do
    push!(implausibilities, sum(implaus) / length(implaus))
    push!(reg_losses, sum(regs) / length(regs))
    return push!(validity_losses, adversarial_loss)
end
```

**Code sketch.** Where `losses`/`implausibilities`/`reg_losses`/`validity_losses` are currently reset per epoch (~lines 949-952), replace the latter three with device accumulators (reset every epoch, before the batch loop):

```julia
losses = Float32[]
implaus_acc = device([0.0f0])
reg_acc     = device([0.0f0])
adv_acc     = device([0.0f0])
```

(`device` is the existing keyword argument of `counterfactual_training`.) Delete the now-unused `implausibilities`/`reg_losses`/`validity_losses` vector declarations.

Inside the batch loop's `ignore_derivatives` block:

```julia
ChainRulesCore.ignore_derivatives() do
    implaus_acc .+= sum(implaus; dims=1) ./ length(implaus)   # stays on device, no sync
    reg_acc     .+= sum(regs; dims=1) ./ length(regs)
    adv_acc     .+= adversarial_loss                          # host scalar already; H2D broadcast, no sync
    nothing
end
```

At epoch end (in the `epoch > burnin` branch, ~lines 1036-1044), replace `sum(implausibilities)/length(implausibilities)` etc. with:

```julia
n_batches = length(train_set)
implaus = Flux.cpu(implaus_acc)[1] / n_batches
log_reg_loss = Flux.cpu(reg_acc)[1] / n_batches
log_adv_loss = Flux.cpu(adv_acc)[1] / n_batches
```

**Edge cases.**
- During burn-in the CF branch sets `implaus = [0.0f0]` (CPU) — `sum([0.0f0]; dims=1) ./ 1` is a CPU 1-vector and `implaus_acc .+= cpu_vector` is a cheap H2D broadcast; the accumulated zeros are never read because the `epoch > burnin` logging branch is false. Identical semantics to today.
- Keep `push!(losses, val)` and the `isfinite(val)` check unchanged — that sync is inherent to reporting/debugging the scalar loss.

**Why it helps.** ~120 syncs/epoch → ~3 syncs/epoch.

**Verification.** As Step 1.

**Acceptance criterion.** *Parity-preserving*: `sum(per-batch-mean)/n_batches` equals the current `mean of per-batch means` (same values, same order of division). Tests pass unmodified; logged `implaus`/`log_reg_loss`/`log_adv_loss` values match the pre-change run bit-for-bit on CPU.

### Step 7: Opt-in keyword `fuse_cf_forwards`

**File/Function.** `counterfactual_training` signature + batch loop (src/native/training.jl); new helper in `src/loss.jl`; wiring in `src/CounterfactualTraining.jl`.

**Change.** Add keyword `fuse_cf_forwards::Bool=false`. When `true`, replace the three separate CF forward passes with one concatenated forward and split the logits. Default `false` ⇒ no numeric change when unused (user's decision: opt-in).

**Current training-loop forwards** (~lines 995-999):

```julia
if !isnothing(perturbed_input)
    implaus, regs = implausibility_and_reg_loss(m, perturbed_input, neighbours, targets_enc)
    adversarial_loss = loss.class_loss(m(advexms), factual_enc)
else
    ...
end
```

**Code sketch.** In the `if !isnothing(perturbed_input)` branch:

```julia
if fuse_cf_forwards
    n_cf = size(perturbed_input, 2)
    n_nb = size(neighbours, 2)
    logits_all = m(cat(perturbed_input, neighbours, advexms; dims=2))
    logits_cf = @view(logits_all[:, 1:n_cf])
    logits_nb = @view(logits_all[:, n_cf+1:n_cf+n_nb])
    logits_ae = @view(logits_all[:, n_cf+n_nb+1:end])
    implaus, regs = implausibility_and_reg_loss_from_logits(logits_cf, logits_nb, targets_enc)
    adversarial_loss = loss.class_loss(logits_ae, factual_enc)
else
    implaus, regs = implausibility_and_reg_loss(m, perturbed_input, neighbours, targets_enc)
    adversarial_loss = loss.class_loss(m(advexms), factual_enc)
end
```

**Supporting helper in `src/loss.jl`** (mirrors the existing `generator_loss`/`generator_loss_from_logits` split pattern, src/native/training.jl ~306-355; the algebra matches `implausibility_and_reg_loss` exactly):

```julia
function implausibility_and_reg_loss_from_logits(logits_cf, logits_nb, targets)
    implaus_x = (logits_cf .- logits_nb)[:, :]' * targets
    implaus = diag(implaus_x[:, :])
    reg_x = (abs2.(logits_nb) .+ abs2.(logits_cf))' * targets
    regs = diag(reg_x[:, :])
    return implaus, regs
end
```

**Wiring.** In `src/CounterfactualTraining.jl`: add `implausibility_and_reg_loss_from_logits` to the `export` list near the other loss exports (~line 8) and to the `Native` submodule's `import ..CounterfactualTraining: ...` list (~line 15).

**Docstring note** on the new `fuse_cf_forwards` keyword of `counterfactual_training`: *fusing changes BatchNorm batch statistics (stats computed over the concatenated mini-batch rather than each tensor at its native width). For BN-free models results are identical; for BN models results differ slightly. Off by default.*

**Why it helps.** 3 launch-bound tiny forwards → 1 forward per batch; ~120 small ResNet forwards/epoch → ~40.

**Verification.**

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. dev/bench_native.jl
```

Add a NEW small CPU test (new testset; adding tests is allowed — the rule prohibits *changing existing* tests) that trains/compares one step of fused vs unfused loss evaluation on a Dense-only (BN-free) model: the two paths must produce identical loss values and gradients.

**Acceptance criterion.** With default `false`, full parity (tests pass unmodified). With `fuse_cf_forwards=true`, the new BN-free test matches the unfused path; the docstring documents the BN caveat.

---

## Phase 3 — Docs (`docs/src/gpu.qmd`)

**File.** `docs/src/gpu.qmd`.

**Changes.**

1. **Keep data on device across batches.** Currently `train_set = Flux.DataLoader((X, y_onehot); ...)` is built on CPU arrays (~lines 85-86), so the loop's `input |> device` does a ~400KB H2D copy + allocation per batch. Move data to device once before the loader:
   ```julia
   X = X |> device
   y_onehot = y_onehot |> device
   train_set = Flux.DataLoader((X, y_onehot); batchsize=128, shuffle=true)
   ```
   After this, `input |> device` in the training loop is a cheap no-op for device arrays. (`unwrap` inside `generate_native!` already calls `Flux.cpu` on the concatenated data, so the cached CPU copy used for subsampling/neighbour search is unaffected.)

2. **Bump `cf_batchsize`.** Change `cf_batchsize = 32` → `cf_batchsize = 128` (~line 109) with a comment: *"cf_batchsize is a GPU-memory knob for the CF search; 128 (= nce, so no chunking) is fastest here — lower it only on memory-constrained GPUs."*

3. **Add a short "Performance tips" paragraph** after the timing/accuracy discussion (~line 196): keep the full dataset on the device before building the DataLoader; set `cf_batchsize` as large as memory allows to avoid chunking; use `accuracy_every` (e.g. `div(nepochs,5)`) to skip per-epoch accuracy when wall-clock matters; for BN models, consider `fuse_cf_forwards=true` only if slightly different BN statistics are acceptable.

**Why it helps.** The docs are the benchmarked entry point; tuning its defaults delivers the Phase 1-2 speedups without any code change by the reader.

**Verification.** Read/re-render the qmd; confirm the example still runs (the Phase-0 harness mirrors it).

**Acceptance criterion.** Documented defaults reflect the optimized configuration (no chunking; data on device); "Performance tips" paragraph present.

---

## Phase 4 — Accuracy (`src/utils.jl`)

**File/Function.** `accuracy(model, train_set; device=identity)` (src/utils.jl, ~lines 40-51).

**Change.** Evaluate with the model in testmode and accumulate match counts on device, syncing once per call.

**Current code:**

```julia
function accuracy(model, train_set; device=identity)
    acc = 0
    for (x, y) in train_set
        x = x |> device
        logits = model(x) |> Flux.cpu
        yhat = Flux.onecold(Flux.softmax(logits))
        y_true = Flux.onecold(y)
        acc += sum(yhat .== y_true)
    end
    return acc / size(train_set.data[1], 2)
end
```

**Code sketch:**

```julia
function accuracy(model, train_set; device=identity)
    acc_dev = device([0])
    Flux.testmode!(model)
    try
        for (x, y) in train_set
            x = x |> device
            y = y |> device
            logits = model(x)
            # argmax(logits) == argmax(softmax(logits)) — softmax is monotone per column,
            # so it can be skipped without changing predictions.
            yhat = vec(argmax(logits; dims=1))
            y_true = vec(argmax(y; dims=1))
            acc_dev .+= sum(yhat .== y_true; dims=1)   # dims=1 keeps result on device (no sync)
        end
    finally
        Flux.trainmode!(model)
    end
    return Flux.cpu(acc_dev)[1] / size(train_set.data[1], 2)
end
```

> **Backend caveat.** The comment in the current code notes that `argmax`/`onecold` may scalar-index on GPU arrays. If `argmax(::GPUArray; dims=1)` errors on the active backend, keep the per-batch `logits |> Flux.cpu` fallback (one sync per batch, as today) — the important part of this step is `testmode!`, not the sync reduction. Verify with the GPU run of the test suite/benchmark before committing to the device-side path.

**Note.** This helper is shared with the old (non-native) branch. Setting testmode here is an intentional correctness fix (accuracy should use running rather than batch statistics) and may change old-branch accuracy numbers slightly — add a `CHANGELOG.md` entry under `[Unreleased]` (a CHANGELOG exists at the repo root).

**Why it helps.** Removes per-batch CPU syncs and per-batch H2D of logits/labels; measures accuracy in eval mode.

**Verification.**

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

**Acceptance criterion.** Tests pass unmodified; accuracy values are stable across runs; the device accumulator is synced exactly once per call; CHANGELOG entry added.

---

## Phase 5 — Verification

**Change.** Full-suite run + benchmark comparison.

**Verification.**

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. dev/bench_native.jl | tee dev/bench_after.txt
```

Compare against `dev/bench_baseline.txt` (Phase 0, same device). Paste the before/after numbers into a `## Results` section appended to this file.

**Acceptance criterion.**
- All testsets pass unmodified (`test/runtests.jl` includes: `loss_tests.jl`, `utils_tests.jl`, `objectives_tests.jl`, `native_helpers_tests.jl`, `native_edge_tests.jl`, `counterfactuals_tests.jl`, `training_tests.jl`, `parity_generator_tests.jl`, `parity_training_tests.jl`).
- Full-objective epoch ≈ **4-6s** (from ~15s) on the reference GPU setup.
- Convergence behaviour unchanged and final accuracy within seed noise of the Phase-0 baseline.

---

## Hand-off checklist

1. Phase 0: run `dev/bench_native.jl`, save `dev/bench_baseline.txt` (note the device).
2. Phase 1 Step 5 (testmode during search) — do FIRST if you want bitwise-stable before/after comparisons on BN models; otherwise do it after Step 4. Run the FULL test suite.
3. Phase 1 Step 1 (fast path): implement, run native+parity tests.
4. Phase 1 Step 2 (one sync): implement, test.
5. Phase 1 Step 3 (device masking): implement, test.
6. Phase 1 Step 4 (broadcast AE tracking): implement, test.
7. Phase 2 Step 6 (defer syncs; remember `sum(x; dims=1)`, not `sum(x)`): implement, test.
8. Phase 2 Step 7 (fuse_cf_forwards + helper + wiring): implement, add the BN-free fused-vs-unfused test, test.
9. Phase 3 (docs) and Phase 4 (accuracy + CHANGELOG): implement, test.
10. Phase 5: re-run benchmark, paste results into `## Results`, close out.

---

## Dependency Graph

```
Phase 0 (benchmark baseline) → all Phases
Phase 1: Steps 1→2→3→4 sequential (same function); Step 5 may go first or last within Phase 1
Phase 2: Step 6, Step 7  (independent of each other; after Phase 1)
Phase 3, Phase 4         (mutually independent; after Phase 1)
Phase 5 (verification)   (last)
```

## Out of Scope / Rejected Ideas

- **Warm-starting the CF search across epochs** — rejected: with `nce ≪ N`, the random subsample overlap between epochs is ~`nce²/N` (e.g. `128²/5000 ≈ 3` samples), so a persistent `X′` buffer almost never benefits. Revisit only if `nce == N`.
- **Switching AD backend** (e.g. Enzyme/ReverseDiff) — out of scope; separate effort, parity risk.
- **Async overlap of CF generation with training** — out of scope; complex and not needed once Phase 1 lands.
- **Device-resident `group_indices` for the `counterfactual_dl` slicing** in `generate_native!` (~lines 803-812: 5 gathers × ~40 batches/epoch with CPU index vectors) — measured impact is minor (once per epoch, tiny index arrays); deferred. Revisit only if profiling after Phase 1 shows it matters.

## Notes / Constraints

- No new dependencies may be added to `Project.toml`.
- No CUDA/AMDGPU-specific calls may appear in `src/`. All code must be array-type-agnostic and follow the existing `device`-keyword pattern (`identity`, `Flux.gpu`, or `AMDGPU.gpu`).
- Preserve the existing preallocation strategy (buffers allocated once, reused every iteration).
- Do not modify existing tests. Add new tests only (e.g. the BN-free fused-vs-unfused check in Step 7).
- Every step states whether it is parity-preserving or an intentional numeric change; see per-step acceptance criteria.
- The test suite uses only Dense (BN-free) models, so Steps 1-4 are bitwise-verifiable there; BN-model differences are confined to the ResNet benchmark/docs and are addressed by Step 5.
