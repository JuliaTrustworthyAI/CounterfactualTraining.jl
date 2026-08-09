# Implementation Plan: GPU-Native Counterfactual Training

## Goal

Add a second branch to `src/` that uses CounterfactualExplanations.jl's components (e.g. `ECCoGenerator`, `generator_loss`, `search` utilities) directly in a GPU-compatible way, **without** allocating a full `CounterfactualExplanation` object per counterfactual. The key changes: 1) batched counterfactual generation on a single `D × N` matrix instead of iterating per sample; 2) GPU support (CUDA + AMDGPU); 3) removing the `CounterfactualExplanation` struct overhead.

## Design decisions

- **Both branches coexist**: `CounterfactualExplanations.jl` and `TaijaParallel.jl` stay in `Project.toml`. The old MPI-based branch is untouched.
- **`Native` submodule**: The `Native` submodule wraps CE.jl's `ECCoGenerator` with a batched, GPU-compatible search loop. It does **not** reimplement convergence, generator types, or loss from CE.jl — those are imported directly from `CounterfactualExplanations`.
- **ECCo generator only**: The default generator used in the paper. We use CE.jl's `ECCoGenerator` directly, not a reimplementation. Other generators can be added later.
- **Array-type agnostic**: No explicit CUDA/AMDGPU calls. A `device` keyword (default `identity`) lets users pass `Flux.gpu` or `AMDGPU.gpu`. No GPU packages added to `Project.toml` — users add them in their environment.
- **Overload `counterfactual_training`**: Dispatch on CE.jl's `AbstractGenerator` → native path (batched, GPU-compatible); the old untyped method remains as the fallback for the MPI-based branch.

## File structure

```
src/
  CounterfactualTraining.jl     # add module Native, includes
  utils.jl                      # reuse as-is
  loss.jl                       # reuse as-is
  objectives.jl                 # (already cleaned up)
  counterfactuals.jl            # keep (old branch)
  training.jl                   # keep (old branch)
  native/
    training.jl                 # overloaded counterfactual_training, batched CE generation
```

The idea: we don't need separate `data.jl`, `convergence.jl`, `generator.jl`, `counterfactuals.jl` files because we import those from CE.jl. We only need `native/training.jl` which contains the overloaded `counterfactual_training` and the batched generation helper.

## Module structure

```julia
module CounterfactualTraining
    # ... existing includes ...
    include("training.jl")  # old branch: counterfactual_training (untyped generator)

    module Native
        import ..CounterfactualTraining: counterfactual_training
        import ..CounterfactualTraining: AbstractObjective
        import ..CounterfactualTraining: implausibility, reg_loss
        import ..CounterfactualTraining: infer_domain_constraints, unwrap
        import ..CounterfactualTraining: VanillaObjective, EnergyDifferentialObjective
        import ..CounterfactualTraining: AdversarialObjective, FullObjective

        using CounterfactualExplanations
        import CounterfactualExplanations: AbstractGenerator, ECCoGenerator
        using Flux

        include("native/training.jl")  # adds method to parent's counterfactual_training
    end
    export Native
end
```

The `Native` submodule exports nothing of its own — the overloaded `counterfactual_training` method is reached through dispatch on CE.jl's `AbstractGenerator`, so users call `CounterfactualTraining.counterfactual_training` with a CE.jl generator and the native method is selected automatically.

## Algorithm reference (ECCo generator, batched)

For a batch of `N` factuals (`X` is `D × N`, `D` features):

**Objective** minimized w.r.t. `X'` (counterfactuals):

```
L(X') = logitcrossentropy(model(X'), targets_onehot)       # classification
      + λ[1] * sum(|X' - X|)                                # L1 distance penalty
      + λ[2] * ϕ(iter) * [mean(energy(model, X', t))        # energy constraint
                           + reg * sum(energy(model, X', t)²)]

where:
  energy(model, X', t) = -model(X')[t, :]                  # negative logit at target, N-vector
  ϕ(iter) = polynomial_decay(maxiter/250, maxiter/25, decay, iter+1)
  polynomial_decay(a, b, decay, t) = a * (b + t)^(-decay)
```

**Search loop** (all batched on `D × N` matrices):

1. `X' = copy(X)` (identity init)
2. For `iter = 1:maxiter`:
   a. Compute gradient of `L` w.r.t. `X'` via `Flux.withgradient`
   b. Apply mutability constraints to gradient (zero out immutable directions)
   c. Update: `Flux.update!(opt_state, X', grads)`
   d. Clamp to domain bounds
   e. Track last valid adversarial example (perturbation within norm bound `ε`)
   f. Check convergence (decision threshold or max iter) — per-sample mask
   g. Early exit if all converged

**Training loss components** (reused from existing `loss.jl`):

- `implausibility(model, cf, neighbours, targets)` = `(E(neighbours) - E(cf))` at target
- `reg_loss(model, cf, neighbours, targets)` = `(|model(neighbours)|² + |model(cf)|²)` at target
- `adversarial_loss` = `logitcrossentropy(model(advexms), factual_labels)`

---

## Step-by-step plan

Each step is self-contained and can be implemented independently (respecting the dependency graph at the end).

---

### Phase 1: Project structure

#### Step 1 — Create directory and module skeleton

Create `src/native/` directory.

In `src/CounterfactualTraining.jl`, after the existing `include("training.jl")` line, add the `Native` submodule:

```julia
module Native
    import ..CounterfactualTraining: counterfactual_training
    import ..CounterfactualTraining: AbstractObjective
    import ..CounterfactualTraining: implausibility, reg_loss
    import ..CounterfactualTraining: infer_domain_constraints, unwrap
    import ..CounterfactualTraining: VanillaObjective, EnergyDifferentialObjective
    import ..CounterfactualTraining: AdversarialObjective, FullObjective

    using CounterfactualExplanations
    import CounterfactualExplanations: AbstractGenerator, ECCoGenerator
    using Flux

    include("native/training.jl")
end
export Native
```

Create one stub file: `src/native/training.jl`. It can start with just `# placeholder` so the module loads.

#### Step 2 — Clean up objectives.jl

In `src/objectives.jl`, remove the line `using CounterfactualExplanations` if present. The file only uses `Flux` and `StatsBase` — the CE.jl import is unused. Verify the package still loads.

---

### Phase 2: Batched counterfactual generation (`src/native/training.jl`)

#### Step 3 — Implement batched `generate_counterfactuals!`

Implement a batched `generate_counterfactuals!` in `native/training.jl` that calls CE.jl's `ECCoGenerator` components (e.g. `generator_loss`, `energy`, `polynomial_decay`) but operates on `D × N` matrices and **without** creating a `CounterfactualExplanation` object.

The function should:
- Take a `D × N` matrix of factuals `X`, target indices, domain/mutability constraints, and a CE.jl `ECCoGenerator`.
- Run the batched search loop described in the algorithm reference above.
- Return `(counterfactuals, last_valid_ae, converged_mask, maxiter)`.

All operations use standard Julia broadcasting / `Flux` primitives that are overloaded for GPU arrays. No explicit `using CUDA` or `using AMDGPU` — the `device` keyword handles transfer.

#### Step 4 — Implement overloaded `counterfactual_training`

Implement the overloaded `counterfactual_training` in `native/training.jl` dispatching on `AbstractGenerator` (from CE.jl). The dispatch type should be CE.jl's `AbstractGenerator`, **not** a custom `AbstractNativeGenerator`. The method simply calls the batched `generate_counterfactuals!` from Step 3 instead of the old `generate!` / `TaijaParallel.parallelize` path.

Signature:

```julia
function counterfactual_training(
    loss::AbstractObjective,
    model,
    generator::AbstractGenerator,
    train_set,
    opt_state;
    device = identity,
    val_set = nothing,
    nepochs = 100,
    burnin = 0.0f0,
    nce = nothing,
    nneighbours = 100,
    convergence = nothing,
    domain = nothing,
    mutability = nothing,
    verbose = 1,
    checkpoint_dir = nothing,
    callback = nothing,
    kwrgs...,
)
```

The `device` keyword is a function: `identity` (CPU), `Flux.gpu` (CUDA), or `AMDGPU.gpu` (AMDGPU). The model is moved to the device; training data should already be on the device (user moves it before constructing the DataLoader).

The body mirrors the old `counterfactual_training` in `src/training.jl`, but:
- Calls `generate_counterfactuals!` (batched) instead of `generate!` (MPI-parallelised).
- Uses a GPU-compatible `accuracy` helper (whole-matrix `Flux.onecold` instead of per-column list comprehension).
- Moves counterfactual tensors to the device inside the backprop loop.

---

### Phase 3: Tests

#### Step 5 — Unit tests for batched CE generation

In `test/runtests.jl`, add a testset that exercises `generate_counterfactuals!` on a small 2-class problem. Import `ECCoGenerator` from CE.jl, construct a simple `Flux.Chain` model, and verify the output shapes and that the loss is finite.

#### Step 6 — Integration test: native training (CPU)

Add a testset that runs `counterfactual_training` with a CE.jl `ECCoGenerator` on CPU for a few epochs. Verify the log length and that accuracy improves above chance.

#### Step 7 — Integration test: native training (GPU)

Add a testset that runs `counterfactual_training` with `device=Flux.gpu` (or `AMDGPU.gpu`) when a GPU is available. Skip gracefully otherwise.

---

### Phase 4: Documentation

#### Step 8 — Update README

Add a section documenting the `Native` submodule, the batched generation path, and the `device` keyword for GPU support (CUDA and AMDGPU).

#### Step 9 — Add CHANGELOG entry

```markdown
## [Unreleased]

### Added

- `Native` submodule with GPU-compatible, batched counterfactual generation
- `counterfactual_training` overload dispatching on CE.jl's `AbstractGenerator`
- `device` keyword for GPU support (CUDA and AMDGPU)
- Batched `generate_counterfactuals!` using CE.jl's `ECCoGenerator` components
  without allocating a `CounterfactualExplanation` per sample
```

---

## Dependency graph

```
Step 1 (module skeleton) → all others
Step 2 (objectives cleanup) — independent

Phase 2 (CE gen + training): Step 1 → Step 3 → Step 4
Phase 3 (tests): Step 4 → Steps 5, 6, 7
Phase 4 (docs): All → Steps 8, 9
```

Parallelizable groups:
- Step 2 (objectives cleanup) is fully independent and can be done in parallel with Step 1.
- Step 3 (batched generation) is the central piece — depends only on the module skeleton.
- Step 4 (training loop) depends on Step 3.
- Tests (Steps 5-7) depend on Step 4.

---

## Implementation notes

### Reusing CE.jl directly

The `Native` submodule imports CE.jl's `AbstractGenerator` and `ECCoGenerator`. It does **not** redefine these types. The batched `generate_counterfactuals!` calls CE.jl's `generator_loss`, `energy`, and `polynomial_decay` utilities directly — these are already array-type agnostic and work on GPU arrays. The only thing we remove is the `CounterfactualExplanation` struct overhead (full search history, gradients, etc.) which is wasteful inside the training loop.

### GPU array compatibility

All array operations use standard Julia functions that are overloaded for GPU arrays:
- `copy`, `clamp.`, `abs`, `abs2`, `max.`, `min.`, `sum`, `mean`, `maximum`
- `Flux.softmax`, `Flux.onehotbatch`, `Flux.onecold`, `Flux.logitcrossentropy`
- `Flux.setup`, `Flux.update!`, `Flux.withgradient`
- `similar` (allocates on the same device as the source array)

No explicit `using CUDA` or `using AMDGPU` in `src/`. The `device` keyword handles transfer.

### Performance considerations

1. **Batching is the main win**: all N counterfactuals are generated as a single `D × N` matrix operation. The model forward pass `model(X')` processes all N samples at once.

2. **Avoid `eachcol` / list comprehensions**: use `Flux.onecold(matrix)` instead of `[argmax(x) for x in eachcol(matrix)]`.

3. **`implausibility` and `reg_loss` optimization** (optional, can be done as follow-up): the current implementations in `loss.jl` build an `N × N` matrix via `(C×N)' * (C×N)`. An optimized version:
   ```julia
   function implausibility(model, cf, samples, targets)
       diff = (-model(samples)) - (-model(cf))  # C × N
       return vec(sum(diff .* targets; dims=1))    # N-vector, O(N) memory
   end
   ```

4. **Early exit**: the search loop breaks when all counterfactuals have converged, avoiding unnecessary iterations.

5. **Checkpointing**: models are saved as CPU copies (`model |> Flux.cpu`) to ensure portability across devices.

### What the arXiv paper adds (if accessible)

The paper (https://arxiv.org/abs/2601.16205) returned HTTP 403 from the development environment. The algorithm was fully reconstructed from:
1. Source code in `src/` (with `# ----- PAPER REF -----` comments)
2. CounterfactualExplanations.jl source on the `counterfactual-training` branch
3. The `CTExperiments` helper package from git history

No critical context is missing for this implementation.
