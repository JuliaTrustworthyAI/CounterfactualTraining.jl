# Implementation Plan: GPU-Native Counterfactual Training

## Goal

Add a second branch to `src/` that does not depend on `CounterfactualExplanations.jl` for counterfactual generation. Instead, implement the ECCo generator from scratch with: 1) GPU support (CUDA + AMDGPU); 2) maximum performance via batched operations.

## Design decisions

- **Both branches coexist**: `CounterfactualExplanations.jl` and `TaijaParallel.jl` stay in `Project.toml`. The old MPI-based branch is untouched.
- **`Native` submodule**: All new types live in `CounterfactualTraining.Native` to avoid the name conflict with `CounterfactualExplanations.ECCoGenerator`.
- **ECCo generator only**: The default generator used in the paper. Other generators can be added later.
- **Array-type agnostic**: No explicit CUDA/AMDGPU calls. A `device` keyword (default `identity`) lets users pass `Flux.gpu` or `AMDGPU.gpu`. No GPU packages added to `Project.toml` — users add them in their environment.
- **Overload `counterfactual_training`**: Dispatch on `AbstractNativeGenerator` → native path; any CE.jl `AbstractGenerator` → old untyped fallback.

## File structure

```
src/
  CounterfactualTraining.jl     # add module Native, includes
  utils.jl                      # reuse as-is
  loss.jl                       # reuse as-is (already CE.jl-free)
  objectives.jl                 # remove unused `using CounterfactualExplanations`
  counterfactuals.jl            # keep (old branch)
  training.jl                   # keep (old branch)
  native/
    data.jl                     # NativeCFData, domain/mutability constraints
    convergence.jl              # MaxIterConvergence, DecisionThresholdConvergence
    generator.jl                # ECCoGenerator, loss/penalty/gradient functions
    counterfactuals.jl          # batched CE generation
    training.jl                  # overloaded counterfactual_training
```

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

        include("native/data.jl")
        include("native/convergence.jl")
        include("native/generator.jl")
        include("native/counterfactuals.jl")
        include("native/training.jl")  # adds method to parent's counterfactual_training

        export AbstractNativeGenerator, ECCoGenerator
        export NativeCFData
        export MaxIterConvergence, DecisionThresholdConvergence
    end
    export Native
end
```

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

    include("native/data.jl")
    include("native/convergence.jl")
    include("native/generator.jl")
    include("native/counterfactuals.jl")
    include("native/training.jl")

    export AbstractNativeGenerator, ECCoGenerator
    export NativeCFData
    export MaxIterConvergence, DecisionThresholdConvergence
end
export Native
```

Create empty stub files for now: `src/native/data.jl`, `src/native/convergence.jl`, `src/native/generator.jl`, `src/native/counterfactuals.jl`, `src/native/training.jl`. Each can start with just `# placeholder` so the module loads.

#### Step 2 — Clean up objectives.jl

In `src/objectives.jl`, remove the line `using CounterfactualExplanations`. The file only uses `Flux` and `StatsBase` — the CE.jl import is unused. Verify the package still loads.

---

### Phase 2: Data structures (`src/native/data.jl`)

#### Step 3 — Implement NativeCFData

A lightweight replacement for `CounterfactualData`:

```julia
Base.@kwdef struct NativeCFData
    X::AbstractMatrix           # D × N feature matrix
    y::Vector{Int}              # N labels (integer indices 1..C)
    domain::Vector{Tuple{Float32,Float32}}  # (lb, ub) per feature
    mutability::Union{Nothing,Vector{Symbol}}  # :both, :none, :increase, :decrease per feature
    y_levels::Vector{Int}       # unique sorted labels
end

function NativeCFData(X::AbstractMatrix, y::AbstractVector; domain=nothing, mutability=nothing)
    y_levels = sort(unique(y))
    y_int = [findfirst(==(label), y_levels) for label in y]
    if isnothing(domain)
        domain = infer_domain_constraints(X)
    end
    return NativeCFData(X, y_int, domain, mutability, y_levels)
end
```

#### Step 4 — Implement apply_domain_constraints!

```julia
function apply_domain_constraints!(X′::AbstractMatrix, domain::Vector{Tuple{Float32,Float32}})
    for i in axes(X′, 1)
        lb, ub = domain[i]
        @view(X′[i, :]) .= clamp.(@view(X′[i, :]), lb, ub)
    end
    return X′
end
```

Must work on CPU arrays, `CuArray`, and `ROCArray`. Standard `clamp.` is supported on all.

#### Step 5 — Implement apply_mutability!

```julia
function apply_mutability!(ΔX::AbstractMatrix, mutability::Union{Nothing,Vector{Symbol}})
    isnothing(mutability) && return ΔX
    for (i, dir) in enumerate(mutability)
        if dir == :none
            @view(ΔX[i, :]) .= zero(eltype(ΔX))
        elseif dir == :increase
            @view(ΔX[i, :]) .= ifelse.(@view(ΔX[i, :]) .< 0, zero(eltype(ΔX)), @view(ΔX[i, :]))
        elseif dir == :decrease
            @view(ΔX[i, :]) .= ifelse.(@view(ΔX[i, :]) .> 0, zero(eltype(ΔX)), @view(ΔX[i, :]))
        end
        # :both → no change
    end
    return ΔX
end
```

---

### Phase 3: Convergence (`src/native/convergence.jl`)

#### Step 6 — Implement convergence types and check function

```julia
abstract type AbstractConvergence end

struct MaxIterConvergence <: AbstractConvergence
    max_iter::Int
end

struct DecisionThresholdConvergence <: AbstractConvergence
    max_iter::Int
    decision_threshold::Float32
end

"""
    check_convergence(conv, probs, target_idx, iter) -> BitVector

Returns a per-sample boolean mask (length N) indicating which counterfactuals
have converged. `probs` is C × N (softmax output), `target_idx` is an N-vector
of target class indices.
"""
function check_convergence(conv::MaxIterConvergence, probs, target_idx, iter)
    return fill(iter >= conv.max_iter, length(target_idx))
end

function check_convergence(conv::DecisionThresholdConvergence, probs, target_idx, iter)
    N = length(target_idx)
    if iter >= conv.max_iter
        return trues(N)
    end
    # Per-sample: is softmax prob at target >= threshold?
    target_probs = [probs[target_idx[i], i] for i in 1:N]
    return target_probs .>= conv.decision_threshold
end
```

Note: the `target_probs` comprehension may need to be replaced with a GPU-compatible version for large batches. A batched alternative:
```julia
target_probs = gather(probs, target_idx)  # see Step 7 for helper
```

#### Step 7 — Implement batched gather helper (GPU-compatible)

```julia
"""
    gather_probs(probs, target_idx) -> Vector

Extracts `probs[target_idx[i], i]` for each sample i.
Works on CPU and GPU arrays.
"""
function gather_probs(probs::AbstractMatrix, target_idx::AbstractVector)
    N = length(target_idx)
    idx = CartesianIndex.(target_idx, 1:N)
    return probs[idx]
end
```

This uses `CartesianIndex` indexing which is supported on GPU arrays. Update `check_convergence` to use this.

---

### Phase 4: Generator (`src/native/generator.jl`)

#### Step 8 — Define generator types

```julia
abstract type AbstractNativeGenerator end

"""
    ECCoGenerator

Native implementation of the ECCo (Energy-Consistent Counterfactual) generator.
Does not depend on CounterfactualExplanations.jl. All operations are batched
and array-type agnostic (works on CPU, CuArray, ROCArray).

# Fields
- `opt`: Flux optimizer for the search (default: Descent(0.25))
- `λ`: penalty weights `[lambda_cost, lambda_energy]`
- `maxiter`: maximum search iterations
- `decision_threshold`: target class probability threshold for convergence
- `decay`: polynomial decay rate for energy constraint multiplier
- `reg_strength`: regularization strength for energy constraint
- `epsilon`: norm bound for adversarial example tracking
- `p`: norm type for adversarial example tracking (default: Inf)
"""
Base.@kwdef struct ECCoGenerator <: AbstractNativeGenerator
    opt::Flux.Optimise.AbstractOptimiser = Flux.Descent(0.25f0)
    λ::Vector{Float32} = [0.001f0, 5.0f0]
    maxiter::Int = 30
    decision_threshold::Float32 = 0.75f0
    decay::Float32 = 0.9f0
    reg_strength::Float32 = 1e-3f0
    epsilon::Float32 = 0.3f0
    p::Real = Inf
end
```

#### Step 9 — Implement energy and polynomial_decay helpers

```julia
"""
    energy(model, X′, target_idx) -> AbstractVector

Computes the energy (negative logit at target class) for each sample.
`model(X′)` is C × N, `target_idx` is an N-vector of class indices.
Returns an N-vector.
"""
function energy(model, X′::AbstractMatrix, target_idx::AbstractVector)
    logits = model(X′)  # C × N
    return -gather_probs(logits, target_idx)
end

"""
    polynomial_decay(a, b, decay, t)

Polynomial decay function as in Welling et al. (2011).
"""
polynomial_decay(a::Real, b::Real, decay::Real, t::Int) = a * (b + t)^(-decay)
```

#### Step 10 — Implement generator objective

```julia
"""
    generator_loss(gen, model, X′, X, targets_onehot, target_idx, iter)

Batched generator objective for ECCo. All operations are differentiable
and array-type agnostic.
"""
function generator_loss(
    gen::ECCoGenerator,
    model,
    X′::AbstractMatrix,
    X::AbstractMatrix,
    targets_onehot::AbstractMatrix,
    target_idx::AbstractVector,
    iter::Int,
)
    # Classification loss
    ℓ = Flux.logitcrossentropy(model(X′), targets_onehot)

    # L1 distance penalty
    h1 = gen.λ[1] * sum(abs, X′ .- X)

    # Energy constraint with polynomial decay multiplier
    maxiter = gen.maxiter
    ϕ = polynomial_decay(maxiter / 250, maxiter / 25, gen.decay, iter + 1)
    e = energy(model, X′, target_idx)  # N-vector
    gen_loss = mean(e)
    reg_loss = sum(abs2, e)
    h2 = gen.λ[2] * ϕ * (gen_loss + gen.reg_strength * reg_loss)

    return ℓ + h1 + h2
end
```

---

### Phase 5: Batched counterfactual generation (`src/native/counterfactuals.jl`)

#### Step 11 — Implement find_neighbours

```julia
using Random
using StatsBase: sample

"""
    find_neighbours(X, y, targets, y_levels; nneighbours=1, rng=Random.default_rng())

For each counterfactual with target `targets[i]`, samples `nneighbours` random
data points from the training set that have label `targets[i]`.
Returns a D × N matrix (one neighbour per counterfactual column).
"""
function find_neighbours(
    X::AbstractMatrix,
    y::Vector{Int},
    targets::Vector{Int},
    y_levels::Vector{Int};
    nneighbours::Int = 1,
    rng = Random.default_rng(),
)
    D, N = size(X)
    neighbours = similar(X, D, N)
    for i in 1:N
        target = targets[i]
        target_idx_in_y = findfirst(==(target), y_levels)
        # Find all training samples with this target label:
        candidates = findall(==(target_idx_in_y), y)
        if isempty(candidates)
            # Fallback: use a random sample
            chosen = rand(rng, 1:size(X, 2))
        else
            chosen = rand(rng, candidates)
        end
        neighbours[:, i] = X[:, chosen]
    end
    return neighbours
end
```

#### Step 12 — Implement protect_immutable! (batched)

```julia
"""
    protect_immutable!(neighbours, counterfactuals, mutability)

Protects immutable features by setting neighbour values according to mutability
directions. For each feature row:
- `:both` → keep neighbour value
- `:none` → use counterfactual value (no change)
- `:increase` → max(counterfactual, neighbour)
- `:decrease` → min(counterfactual, neighbour)
"""
function protect_immutable!(
    neighbours::AbstractMatrix,
    counterfactuals::AbstractMatrix,
    mutability::Union{Nothing,Vector{Symbol}},
)
    isnothing(mutability) && return neighbours
    for (j, dir) in enumerate(mutability)
        if dir == :none
            neighbours[j, :] = counterfactuals[j, :]
        elseif dir == :increase
            neighbours[j, :] = max.(counterfactuals[j, :], neighbours[j, :])
        elseif dir == :decrease
            neighbours[j, :] = min.(counterfactuals[j, :], neighbours[j, :])
        end
        # :both → keep neighbour as-is
    end
    return neighbours
end
```

#### Step 13 — Implement generate_counterfactuals_batch

This is the core function. All operations are batched on `D × N` matrices.

```julia
using Flux

"""
    generate_counterfactuals_batch(
        model, X, targets, data, gen;
        convergence=DecisionThresholdConvergence(gen.maxiter, gen.decision_threshold),
        verbose=1,
    )

Generates counterfactuals for all N factuals simultaneously using batched
gradient descent. Returns (counterfactuals, last_valid_ae, converged_mask, maxiter).
"""
function generate_counterfactuals_batch(
    model,
    X::AbstractMatrix,
    targets::Vector{Int},
    data::NativeCFData,
    gen::ECCoGenerator;
    convergence = DecisionThresholdConvergence(gen.maxiter, gen.decision_threshold),
    verbose::Int = 1,
)
    D, N = size(X)

    # Target indices into y_levels
    target_idx = [findfirst(==(t), data.y_levels) for t in targets]
    targets_onehot = Flux.onehotbatch(target_idx, 1:length(data.y_levels))

    # Initialize counterfactuals at factuals (identity init)
    X′ = copy(X)
    last_valid_ae = copy(X)
    converged_mask = falses(N)

    # Optimizer state for the counterfactual parameters
    opt_state = Flux.setup(gen.opt, X′)

    for iter in 1:gen.maxiter
        # 1. Compute gradient of generator objective w.r.t. X'
        _, grads = Flux.withgradient(X′) do x′
            generator_loss(gen, model, x′, X, targets_onehot, target_idx, iter)
        end

        # 2. Apply mutability constraints to gradient
        apply_mutability!(grads[1], data.mutability)

        # 3. Update counterfactuals
        Flux.update!(opt_state, X′, grads[1])

        # 4. Apply domain constraints (clamp to bounds)
        apply_domain_constraints!(X′, data.domain)

        # 5. Track last valid adversarial examples
        perturbations = X′ .- X
        if gen.p == Inf
            norms = vec(maximum(abs, perturbations; dims = 1))
        else
            norms = vec(sum(abs .^ gen.p, perturbations; dims = 1) .^ (1 / gen.p))
        end
        valid_ae = norms .<= gen.epsilon
        last_valid_ae[:, valid_ae] = X′[:, valid_ae]

        # 6. Check convergence
        probs = Flux.softmax(model(X′))
        new_converged = check_convergence(convergence, probs, target_idx, iter)
        converged_mask .|= new_converged

        # 7. Early exit if all converged
        all(converged_mask) && break
    end

    return X′, last_valid_ae, converged_mask, gen.maxiter
end
```

#### Step 14 — Implement generate_native! (top-level, replaces generate!)

```julia
using StatsBase: sample

"""
    generate_native!(
        model, train_set, gen::ECCoGenerator;
        nsamples=nothing,
        nneighbours=1,
        convergence,
        domain=nothing,
        mutability=nothing,
        verbose=1,
    )

Top-level counterfactual generation. Generates counterfactuals for a subset of
the training data, finds neighbours, applies mutability protection, and
partitions results into a data loader aligned with train_set batches.

Returns (dl, percent_valid, nothing) — same interface as the old generate!().
"""
function generate_native!(
    model,
    train_set,
    gen::ECCoGenerator;
    nsamples::Union{Nothing,Int} = nothing,
    nneighbours::Int = 1,
    convergence = DecisionThresholdConvergence(gen.maxiter, gen.decision_threshold),
    domain = nothing,
    mutability = nothing,
    verbose::Int = 1,
)
    # Unwrap training data
    X, y_raw = unwrap(train_set)

    # Build NativeCFData
    data = NativeCFData(X, y_raw; domain = domain, mutability = mutability)

    # Determine sample size
    N = size(X, 2)
    nsamples = isnothing(nsamples) ? N : min(nsamples, N)
    if nsamples < length(train_set)
        @warn "Need at least one counterfactual per batch. Setting nsamples=$(length(train_set))." maxlog = 1
        nsamples = length(train_set)
    end

    # Subsample factuals
    if nsamples < N
        idx_sub = StatsBase.sample(1:N, nsamples; replace = false)
        X_sub = X[:, idx_sub]
    else
        idx_sub = collect(1:N)
        X_sub = X
    end

    # Predict factual labels and assign random targets
    factual_preds = vec(Flux.onecold(model(X_sub)))
    targets = Vector{Int}(undef, nsamples)
    for i in 1:nsamples
        available = setdiff(data.y_levels, factual_preds[i])
        targets[i] = rand(available)
    end

    # Generate counterfactuals (batched)
    counterfactuals, last_valid_ae, converged_mask, maxiter = generate_counterfactuals_batch(
        model, X_sub, targets, data, gen; convergence = convergence, verbose = verbose
    )

    # Find neighbours in target class
    neighbours = find_neighbours(X, data.y, targets, data.y_levels; nneighbours = nneighbours)

    # Protect immutable features
    protect_immutable!(neighbours, counterfactuals, data.mutability)

    # One-hot encodings
    targets_enc = [Flux.onehot(t, 1:length(data.y_levels)) for t in targets]
    factual_enc = [Flux.onehot(y_raw[idx_sub[i]], 1:length(data.y_levels)) for i in 1:nsamples]

    # Validity
    percent_valid = sum(converged_mask) / nsamples

    # Partition into batch-aligned data loader
    # Same 5-tuple format as old generate!:
    # (counterfactuals, advexms, targets_enc, neighbours, factual_enc)
    group_indices = split_obs(1:nsamples, length(train_set))
    dl = [
        (
            stack(hcat, counterfactuals[:, group_indices[i]]...)),
            stack(hcat, last_valid_ae[:, group_indices[i]]...)),
            stack(hcat, targets_enc[group_indices[i]]...)),
            stack(hcat, neighbours[:, group_indices[i]]...)),
            stack(hcat, factual_enc[group_indices[i]]...)),
        ) for i in eachindex(group_indices)
    ]

    return dl, percent_valid, nothing
end

# Helper (replaces TaijaParallel.split_obs)
function split_obs(indices::AbstractRange, n::Int)
    return [collect(part) for part in Base.Iterators.partition(indices, max(1, length(indices) ÷ n))]
end
```

---

### Phase 6: Native training loop (`src/native/training.jl`)

#### Step 15 — Implement accuracy_gpu

```julia
"""
    accuracy_gpu(model, train_set)

GPU-compatible accuracy. Uses Flux.onecold on the whole matrix instead of
per-column list comprehension.
"""
function accuracy_gpu(model, train_set)
    acc = 0
    for (x, y) in train_set
        yhat = Flux.onecold(Flux.softmax(model(x)))
        y_true = Flux.onecold(y)
        acc += sum(yhat .== y_true)
    end
    return acc / size(train_set.data[1], 2)
end
```

#### Step 16 — Implement overloaded counterfactual_training

This adds a new method to the existing `CounterfactualTraining.counterfactual_training`
function by dispatching on `AbstractNativeGenerator`.

```julia
using ChainRulesCore: ChainRulesCore
using Flux
using JLD2
using ProgressMeter
using UnicodePlots

"""
    counterfactual_training(
        loss::AbstractObjective,
        model,
        generator::AbstractNativeGenerator,
        train_set,
        opt_state;
        device = identity,
        val_set = nothing,
        nepochs = 100,
        burnin = 0.0,
        nce = nothing,
        nneighbours = 100,
        convergence = DecisionThresholdConvergence(),
        domain = nothing,
        mutability = nothing,
        verbose = 1,
        checkpoint_dir = nothing,
        callback = nothing,
    )

Native GPU-compatible counterfactual training. Dispatches here when the generator
is an AbstractNativeGenerator (e.g., ECCoGenerator from the Native submodule).

The `device` keyword is a function: `identity` (CPU), `Flux.gpu` (CUDA),
or `AMDGPU.gpu` (AMDGPU). The model is moved to the device; training data
should already be on the device (user moves it before constructing the DataLoader).
"""
function counterfactual_training(
    loss::AbstractObjective,
    model,
    generator::AbstractNativeGenerator,
    train_set,
    opt_state;
    device = identity,
    val_set = nothing,
    nepochs::Int = 100,
    burnin = 0.0f0,
    nce::Union{Nothing,Int} = nothing,
    nneighbours::Int = 100,
    convergence = DecisionThresholdConvergence(generator.maxiter, generator.decision_threshold),
    domain = nothing,
    mutability = nothing,
    verbose::Int = 1,
    checkpoint_dir::Union{Nothing,String} = nothing,
    callback::Union{Nothing,Function} = nothing,
    kwrgs...,
)
    # Move model to device
    model = model |> device
    opt_state = Flux.setup(typeof(opt_state).parameters[1] |> device, model)

    # Setup
    burnin = Int(round(burnin * nepochs))
    nce = isnothing(nce) ? length(train_set) : nce
    nce_per_batch = Int(ceil(nce / length(train_set)))
    nce_batch_ratio = nce_per_batch / train_set.batchsize

    log = []
    start_epoch = 1

    # Checkpoint loading
    if !isnothing(checkpoint_dir) && isfile(joinpath(checkpoint_dir, "checkpoint.jld2"))
        @info "Found checkpoint file in $checkpoint_dir. Loading..."
        try
            _model, _opt_state, epoch, log = JLD2.load(
                joinpath(checkpoint_dir, "checkpoint.jld2"),
                "model", "opt_state", "epoch", "log",
            )
            model = _model |> device
            opt_state = _opt_state
            start_epoch = epoch + 1
            if start_epoch <= nepochs
                @info "Resuming training from epoch $start_epoch."
            else
                @info "Already completed 100% of training. Skipping..."
            end
        catch
            @warn "Could not load checkpoint. Starting training from scratch."
        end
    end

    if verbose in [1, 2]
        p = Progress(nepochs - start_epoch; barglyphs = BarGlyphs("[=> ]"), color = :yellow)
    end

    for epoch in start_epoch:nepochs
        losses = Float32[]
        implausibilities = Float32[]
        reg_losses = Float32[]
        validity_losses = Float32[]
        start = time()

        # Generate counterfactuals (batched, on device)
        if epoch > burnin && needs_counterfactuals(loss)
            counterfactual_dl, percent_valid, _ = generate_native!(
                model, train_set, generator;
                nsamples = nce,
                nneighbours = nneighbours,
                convergence = convergence,
                domain = domain,
                mutability = mutability,
                verbose = verbose,
            )
            avg_iter = generator.maxiter
        else
            counterfactual_dl = fill(ntuple(_ -> nothing, 5), length(train_set))
            percent_valid = 1.0f0
            avg_iter = 0
        end

        # Backprop
        for (i, batch) in enumerate(train_set)
            input, label = batch
            input = input |> device
            label = label |> device
            perturbed_input, advexms, targets_enc, neighbours, factual_enc = counterfactual_dl[i]

            val, grads = Flux.withgradient(model) do m
                logits = m(input)

                if !isnothing(perturbed_input)
                    perturbed_input = perturbed_input |> device
                    advexms = advexms |> device
                    targets_enc = targets_enc |> device
                    neighbours = neighbours |> device
                    factual_enc = factual_enc |> device

                    implaus = implausibility(m, perturbed_input, neighbours, targets_enc)
                    regs = reg_loss(m, perturbed_input, neighbours, targets_enc)
                    adversarial_loss = loss.class_loss(m(advexms), factual_enc)
                else
                    implaus = [0.0f0]
                    regs = [0.0f0]
                    adversarial_loss = 0.0f0
                end

                ChainRulesCore.ignore_derivatives() do
                    push!(implausibilities, sum(implaus) / length(implaus))
                    push!(reg_losses, sum(regs) / length(regs))
                    push!(validity_losses, adversarial_loss)
                end

                return loss(logits, label, implaus, regs, adversarial_loss)
            end

            push!(losses, val)

            if !isfinite(val)
                @warn "loss is $val on item $i" epoch
                continue
            end

            Flux.update!(opt_state, model, grads[1])
        end

        # Logging
        time_taken = time() - start
        acc = accuracy_gpu(model, train_set)
        acc_val = isnothing(val_set) ? nothing : accuracy_gpu(model, val_set)
        train_loss = sum(losses) / length(losses)

        if epoch > burnin
            implaus = sum(implausibilities) / length(implausibilities)
            log_reg_loss = sum(reg_losses) / length(reg_losses)
            log_adv_loss = sum(validity_losses) / length(validity_losses)
        else
            implaus = nothing
            log_reg_loss = nothing
            log_adv_loss = nothing
        end

        push!(log, (;
            acc, acc_val, train_loss, implaus, log_reg_loss, log_adv_loss,
            time_taken, percent_valid, avg_iter,
        ))

        # Checkpointing (save CPU version)
        if !isnothing(checkpoint_dir)
            jldsave(
                joinpath(checkpoint_dir, "checkpoint.jld2");
                model = model |> Flux.cpu,
                opt_state = opt_state,
                epoch, log,
            )
        end

        if verbose in [1, 2]
            next!(p)
        end
    end

    return model, log
end
```

---

### Phase 7: Tests

#### Step 17 — Unit tests for data structures

In `test/runtests.jl`, add:

```julia
@testset "Native data structures" begin
    using CounterfactualTraining.Native
    X = randn(2, 100)
    y = rand(1:3, 100)
    data = NativeCFData(X, y)
    @test length(data.y_levels) == 3
    @test length(data.domain) == 2

    # Domain constraints
    X′ = copy(X)
    X′[1, :] .= 100.0
    apply_domain_constraints!(X′, data.domain)
    @test all(X′[1, :] .<= data.domain[1][2])

    # Mutability
    ΔX = ones(2, 10)
    apply_mutability!(ΔX, [:decrease, :none])
    @test all(ΔX[1, :] .== 0)
    @test all(ΔX[2, :] .== 0)
end
```

#### Step 18 — Unit tests for generator

```julia
@testset "Native generator" begin
    using CounterfactualTraining.Native
    model = Flux.Chain(Dense(2, 3))
    X = randn(2, 10)
    gen = ECCoGenerator()
    data = NativeCFData(X, ones(10))
    target_idx = fill(2, 10)

    # Energy
    e = energy(model, X, target_idx)
    @test length(e) == 10

    # Polynomial decay
    @test polynomial_decay(1.0, 1.0, 0.9, 1) ≈ 1.0 * 2.0^(-0.9)

    # Generator loss is finite and differentiable
    targets_onehot = Flux.onehotbatch(target_idx, 1:3)
    loss_val = generator_loss(gen, model, X, X, targets_onehot, target_idx, 1)
    @test isfinite(loss_val)

    _, grads = Flux.withgradient(X) do x′
        generator_loss(gen, model, x′, X, targets_onehot, target_idx, 1)
    end
    @test size(grads[1]) == size(X)
end
```

#### Step 19 — Unit tests for batched CE generation

```julia
@testset "Batched CE generation" begin
    using CounterfactualTraining.Native
    # Simple 2-class problem
    X = hcat(randn(2, 50), randn(2, 50) .+ 3)
    y = vcat(fill(1, 50), fill(2, 50))
    model = Flux.Chain(Dense(2, 4, relu), Dense(4, 2))
    data = NativeCFData(X, y)
    gen = ECCoGenerator(maxiter = 50)
    targets = fill(2, 20)  # flip class 1 to class 2

    cfs, advexms, converged, maxiter = generate_counterfactuals_batch(
        model, X[:, 1:20], targets, data, gen
    )
    @test size(cfs) == (2, 20)
    @test size(advexms) == (2, 20)
    @test length(converged) == 20
end
```

#### Step 20 — Integration test: native training (CPU)

```julia
@testset "Native training (CPU)" begin
    using CounterfactualTraining.Native
    # Moons-like data
    X = hcat(randn(2, 100), (randn(2, 100) .+ [3.0, -1.0]))
    y = vcat(fill(1, 100), fill(2, 100))
    train_set = Flux.DataLoader((Float32.(X), Flux.onehotbatch(y, 1:2)); batchsize = 32)
    model = Flux.Chain(Dense(2, 8, relu), Dense(8, 2))
    opt_state = Flux.setup(Flux.Adam(1e-3), model)
    gen = ECCoGenerator(maxiter = 20)
    obj = VanillaObjective()
    model, log = counterfactual_training(obj, model, gen, train_set, opt_state; nepochs = 5, verbose = 0)
    @test length(log) == 5
    @test log[end].acc > 0.5
end
```

#### Step 21 — Integration test: native training (GPU)

```julia
@testset "Native training (GPU)" begin
    using CounterfactualTraining.Native
    has_gpu = false
    device = identity
    try
        using CUDA
        if CUDA.functional()
            has_gpu = true
            device = Flux.gpu
        end
    catch
        try
            using AMDGPU
            if AMDGPU.functional()
                has_gpu = true
                device = AMDGPU.gpu
            end
        catch
        end
    end

    if !has_gpu
        @info "No GPU available, skipping GPU test"
        return
    end

    X = hcat(randn(2, 100), (randn(2, 100) .+ [3.0, -1.0]))
    y = vcat(fill(1, 100), fill(2, 100))
    X_gpu = Float32.(X) |> device
    y_gpu = Flux.onehotbatch(y, 1:2) |> device
    train_set = Flux.DataLoader((X_gpu, y_gpu); batchsize = 32)
    model = (Flux.Chain(Dense(2, 8, relu), Dense(8, 2)) |> device)
    opt_state = Flux.setup(Flux.Adam(1e-3), model)
    gen = ECCoGenerator(maxiter = 20)
    obj = VanillaObjective()
    model, log = counterfactual_training(obj, model, gen, train_set, opt_state; nepochs = 5, verbose = 0, device = device)
    @test length(log) == 5
    @test log[end].acc > 0.5
end
```

---

### Phase 8: Documentation

#### Step 22 — Update README

Add a section documenting the Native submodule, ECCoGenerator, and `device` keyword.

#### Step 23 — Add CHANGELOG entry

```markdown
## [Unreleased]

### Added

- `Native` submodule with GPU-compatible counterfactual generation
- `ECCoGenerator` — native implementation of the ECCo generator
- `counterfactual_training` overload dispatching on `AbstractNativeGenerator`
- `device` keyword for GPU support (CUDA and AMDGPU)
```

---

## Dependency graph

```
Step 1 (module skeleton) → all others
Step 2 (objectives cleanup) — independent

Phase 2 (data): Step 3 → Step 4, Step 5
Phase 3 (convergence): Step 6 → Step 7
Phase 4 (generator): Step 8 → Step 9 → Step 10
Phase 5 (CE gen): Steps 3-7, 9-10 → Step 11 → Step 12 → Step 13 → Step 14
Phase 6 (training): Steps 13, 14 → Step 15 → Step 16
Phase 7 (tests): Step 16 → Steps 17-21
Phase 8 (docs): All → Steps 22-23
```

Parallelizable groups:
- Steps 3-5 (data), Steps 6-7 (convergence), Steps 8-10 (generator) can be done in parallel
- Step 11 (neighbours) depends only on data structures
- Step 13 (core CE generation) is the central piece — depends on all of phases 2-4
- Step 16 (training loop) depends on everything

---

## Implementation notes

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
