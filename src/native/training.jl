# Batched counterfactual generation for the Native submodule.
#
# This file implements a GPU-compatible, batched counterfactual search loop
# that operates on D×N matrices without allocating a CounterfactualExplanation
# per sample.  It uses CE.jl's `CounterfactualData` for domain/mutability
# constraints and CE.jl's `polynomial_decay` utility, but defines its own
# lightweight generator struct to avoid the Flux 0.16 incompatibility in CE.jl's
# `GradientBasedGenerator` (whose `opt` field is typed against the old
# `Flux.Optimise.AbstractOptimiser`).

using ChainRulesCore: ChainRulesCore
using JLD2
using ProgressMeter
using StatsBase
using UnicodePlots
using Random

# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
"""
    NativeGenerator

Lightweight generator for batched counterfactual search.  Holds only the
fields needed by [`generate_counterfactuals!`](@ref): penalty weights `λ`
and a Flux optimiser `opt`.

This avoids the Flux 0.16 incompatibility in CE.jl's
`GradientBasedGenerator` (whose `opt` field is typed against the old
`Flux.Optimise.AbstractOptimiser`).

# Keyword fields
- `λ::Vector{Float32}` — penalty weights `[λ₁, λ₂]` for the L1 distance
  penalty and the energy constraint (default `[0.1f0, 1.0f0]`).
- `opt` — a Flux optimiser (default `Flux.Descent(0.1f0)`).
"""
Base.@kwdef struct NativeGenerator
    λ::Vector{Float32} = [0.1f0, 1.0f0]
    opt = Flux.Descent(0.1f0)
end

# ---------------------------------------------------------------------------
# Helper: batched energy from pre-computed logits
# ---------------------------------------------------------------------------
"""
    batched_energy_from_logits(logits::AbstractMatrix, target_idx::AbstractVector{Int})

Returns a length-`N` vector of negative logits at the target class for each
sample, given the `C×N` logits matrix.  Uses linear indexing for GPU
compatibility.

This is the core indexing logic extracted from [`batched_energy`](@ref) so
that callers who already have `logits = model(X′)` can avoid a redundant
forward pass.
"""
function batched_energy_from_logits(logits::AbstractMatrix, target_idx::AbstractVector{Int})
    N = length(target_idx)
    C = size(logits, 1)
    linear_idx = (0:(N - 1)) .* C .+ target_idx
    return -logits[linear_idx]
end

# ---------------------------------------------------------------------------
# Helper: batched energy (convenience wrapper that calls the model)
# ---------------------------------------------------------------------------
"""
    batched_energy(model, X′::AbstractMatrix, target_idx::AbstractVector)

Returns a length-`N` vector of negative logits at the target class for each
sample in `X′`.  Calls `model(X′)` internally and delegates to
[`batched_energy_from_logits`](@ref).
"""
function batched_energy(model, X′::AbstractMatrix, target_idx::AbstractVector{Int})
    return batched_energy_from_logits(model(X′), target_idx)
end

# ---------------------------------------------------------------------------
# Preparer: precompute mutability masks (hoisted out of the search loop)
# ---------------------------------------------------------------------------
"""
    prepare_mutability_masks(mutability, D; device=identity)

Precompute the per-direction boolean masks used by
[`batched_apply_mutability!`](@ref).

Returns `(none_mask, inc_mask, dec_mask)`, each a `D×1` boolean array on the
compute device, or `nothing` if `mutability` is `nothing`.
"""
function prepare_mutability_masks(
    mutability::Union{Nothing,Vector{Symbol}}, D::Int; device=identity
)
    isnothing(mutability) && return nothing
    none_mask = reshape([dir == :none for dir in mutability], D, 1) |> device
    inc_mask = reshape([dir == :increase for dir in mutability], D, 1) |> device
    dec_mask = reshape([dir == :decrease for dir in mutability], D, 1) |> device
    return (none_mask, inc_mask, dec_mask)
end

# ---------------------------------------------------------------------------
# Helper: apply mutability constraints to a batched gradient
# ---------------------------------------------------------------------------
"""
    batched_apply_mutability!(ΔX::AbstractMatrix, masks)

Apply precomputed mutability masks to a batched gradient update in-place.
`masks` is a 3-tuple `(none_mask, inc_mask, dec_mask)` as returned by
[`prepare_mutability_masks`](@ref), or `nothing` (no-op).

Zeros out gradient components along immutable feature directions in-place.
Uses vectorized broadcasting for GPU compatibility.
"""
function batched_apply_mutability!(ΔX::AbstractMatrix, masks::Nothing)
    return ΔX  # no-op when mutability is nothing
end

function batched_apply_mutability!(
    ΔX::AbstractMatrix, masks::Tuple{<:AbstractMatrix,<:AbstractMatrix,<:AbstractMatrix}
)
    T = eltype(ΔX)
    none_mask, inc_mask, dec_mask = masks
    ΔX .*= ifelse.(none_mask, zero(T), one(T))
    ΔX .*= ifelse.(inc_mask .& (ΔX .< zero(T)), zero(T), one(T))
    ΔX .*= ifelse.(dec_mask .& (ΔX .> zero(T)), zero(T), one(T))
    return ΔX
end

"""
    batched_apply_mutability!(ΔX::AbstractMatrix, mutability; device=identity)

Zeros out gradient components along immutable feature directions in-place.
`mutability` is a vector of `Symbol`s (`:both`, `:none`, `:increase`, `:decrease`),
one per feature row. Uses vectorized broadcasting for GPU compatibility.
The `device` keyword moves the mask arrays to the compute device before
broadcasting, preventing non-bitstype CPU array capture in GPU kernels.

This is a convenience wrapper that builds the masks on the fly and delegates
to [`batched_apply_mutability!(ΔX, masks)`](@ref). For repeated calls with
the same `mutability`, precompute masks with [`prepare_mutability_masks`](@ref)
and call the mask-accepting signature directly.
"""
function batched_apply_mutability!(
    ΔX::AbstractMatrix, mutability::Union{Nothing,Vector{Symbol}}; device=identity
)
    isnothing(mutability) && return ΔX
    D = size(ΔX, 1)
    masks = prepare_mutability_masks(mutability, D; device=device)
    return batched_apply_mutability!(ΔX, masks)
end

# ---------------------------------------------------------------------------
# Preparer: precompute domain bounds (hoisted out of the search loop)
# ---------------------------------------------------------------------------
"""
    prepare_domain_bounds(domain, D; device=identity)

Precompute the lower/upper bound arrays used by
[`batched_apply_domain_constraints!`](@ref).

Returns `(lb, ub)`, each a `D×1` array on the compute device, or `nothing` if
`domain` is `nothing`.
"""
function prepare_domain_bounds(domain, D::Int; device=identity)
    isnothing(domain) && return nothing
    lb = reshape([domain[i][1] for i in 1:D], D, 1) |> device
    ub = reshape([domain[i][2] for i in 1:D], D, 1) |> device
    return (lb, ub)
end

# ---------------------------------------------------------------------------
# Helper: clamp counterfactuals to domain bounds
# ---------------------------------------------------------------------------
"""
    batched_apply_domain_constraints!(X′::AbstractMatrix, bounds)

Clamp each feature row of `X′` to the precomputed domain bounds in-place.
`bounds` is a 2-tuple `(lb, ub)` as returned by [`prepare_domain_bounds`](@ref),
or `nothing` (no-op).

Uses vectorized broadcasting for GPU compatibility.
"""
function batched_apply_domain_constraints!(X′::AbstractMatrix, bounds::Nothing)
    return X′
end

function batched_apply_domain_constraints!(
    X′::AbstractMatrix, bounds::Tuple{<:AbstractMatrix,<:AbstractMatrix}
)
    lb, ub = bounds
    X′ .= clamp.(X′, lb, ub)
    return X′
end

"""
    batched_apply_domain_constraints!(X′::AbstractMatrix, data::CounterfactualData; device=identity)

Clamps each feature row of `X′` to the domain bounds stored in `data.domain`.
Uses vectorized broadcasting for GPU compatibility.
The `device` keyword moves the bounds arrays to the compute device before
broadcasting, preventing non-bitstype CPU array capture in GPU kernels.

This is a convenience wrapper that builds the bounds on the fly and delegates
to [`batched_apply_domain_constraints!(X′, bounds)`](@ref). For repeated calls
with the same `data`, precompute bounds with [`prepare_domain_bounds`](@ref)
and call the bounds-accepting signature directly.
"""
function batched_apply_domain_constraints!(
    X′::AbstractMatrix, data::CounterfactualData; device=identity
)
    domain = data.domain
    isnothing(domain) && return X′
    D = size(X′, 1)
    bounds = prepare_domain_bounds(domain, D; device=device)
    return batched_apply_domain_constraints!(X′, bounds)
end

# ---------------------------------------------------------------------------
# Helper: batched convergence check
# ---------------------------------------------------------------------------
"""
    check_batched_convergence(probs, target_idx, iter, maxiter, threshold)

Returns a `BitVector` of length `N` indicating which samples have converged.
Convergence is reached when the target-class probability ≥ `threshold` or
`iter ≥ maxiter`.
"""
function check_batched_convergence(
    probs::AbstractMatrix,
    target_idx::AbstractVector{Int},
    iter::Int,
    maxiter::Int,
    threshold::Float32,
)
    N = length(target_idx)
    if iter >= maxiter
        return trues(N)
    end
    C = size(probs, 1)
    linear_idx = (0:(N - 1)) .* C .+ target_idx
    target_probs = probs[linear_idx]
    return target_probs .>= threshold
end

# ---------------------------------------------------------------------------
# Helper: track last valid adversarial examples
# ---------------------------------------------------------------------------
"""
    track_adversarial_examples!(last_valid_ae, X, X′, epsilon, p, perturbations, norms)

In-place version that writes the perturbation matrix into `perturbations` and
the per-sample norms into `norms` (both preallocated, same shape/type as needed).
Updates `last_valid_ae` in-place: for samples whose perturbation norm (in
`p`-norm) is ≤ `epsilon`, the current counterfactual is stored.
"""
function track_adversarial_examples!(
    last_valid_ae::AbstractMatrix,
    X::AbstractMatrix,
    X′::AbstractMatrix,
    epsilon::Float32,
    p::Real,
    perturbations::AbstractMatrix,
    norms::AbstractVector,
)
    @assert size(perturbations) == size(X′)
    @assert length(norms) == size(X′, 2)
    perturbations .= X′ .- X
    if p == Inf
        norms .= vec(maximum(abs, perturbations; dims=1))
    else
        norms .= vec(sum(abs.(perturbations) .^ p; dims=1) .^ (1 / p))
    end
    valid_ae = norms .<= epsilon
    last_valid_ae[:, valid_ae] .= X′[:, valid_ae]
    return last_valid_ae
end

"""
    track_adversarial_examples!(last_valid_ae, X, X′, epsilon, p)

Convenience wrapper that allocates `perturbations` and `norms` on the fly.
For repeated calls (e.g. inside the counterfactual search loop), prefer the
6-argument signature with preallocated buffers.
"""
function track_adversarial_examples!(
    last_valid_ae::AbstractMatrix,
    X::AbstractMatrix,
    X′::AbstractMatrix,
    epsilon::Float32,
    p::Real,
)
    perturbations = similar(X′)
    norms = similar(X′, size(X′, 2))
    return track_adversarial_examples!(
        last_valid_ae, X, X′, epsilon, p, perturbations, norms
    )
end

# ---------------------------------------------------------------------------
# Batched generator loss
# ---------------------------------------------------------------------------
"""
    generator_loss(gen, model, X′, X, targets_onehot, target_idx, iter,
                   reg_strength, decay, maxiter)

Batched version of the ECCo objective.  Operates on `D×N` matrices.
"""
function generator_loss(
    gen::NativeGenerator,
    model,
    X′::AbstractMatrix,
    X::AbstractMatrix,
    targets_onehot::AbstractMatrix,
    target_idx::AbstractVector{Int},
    iter::Int,
    reg_strength::Float32,
    decay::Float32,
    maxiter::Int,
)
    logits = model(X′)
    return generator_loss_from_logits(
        gen, logits, X′, X, targets_onehot, target_idx, iter, reg_strength, decay, maxiter
    )
end

"""
    generator_loss_from_logits(gen, logits, X′, X, targets_onehot, target_idx, iter,
                               reg_strength, decay, maxiter)

Like [`generator_loss`](@ref) but accepts precomputed `logits = model(X′)`
instead of calling `model` internally. Used by [`generate_counterfactuals!`](@ref)
to share the forward pass between the gradient computation and the convergence
check.
"""
function generator_loss_from_logits(
    gen::NativeGenerator,
    logits::AbstractMatrix,
    X′::AbstractMatrix,
    X::AbstractMatrix,
    targets_onehot::AbstractMatrix,
    target_idx::AbstractVector{Int},
    iter::Int,
    reg_strength::Float32,
    decay::Float32,
    maxiter::Int,
)
    ℓ = Flux.logitcrossentropy(logits, targets_onehot; agg=sum)
    h1 = gen.λ[1] * sum(abs, X′ .- X)
    b = Float32(round(maxiter / 25))
    a = b / 10.0f0
    ϕ = polynomial_decay(a, b, decay, iter)
    e = batched_energy_from_logits(logits, target_idx)
    gen_loss = sum(e)
    reg_loss_val = sum(abs2, e)
    h2 = gen.λ[2] * ϕ * (gen_loss + reg_strength * reg_loss_val)
    return ℓ + h1 + h2
end

# ---------------------------------------------------------------------------
# Main entry point: batched generate_counterfactuals!
# ---------------------------------------------------------------------------
"""
    generate_counterfactuals!(
        model,
        X::AbstractMatrix,
        targets::Vector{Int},
        data::CounterfactualData,
        generator::NativeGenerator;
        maxiter = 30,
        decision_threshold = 0.75f0,
        decay = 0.9f0,
        reg_strength = 1.0f-3,
        epsilon = 0.3f0,
        p = Inf,
        device = identity,
        cf_batchsize = 128,
    )

Generates counterfactual explanations for a batch of `N` factuals in a
fully batched (GPU-compatible) fashion.

# Arguments
- `model`: A Flux model (or any callable that accepts a `D×N` matrix and
  returns a `C×N` matrix of logits).
- `X`: `D×N` matrix of factuals.
- `targets`: Length-`N` vector of target class indices (1-based integers).
- `data`: A `CounterfactualData` object providing domain bounds and mutability.
- `generator`: A [`NativeGenerator`](@ref).

# Keyword arguments
- `maxiter`: Maximum number of search iterations.
- `decision_threshold`: Target probability threshold for convergence.
- `decay`: Decay rate for the polynomial decay schedule.
- `reg_strength`: Regularization strength for the energy penalty.
- `epsilon`: Norm bound for tracking valid adversarial examples.
- `p`: Norm order for the adversarial example bound (default `Inf`).
- `device`: Function to move data to the compute device (`identity` for CPU,
  `Flux.gpu` for GPU). Factuals and one-hot targets are moved to the device
  before the search loop.
- `cf_batchsize`: Mini-batch size for the counterfactual search forward/backward
  passes. Controls peak GPU memory: the search processes `cf_batchsize` samples
  at a time through the model. Default `128`. Set to a larger value for GPUs
  with more memory, or smaller for memory-constrained GPUs. When
  `cf_batchsize >= nsamples`, no chunking occurs.

# Returns
- `counterfactuals::AbstractMatrix`: The final counterfactuals (`D×N`).
- `last_valid_ae::AbstractMatrix`: Last valid adversarial examples (`D×N`).
- `converged_mask::BitVector`: Per-sample convergence flag.
- `maxiter::Int`: The maximum number of iterations used.
"""
function generate_counterfactuals!(
    model,
    X::AbstractMatrix,
    targets::Vector{Int},
    data::CounterfactualData,
    generator::NativeGenerator;
    maxiter::Int=30,
    decision_threshold::Float32=0.75f0,
    decay::Float32=0.9f0,
    reg_strength::Float32=1.0f-3,
    epsilon::Float32=0.3f0,
    p::Real=Inf,
    device=identity,
    cf_batchsize::Int=128,
)
    # Move factuals to device so model calls and gradient computation run on GPU
    X = X |> device

    N = size(X, 2)

    # Target encoding (one-hot), matching the element type of X
    nclasses = size(model(X[:, 1:1]), 1)  # infer from single sample (avoids full-batch forward)
    targets_onehot = Float32.(Flux.onehotbatch(targets, 1:nclasses)) |> device

    # Initialise counterfactuals as copy of factuals
    X′ = copy(X)

    # Initialise last valid adversarial examples
    last_valid_ae = copy(X)

    # Optimiser state for the counterfactual search
    opt_state = Flux.setup(generator.opt, X′)

    # Convergence mask
    converged = falses(N)

    # Precompute mutability masks and domain bounds once (hoisted out of the loop)
    D = size(X, 1)
    mutability_masks = prepare_mutability_masks(data.mutability, D; device=device)
    domain_bounds = prepare_domain_bounds(data.domain, D; device=device)

    # Preallocate buffers reused across iterations to avoid per-iteration
    # allocations of D×N matrices. X′_old holds the pre-update state so the
    # update can be computed as (X′ .- X′_old) and mutability applied to the
    # update (not the gradient) — see comment below. `update` holds that
    # post-optimizer, pre-mutability update.
    X′_old = similar(X′)
    update = similar(X′)

    # Preallocate gradient buffer (reused across chunked forward/backward passes)
    ΔX = similar(X′)

    # Preallocate buffers for track_adversarial_examples! (reused each iteration)
    perturbations = similar(X′)
    norms = similar(X′, size(X′, 2))

    for iter in 1:maxiter
        # Compute gradient of the generator loss w.r.t. X′ AND extract logits
        # for the convergence check — sharing the forward pass. The convergence
        # check is on the pre-update X′ (= post-update from the previous
        # iteration), so we detect convergence one iteration "late". This saves
        # one forward pass per iteration (the separate convergence forward).
        for start in 1:cf_batchsize:N
            stop = min(start + cf_batchsize - 1, N)
            local logits_chunk
            y, back = Flux.pullback(X′[:, start:stop]) do xc
                logits_chunk = model(xc)
                return generator_loss_from_logits(
                    generator,
                    logits_chunk,
                    xc,
                    X[:, start:stop],
                    targets_onehot[:, start:stop],
                    @view(targets[start:stop]),
                    iter,
                    reg_strength,
                    decay,
                    maxiter,
                )
            end
            ΔX[:, start:stop] .= back(one(y))[1]

            # Convergence check using the SAME logits_chunk (no extra forward)
            if iter < maxiter
                probs_chunk = Flux.softmax(logits_chunk; dims=1)
                target_idx_chunk = targets[start:stop]
                C = size(probs_chunk, 1)
                n_chunk = length(target_idx_chunk)
                linear_idx = (0:(n_chunk - 1)) .* C .+ target_idx_chunk
                target_probs = probs_chunk[linear_idx] |> Flux.cpu
                converged[start:stop] .= target_probs .>= decision_threshold
            end
        end

        if iter >= maxiter
            converged = trues(N)
        end

        # Early exit BEFORE the update (saves the update step when all converged)
        if all(converged)
            break
        end

        # Zero gradients for already-converged samples so the optimizer does no
        # work on them and they don't drift. This is safe because converged
        # samples are already at their final state; subsequent iterations only
        # need to search for the remaining unconverged samples.
        if any(converged)
            ΔX[:, converged] .= zero(eltype(ΔX))
        end

        # Apply optimizer step on the full gradient, then zero immutable
        # directions in the resulting update. This matches CE.jl's ordering
        # (search.jl: generate_perturbations → apply_mutability), where the
        # optimizer sees the full gradient and mutability is applied to the
        # update (Δstate), not the gradient. For Descent this is equivalent
        # to zeroing the gradient first; for momentum optimizers it prevents
        # accumulated momentum from moving immutable features.
        # X′_old and update are preallocated buffers reused each iteration.
        copyto!(X′_old, X′)
        Flux.update!(opt_state, X′, ΔX)
        update .= X′ .- X′_old

        # Also zero the update for converged samples. For Descent this is
        # redundant (zero gradient → zero update), but for momentum optimizers
        # (Adam, Momentum) the optimizer may produce a non-zero update from
        # accumulated momentum even with a zero gradient. Zeroing the update
        # guarantees converged samples don't move regardless of optimizer.
        if any(converged)
            update[:, converged] .= zero(eltype(update))
        end

        batched_apply_mutability!(update, mutability_masks)
        X′ .= X′_old .+ update

        # Clamp to domain bounds
        batched_apply_domain_constraints!(X′, domain_bounds)

        # Track last valid adversarial examples
        track_adversarial_examples!(last_valid_ae, X, X′, epsilon, p, perturbations, norms)
    end

    return X′, last_valid_ae, converged, maxiter
end

# ---------------------------------------------------------------------------
# Helper: find neighbours in target class
# ---------------------------------------------------------------------------
"""
    find_neighbours(X, y, targets, y_levels; nneighbours=1, rng=Random.default_rng())

For each counterfactual with target `targets[i]`, samples a random training
point that has label `targets[i]`.  Returns a `D×N` matrix (one neighbour
per counterfactual column).

Uses a precomputed class→indices dictionary to avoid per-sample `findall`
calls.  The `nneighbours` keyword is accepted for API compatibility but
currently only one neighbour per sample is returned.
"""
function find_neighbours(
    X::AbstractMatrix,
    y::AbstractVector,
    targets::Vector{Int},
    y_levels::AbstractVector;
    nneighbours::Int=1,
    rng=Random.default_rng(),
)
    D = size(X, 1)
    N = length(targets)
    y_plain = Vector{Int}(y)  # convert from CategoricalVector if needed
    # Precompute candidate indices per class (one findall per class, not per sample)
    class_idx = Dict{Int,Vector{Int}}(c => findall(==(c), y_plain) for c in y_levels)
    # Draw one index per target (cheap CPU loop — just random draws)
    chosen = Vector{Int}(undef, N)
    n_X = size(X, 2)
    for i in 1:N
        candidates = get(class_idx, targets[i], Int[])
        chosen[i] = isempty(candidates) ? rand(rng, 1:n_X) : rand(rng, candidates)
    end
    # Single gather (one indexing operation instead of N column assignments)
    neighbours = X[:, chosen]
    return neighbours
end

# ---------------------------------------------------------------------------
# Helper: protect immutable features (batched)
# ---------------------------------------------------------------------------
"""
    protect_immutable!(neighbours, counterfactuals, masks)

Protects immutable features by setting neighbour values according to mutability
directions.  For each feature row:
- `:both` → keep neighbour value
- `:none` → use counterfactual value (no change)
- `:increase` → max(counterfactual, neighbour)
- `:decrease` → min(counterfactual, neighbour)

Accepts precomputed masks (a 3-tuple of `D×1` boolean arrays as returned by
[`prepare_mutability_masks`](@ref)), or `nothing` (no-op).  The mask-based
signature uses broadcast `ifelse.` with `D×1` masks against `D×N` arrays,
making it GPU-safe.

The old signature accepting a `Vector{Symbol}` is kept as a back-compat
wrapper that builds masks on the fly (CPU-only).
"""
function protect_immutable!(
    neighbours::AbstractMatrix, counterfactuals::AbstractMatrix, masks::Nothing
)
    return neighbours
end

function protect_immutable!(
    neighbours::AbstractMatrix,
    counterfactuals::AbstractMatrix,
    masks::Tuple{<:AbstractMatrix,<:AbstractMatrix,<:AbstractMatrix},
)
    none_mask, inc_mask, dec_mask = masks
    # :none → use counterfactual value
    neighbours .= ifelse.(none_mask, counterfactuals, neighbours)
    # :increase → max(counterfactual, neighbour)
    neighbours .= ifelse.(inc_mask, max.(counterfactuals, neighbours), neighbours)
    # :decrease → min(counterfactual, neighbour)
    neighbours .= ifelse.(dec_mask, min.(counterfactuals, neighbours), neighbours)
    return neighbours
end

# ---------------------------------------------------------------------------
# Helper: split observations into groups
# ---------------------------------------------------------------------------
"""
    split_obs(indices, n)

Split `indices` into `n` roughly equal groups.  Returns a vector of vectors.
"""
function split_obs(indices::AbstractRange, n::Int)
    return [
        collect(part) for
        part in Base.Iterators.partition(indices, max(1, length(indices) ÷ n))
    ]
end

# ---------------------------------------------------------------------------
# Top-level: generate_native!
# ---------------------------------------------------------------------------
"""
    generate_native!(
        model, train_set, generator::NativeGenerator;
        nsamples=nothing, nneighbours=1, domain=nothing, mutability=nothing,
        maxiter=30, decision_threshold=0.75f0, decay=0.9f0,
        reg_strength=1.0f-3, epsilon=0.3f0, p=Inf, verbose=1, device=identity,
        cf_batchsize=128,
        cached_X=nothing, cached_y_raw=nothing, cached_data=nothing,
    )

Top-level counterfactual generation for the native branch.  Generates
counterfactuals for a subset of the training data, finds neighbours, applies
mutability protection, and partitions results into a data loader aligned
with `train_set` batches.

The `device` keyword (a function: `identity` for CPU, `Flux.gpu` for GPU)
moves the subsampled factuals to the device for model calls and the
counterfactual search. Results stay on the compute device; the returned
data loader contains device arrays ready for use in the training loop.

Returns `(dl, percent_valid, nothing)` — same interface as the old `generate!()`.

# Keyword arguments
- `cf_batchsize`: Mini-batch size for the counterfactual search forward/backward
  passes. Controls peak GPU memory: the search processes `cf_batchsize` samples
  at a time through the model. Default `128`. Set to a larger value for GPUs
  with more memory, or smaller for memory-constrained GPUs. When
  `cf_batchsize >= nsamples`, no chunking occurs.

# Cached keyword arguments
- `cached_X`: Pre-unwrapped feature matrix (CPU). When provided (by
  `counterfactual_training`), avoids calling `unwrap(train_set)` every epoch.
  When `nothing` (default, e.g. standalone calls), `unwrap` is called on the fly.
- `cached_y_raw`: Pre-unwrapped label vector (CPU). Paired with `cached_X`.
- `cached_data`: Pre-built `CounterfactualData` object. When provided, the
  `domain` and `mutability` keyword arguments are ignored (they are already
  baked into `cached_data`). When `nothing` (default), `CounterfactualData` is
  constructed from `X`, `y_raw`, `domain`, and `mutability`.
"""
function generate_native!(
    model,
    train_set,
    generator::NativeGenerator;
    nsamples::Union{Nothing,Int}=nothing,
    nneighbours::Int=1,
    domain=nothing,
    mutability=nothing,
    maxiter::Int=30,
    decision_threshold::Float32=0.75f0,
    decay::Float32=0.9f0,
    reg_strength::Float32=1.0f-3,
    epsilon::Float32=0.3f0,
    p::Real=Inf,
    verbose::Int=1,
    device=identity,
    cf_batchsize::Int=128,
    # Cached across epochs (built once by counterfactual_training):
    cached_X::Union{Nothing,AbstractMatrix}=nothing,
    cached_y_raw::Union{Nothing,AbstractVector}=nothing,
    cached_data::Union{Nothing,CounterfactualData}=nothing,
)
    # Use cached unwrap/CounterfactualData when provided (avoids rebuilding
    # every epoch); otherwise build on the fly for standalone use.
    if isnothing(cached_X) || isnothing(cached_y_raw)
        X, y_raw = unwrap(train_set)
        X = Flux.cpu(X)
    else
        X, y_raw = cached_X, cached_y_raw
    end
    data = if isnothing(cached_data)
        CounterfactualData(X, y_raw; domain=domain, mutability=mutability)
    else
        cached_data
    end
    D = size(X, 1)

    # Determine sample size
    N = size(X, 2)
    nsamples = isnothing(nsamples) ? N : min(nsamples, N)
    if nsamples < length(train_set)
        @warn "Need at least one counterfactual per batch. Setting nsamples=$(length(train_set))." maxlog =
            1
        nsamples = length(train_set)
    end

    # Subsample factuals
    if nsamples < N
        idx_sub = StatsBase.sample(1:N, nsamples; replace=false)
        X_sub = X[:, idx_sub]
    else
        idx_sub = collect(1:N)
        X_sub = X
    end

    # Move subsampled factuals to device for model calls
    X_sub_dev = X_sub |> device

    # Predict factual labels (chunked to bound GPU memory)
    factual_preds = Int[]
    for start in 1:cf_batchsize:nsamples
        stop = min(start + cf_batchsize - 1, nsamples)
        logits_chunk = model(X_sub_dev[:, start:stop]) |> Flux.cpu
        append!(factual_preds, vec(Flux.onecold(Flux.softmax(logits_chunk))))
    end
    y_levels = data.y_levels
    targets = Vector{Int}(undef, nsamples)
    for i in 1:nsamples
        available = setdiff(y_levels, factual_preds[i])
        targets[i] = rand(available)
    end

    # Generate counterfactuals (batched, on device)
    counterfactuals, last_valid_ae, converged_mask, maxiter = generate_counterfactuals!(
        model,
        X_sub_dev,
        targets,
        data,
        generator;
        maxiter=maxiter,
        decision_threshold=decision_threshold,
        decay=decay,
        reg_strength=reg_strength,
        epsilon=epsilon,
        p=p,
        device=device,
        cf_batchsize=cf_batchsize,
    )

    # Find neighbours in target class (on CPU — uses findall on y_raw)
    neighbours = find_neighbours(X, y_raw, targets, y_levels; nneighbours=nneighbours)

    # Move neighbours to device for protect_immutable! (counterfactuals already on device)
    neighbours = neighbours |> device

    # Precompute mutability masks on device (reused by protect_immutable!)
    mutability_masks = prepare_mutability_masks(data.mutability, D; device=device)

    # Protect immutable features (GPU-safe broadcast with D×1 masks)
    protect_immutable!(neighbours, counterfactuals, mutability_masks)

    # One-hot encodings (batched, on device)
    nclasses = length(y_levels)
    targets_enc = Flux.onehotbatch(targets, 1:nclasses) |> device
    factual_enc = Flux.onehotbatch(y_raw[idx_sub], 1:nclasses) |> device

    # Validity
    percent_valid = sum(converged_mask) / nsamples

    # Partition into batch-aligned data loader (all arrays on device)
    group_indices = split_obs(1:nsamples, length(train_set))
    dl = [
        (
            counterfactuals[:, group_indices[i]],
            last_valid_ae[:, group_indices[i]],
            targets_enc[:, group_indices[i]],
            neighbours[:, group_indices[i]],
            factual_enc[:, group_indices[i]],
        ) for i in eachindex(group_indices)
    ]

    return dl, percent_valid, nothing
end

# ---------------------------------------------------------------------------
# Main entry point: overloaded counterfactual_training
# ---------------------------------------------------------------------------
"""
    counterfactual_training(
        loss::AbstractObjective,
        model,
        generator::NativeGenerator,
        train_set,
        opt_state;
        device = identity,
        val_set = nothing,
        nepochs = 100,
        burnin = 0.0f0,
        nce = nothing,
        nneighbours = 100,
        domain = nothing,
        mutability = nothing,
        maxiter = 30,
        decision_threshold = 0.75f0,
        decay = 0.9f0,
        reg_strength = 1.0f-3,
        epsilon = 0.3f0,
        p = Inf,
        verbose = 1,
        checkpoint_dir = nothing,
        callback = nothing,
        cf_batchsize = 128,
        accuracy_every::Real = Inf,
    )

Native GPU-compatible counterfactual training.  Dispatches here when the
generator is a [`NativeGenerator`](@ref).

The `device` keyword is a function: `identity` (CPU), `Flux.gpu` (CUDA),
or `AMDGPU.gpu` (AMDGPU).  The model is moved to the device; training data
should already be on the device (user moves it before constructing the
DataLoader).

# Keyword arguments
- `cf_batchsize`: Mini-batch size for the counterfactual search forward/backward
  passes. Controls peak GPU memory: the search processes `cf_batchsize` samples
  at a time through the model. Default `128`. Set to a larger value for GPUs
  with more memory, or smaller for memory-constrained GPUs. When
  `cf_batchsize >= nsamples`, no chunking occurs.
- `accuracy_every`: Compute training (and validation) accuracy only every
  `accuracy_every` epochs. When `epoch % accuracy_every != 0`, the logged
  `acc` and `acc_val` fields are `nothing`. Default `Inf` (accuracy is never
  computed unless explicitly requested). Set to `1` for every epoch, or a
  larger value (e.g. `10`) to reduce per-epoch wall-clock time for large
  models and datasets.
"""
function counterfactual_training(
    loss::AbstractObjective,
    model,
    generator::NativeGenerator,
    train_set,
    opt_state;
    device=identity,
    val_set=nothing,
    nepochs::Int=100,
    burnin=0.0f0,
    nce::Union{Nothing,Int}=nothing,
    nneighbours::Int=100,
    domain=nothing,
    mutability=nothing,
    maxiter::Int=30,
    decision_threshold::Float32=0.75f0,
    decay::Float32=0.9f0,
    reg_strength::Float32=1.0f-3,
    epsilon::Float32=0.3f0,
    p::Real=Inf,
    verbose::Int=1,
    checkpoint_dir::Union{Nothing,String}=nothing,
    callback::Union{Nothing,Function}=nothing,
    cf_batchsize::Int=128,
    accuracy_every::Real=Inf,
    kwrgs...,
)
    # Move model and optimizer state to device
    model = model |> device
    opt_state = opt_state |> device

    # Setup
    burnin = Int(round(burnin * nepochs))
    nce = isnothing(nce) ? length(train_set) : nce

    log = []
    start_epoch = 1

    # Checkpoint loading
    if !isnothing(checkpoint_dir) && isfile(joinpath(checkpoint_dir, "checkpoint.jld2"))
        @info "Found checkpoint file in $checkpoint_dir. Loading..."
        try
            _model, _opt_state, epoch, _log = JLD2.load(
                joinpath(checkpoint_dir, "checkpoint.jld2"),
                "model",
                "opt_state",
                "epoch",
                "log",
            )
            model = _model |> device
            opt_state = _opt_state |> device
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
        prog = Progress(nepochs - start_epoch; barglyphs=BarGlyphs("[=> ]"), color=:yellow)
    end

    # Cache unwrap + CounterfactualData across epochs (built once, reused
    # every epoch). Only needed when counterfactuals are generated.
    cached_X = nothing
    cached_y_raw = nothing
    cached_data = nothing
    if needs_counterfactuals(loss)
        cached_X, cached_y_raw = unwrap(train_set)
        cached_X = Flux.cpu(cached_X)
        cached_data = CounterfactualData(
            cached_X, cached_y_raw; domain=domain, mutability=mutability
        )
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
                model,
                train_set,
                generator;
                nsamples=nce,
                nneighbours=nneighbours,
                domain=domain,
                mutability=mutability,
                maxiter=maxiter,
                decision_threshold=decision_threshold,
                decay=decay,
                reg_strength=reg_strength,
                epsilon=epsilon,
                p=p,
                verbose=verbose,
                device=device,
                cf_batchsize=cf_batchsize,
                cached_X=cached_X,
                cached_y_raw=cached_y_raw,
                cached_data=cached_data,
            )
            avg_iter = maxiter
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
                    implaus, regs = implausibility_and_reg_loss(
                        m, perturbed_input, neighbours, targets_enc
                    )
                    adversarial_loss = loss.class_loss(m(advexms), factual_enc)
                else
                    implaus = [0.0f0]
                    regs = [0.0f0]
                    adversarial_loss = 0.0f0
                end

                ChainRulesCore.ignore_derivatives() do
                    push!(implausibilities, sum(implaus) / length(implaus))
                    push!(reg_losses, sum(regs) / length(regs))
                    return push!(validity_losses, adversarial_loss)
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
        if epoch % accuracy_every == 0
            acc = accuracy(model, train_set; device=device)
            acc_val = isnothing(val_set) ? nothing : accuracy(model, val_set; device=device)
        else
            acc = nothing
            acc_val = nothing
        end
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

        push!(
            log,
            (;
                acc,
                acc_val,
                train_loss,
                implaus,
                log_reg_loss,
                log_adv_loss,
                time_taken,
                percent_valid,
                avg_iter,
            ),
        )

        # Checkpointing (save CPU version)
        if !isnothing(checkpoint_dir)
            jldsave(
                joinpath(checkpoint_dir, "checkpoint.jld2");
                model=model |> Flux.cpu,
                opt_state=opt_state |> Flux.cpu,
                epoch,
                log,
            )
        end

        # Progress bar
        if verbose in [1, 2]
            next!(prog)
        end

        # Logging
        if verbose >= 2
            @info "Iteration $epoch:"
            @info "Training accuracy: $acc"
            @info "Train loss: $train_loss"
            if !isnothing(implaus)
                @info "Implausibility: $implaus"
                @info "Reg loss: $log_reg_loss"
                @info "Adv loss: $log_adv_loss"
            end
            @info "Valid CFs: $(percent_valid * 100)%"
        end
    end

    return model, log
end
