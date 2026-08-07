# Batched counterfactual generation for the Native submodule.
#
# This file implements a GPU-compatible, batched counterfactual search loop
# that operates on D×N matrices without allocating a CounterfactualExplanation
# per sample.  It uses CE.jl's `CounterfactualData` for domain/mutability
# constraints and CE.jl's `polynomial_decay` utility, but defines its own
# lightweight generator struct to avoid the Flux 0.16 incompatibility in CE.jl's
# `GradientBasedGenerator` (whose `opt` field is typed against the old
# `Flux.Optimise.AbstractOptimiser`).

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
# Helper: batched energy (negative target logits for all N samples)
# ---------------------------------------------------------------------------
"""
    batched_energy(model, X′::AbstractMatrix, target_idx::AbstractVector)

Returns a length-`N` vector of negative logits at the target class for each
sample in `X′`.  Uses `CartesianIndex` for GPU compatibility.
"""
function batched_energy(model, X′::AbstractMatrix, target_idx::AbstractVector{Int})
    logits = model(X′)          # C × N
    N = size(X′, 2)
    idx = CartesianIndex.(target_idx, 1:N)
    return -logits[idx]
end

# ---------------------------------------------------------------------------
# Helper: apply mutability constraints to a batched gradient
# ---------------------------------------------------------------------------
"""
    batched_apply_mutability!(ΔX::AbstractMatrix, mutability)

Zeros out gradient components along immutable feature directions in-place.
`mutability` is a vector of `Symbol`s (`:both`, `:none`, `:increase`, `:decrease`),
one per feature row.
"""
function batched_apply_mutability!(ΔX::AbstractMatrix, mutability::Union{Nothing,Vector{Symbol}})
    isnothing(mutability) && return ΔX
    T = eltype(ΔX)
    for (i, dir) in enumerate(mutability)
        if dir == :none
            @view(ΔX[i, :]) .= zero(T)
        elseif dir == :increase
            @view(ΔX[i, :]) .= ifelse.(@view(ΔX[i, :]) .< zero(T), zero(T), @view(ΔX[i, :]))
        elseif dir == :decrease
            @view(ΔX[i, :]) .= ifelse.(@view(ΔX[i, :]) .> zero(T), zero(T), @view(ΔX[i, :]))
        end
        # :both → no modification
    end
    return ΔX
end

# ---------------------------------------------------------------------------
# Helper: clamp counterfactuals to domain bounds
# ---------------------------------------------------------------------------
"""
    batched_apply_domain_constraints!(X′::AbstractMatrix, data::CounterfactualData)

Clamps each feature row of `X′` to the domain bounds stored in `data.domain`.
"""
function batched_apply_domain_constraints!(X′::AbstractMatrix, data::CounterfactualData)
    domain = data.domain
    isnothing(domain) && return X′
    for i in axes(X′, 1)
        lb, ub = domain[i]
        @view(X′[i, :]) .= clamp.(@view(X′[i, :]), lb, ub)
    end
    return X′
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
    idx = CartesianIndex.(target_idx, 1:N)
    target_probs = probs[idx]
    return target_probs .>= threshold
end

# ---------------------------------------------------------------------------
# Helper: track last valid adversarial examples
# ---------------------------------------------------------------------------
"""
    track_adversarial_examples!(last_valid_ae, X, X′, epsilon, p)

Updates `last_valid_ae` in-place: for samples whose perturbation norm (in
`p`-norm) is ≤ `epsilon`, the current counterfactual is stored.
"""
function track_adversarial_examples!(
    last_valid_ae::AbstractMatrix,
    X::AbstractMatrix,
    X′::AbstractMatrix,
    epsilon::Float32,
    p::Real,
)
    perturbations = X′ .- X
    if p == Inf
        norms = vec(maximum(abs, perturbations; dims = 1))
    else
        norms = vec(sum(abs .^ p, perturbations; dims = 1) .^ (1 / p))
    end
    valid_ae = norms .<= epsilon
    last_valid_ae[:, valid_ae] .= X′[:, valid_ae]
    return last_valid_ae
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
    # Classification loss
    ℓ = Flux.logitcrossentropy(model(X′), targets_onehot)

    # L1 distance penalty
    h1 = gen.λ[1] * sum(abs, X′ .- X)

    # Energy constraint with polynomial decay
    ϕ = polynomial_decay(Float32(maxiter) / 250.0f0, Float32(maxiter) / 25.0f0, decay, iter + 1)
    e = batched_energy(model, X′, target_idx)  # N-vector
    gen_loss = sum(e) / length(e)
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
    maxiter::Int = 30,
    decision_threshold::Float32 = 0.75f0,
    decay::Float32 = 0.9f0,
    reg_strength::Float32 = 1.0f-3,
    epsilon::Float32 = 0.3f0,
    p::Real = Inf,
)
    N = size(X, 2)

    # Target encoding (one-hot), matching the element type of X
    nclasses = size(model(X), 1)  # infer number of classes from model output
    targets_onehot = Float32.(Flux.onehotbatch(targets, 1:nclasses))

    # Initialise counterfactuals as copy of factuals
    X′ = copy(X)

    # Initialise last valid adversarial examples
    last_valid_ae = copy(X)

    # Optimiser state for the counterfactual search
    opt_state = Flux.setup(generator.opt, X′)

    # Convergence mask
    converged = falses(N)

    for iter in 1:maxiter
        # Compute gradient of the generator loss w.r.t. X′
        grads_val = Flux.withgradient(X′) do x
            generator_loss(generator, model, x, X, targets_onehot, targets, iter,
                           reg_strength, decay, maxiter)
        end
        ΔX = grads_val.grad[1]

        # Apply mutability constraints
        batched_apply_mutability!(ΔX, data.mutability)

        # Update counterfactuals
        Flux.update!(opt_state, X′, ΔX)

        # Clamp to domain bounds
        batched_apply_domain_constraints!(X′, data)

        # Track last valid adversarial examples
        track_adversarial_examples!(last_valid_ae, X, X′, epsilon, p)

        # Check convergence
        probs = Flux.softmax(model(X′); dims = 1)
        converged = check_batched_convergence(probs, targets, iter, maxiter, decision_threshold)

        # Early exit if all samples have converged
        if all(converged)
            break
        end
    end

    return X′, last_valid_ae, converged, maxiter
end
