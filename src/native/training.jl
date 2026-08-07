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

# ---------------------------------------------------------------------------
# Helper: find neighbours in target class
# ---------------------------------------------------------------------------
"""
    find_neighbours(X, y, targets, y_levels; rng=Random.default_rng())

For each counterfactual with target `targets[i]`, samples a random training
point that has label `targets[i]`.  Returns a `D×N` matrix (one neighbour
per counterfactual column).
"""
function find_neighbours(
    X::AbstractMatrix,
    y::AbstractVector,
    targets::Vector{Int},
    y_levels::AbstractVector;
    nneighbours::Int = 1,
    rng = Random.default_rng(),
)
    D = size(X, 1)
    N = length(targets)
    neighbours = similar(X, D, N)
    y_plain = Vector{Int}(y)  # convert from CategoricalVector if needed
    for i in 1:N
        target = targets[i]
        candidates = findall(==(target), y_plain)
        if isempty(candidates)
            chosen = rand(rng, 1:size(X, 2))
        else
            chosen = rand(rng, candidates)
        end
        neighbours[:, i] = X[:, chosen]
    end
    return neighbours
end

# ---------------------------------------------------------------------------
# Helper: protect immutable features (batched)
# ---------------------------------------------------------------------------
"""
    protect_immutable!(neighbours, counterfactuals, mutability)

Protects immutable features by setting neighbour values according to mutability
directions.  For each feature row:
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

# ---------------------------------------------------------------------------
# Helper: split observations into groups
# ---------------------------------------------------------------------------
"""
    split_obs(indices, n)

Split `indices` into `n` roughly equal groups.  Returns a vector of vectors.
"""
function split_obs(indices::AbstractRange, n::Int)
    return [collect(part) for part in Base.Iterators.partition(indices, max(1, length(indices) ÷ n))]
end

# ---------------------------------------------------------------------------
# Top-level: generate_native!
# ---------------------------------------------------------------------------
"""
    generate_native!(
        model, train_set, generator::NativeGenerator;
        nsamples=nothing, nneighbours=1, domain=nothing, mutability=nothing,
        maxiter=30, decision_threshold=0.75f0, decay=0.9f0,
        reg_strength=1.0f-3, epsilon=0.3f0, p=Inf, verbose=1,
    )

Top-level counterfactual generation for the native branch.  Generates
counterfactuals for a subset of the training data, finds neighbours, applies
mutability protection, and partitions results into a data loader aligned
with `train_set` batches.

Returns `(dl, percent_valid, nothing)` — same interface as the old `generate!()`.
"""
function generate_native!(
    model,
    train_set,
    generator::NativeGenerator;
    nsamples::Union{Nothing,Int} = nothing,
    nneighbours::Int = 1,
    domain = nothing,
    mutability = nothing,
    maxiter::Int = 30,
    decision_threshold::Float32 = 0.75f0,
    decay::Float32 = 0.9f0,
    reg_strength::Float32 = 1.0f-3,
    epsilon::Float32 = 0.3f0,
    p::Real = Inf,
    verbose::Int = 1,
)
    # Unwrap training data
    X, y_raw = unwrap(train_set)

    # Build CounterfactualData
    data = CounterfactualData(X, y_raw; domain = domain, mutability = mutability)

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
    y_levels = data.y_levels
    targets = Vector{Int}(undef, nsamples)
    for i in 1:nsamples
        available = setdiff(y_levels, factual_preds[i])
        targets[i] = rand(available)
    end

    # Generate counterfactuals (batched)
    counterfactuals, last_valid_ae, converged_mask, maxiter = generate_counterfactuals!(
        model, X_sub, targets, data, generator;
        maxiter = maxiter, decision_threshold = decision_threshold,
        decay = decay, reg_strength = reg_strength,
        epsilon = epsilon, p = p,
    )

    # Find neighbours in target class
    neighbours = find_neighbours(X, y_raw, targets, y_levels; nneighbours = nneighbours)

    # Protect immutable features
    protect_immutable!(neighbours, counterfactuals, data.mutability)

    # One-hot encodings
    nclasses = length(y_levels)
    targets_enc = [Flux.onehot(t, 1:nclasses) for t in targets]
    factual_enc = [Flux.onehot(y_raw[idx_sub[i]], 1:nclasses) for i in 1:nsamples]

    # Validity
    percent_valid = sum(converged_mask) / nsamples

    # Partition into batch-aligned data loader
    group_indices = split_obs(1:nsamples, length(train_set))
    dl = [
        (
            stack(counterfactuals[:, group_indices[i]]),
            stack(last_valid_ae[:, group_indices[i]]),
            stack(targets_enc[group_indices[i]]),
            stack(neighbours[:, group_indices[i]]),
            stack(factual_enc[group_indices[i]]),
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
    )

Native GPU-compatible counterfactual training.  Dispatches here when the
generator is a [`NativeGenerator`](@ref).

The `device` keyword is a function: `identity` (CPU), `Flux.gpu` (CUDA),
or `AMDGPU.gpu` (AMDGPU).  The model is moved to the device; training data
should already be on the device (user moves it before constructing the
DataLoader).
"""
function counterfactual_training(
    loss::AbstractObjective,
    model,
    generator::NativeGenerator,
    train_set,
    opt_state;
    device = identity,
    val_set = nothing,
    nepochs::Int = 100,
    burnin = 0.0f0,
    nce::Union{Nothing,Int} = nothing,
    nneighbours::Int = 100,
    domain = nothing,
    mutability = nothing,
    maxiter::Int = 30,
    decision_threshold::Float32 = 0.75f0,
    decay::Float32 = 0.9f0,
    reg_strength::Float32 = 1.0f-3,
    epsilon::Float32 = 0.3f0,
    p::Real = Inf,
    verbose::Int = 1,
    checkpoint_dir::Union{Nothing,String} = nothing,
    callback::Union{Nothing,Function} = nothing,
    kwrgs...,
)
    # Move model to device
    model = model |> device

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
                domain = domain,
                mutability = mutability,
                maxiter = maxiter,
                decision_threshold = decision_threshold,
                decay = decay,
                reg_strength = reg_strength,
                epsilon = epsilon,
                p = p,
                verbose = verbose,
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
        acc = accuracy(model, train_set)
        acc_val = isnothing(val_set) ? nothing : accuracy(model, val_set)
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
        elseif verbose > 2
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
