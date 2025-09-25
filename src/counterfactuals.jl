using Base.Iterators
using CounterfactualExplanations
using CounterfactualExplanations: Convergence
using CounterfactualExplanations: Evaluation
using CounterfactualExplanations: Models
using CounterfactualExplanations: find_potential_neighbours
using Flux
using StatsBase
using TaijaParallel

"""
    generate!(
        model,
        data,
        generator::AbstractGenerator;
        nsamples::Union{Int,Nothing}=nothing,
        convergence=Convergence.MaxIterConvergence(),
        parallelizer=nothing,
        input_encoder=nothing,
        verbose=1,
        domain=nothing,
    )

This function generates counterfactual explanations for the whole dataset `data` or a subset thereof (`nsamples`). It is supposed to be used outside of the mini-batch training loop.
"""
function generate!(
    model,
    data,
    generator::AbstractGenerator;
    nsamples::Union{Int,Nothing}=nothing,
    nneighbours::Int=100,
    convergence=Convergence.MaxIterConvergence(),
    parallelizer=nothing,
    input_encoder=nothing,
    verbose=1,
    domain=nothing,
    mutability=nothing,
    callback::Function=get_last_valid_ae,
)
    xs, factual_enc, targets, counterfactual_data, M = setup_counterfactual_search(
        data, model, domain, input_encoder, mutability, nneighbours, nsamples
    )

    # ----- PAPER REF ----- #
    # Below counterfactual are generated in parallel. 
    # The generate_counterfactual() implements the FOR loop in Algorithm 1.

    # Generate counterfactuals:
    ces = TaijaParallel.parallelize(
        parallelizer,
        CounterfactualExplanations.generate_counterfactual,
        xs,
        targets,
        counterfactual_data,
        M,
        generator;
        convergence=convergence,
        initialization=:identity,
        return_flattened=true,
        verbose=verbose > 1,
        callback=callback,
    )

    # ----- PAPER REF ----- #
    # `advexms` correspond to the nascent counterfactuals.
    # `neighbours` correspond to the neighbours in the target class (one per ce).
    # `counterfactuals` correspond to the mature counterfactuals.

    advexms = (ce -> eltype(xs[1]).(ce.search[:last_valid_ae])).(ces)                               # get adversarial example
    targets = (ce -> ce.target).(ces)                                                               # get targets
    neighbours =
        (ce -> eltype(xs[1]).(find_potential_neighbours(ce, counterfactual_data, 1))).(ces)   # randomly draw a sample from the target class
    validities = (ce -> ce.search[:converged]).(ces)
    # Extract counterfactual if converged, else use neighbour (no penalty):
    counterfactuals = [
        validities[i] ? eltype(xs[1]).(ce.counterfactual) : neighbours[i] for
        (i, ce) in enumerate(ces)
    ]

    # ----- PAPER REF ----- #
    # This is where immutable features are protected by setting feature values of 
    # counterfactuals equal to those of their corresponding neighbours (i.e. imposing
    # a point mass prior).

    protect_immutable!(neighbours, counterfactuals, counterfactual_data.mutability)     # adjust for mutability
    targets_enc = (ce -> target_encoded(ce, counterfactual_data)).(ces)                 # one-hot encoded targets

    n_total = length(counterfactuals)
    percent_valid = sum(reduce(vcat, validities)) / n_total
    group_indices = TaijaParallel.split_obs(1:n_total, length(data))

    # Partition data:
    dl = [
        (
            stack(hcat(counterfactuals[i]...)),
            stack(hcat(advexms[i]...)),
            stack(hcat(targets_enc[i]...)),
            stack(hcat(neighbours[i]...)),
            stack(hcat(eachcol(factual_enc)[i]...)),
        ) for i in group_indices
    ]
    @assert length(dl) == length(data)

    return dl, percent_valid, ces
end

"""
    get_last_valid_ae(ce::CounterfactualExplanation)

A callback function used to store the last counterfactual that is also a valid adversarial example based on the global AE criterium (see [`get_global_ae_criterium`](@ref)).
"""
function get_last_valid_ae(ce::CounterfactualExplanation)
    # Find last counterfactual that meets imperceptability criterium:
    xs = [CounterfactualExplanations.decode_state(ce, x) for x in ce.search[:path]]
    perturbations = [x - ce.factual for x in xs]
    aecrit = get_global_ae_criterium()
    idx_advexm = [aecrit(x) for x in perturbations]
    if length(xs[idx_advexm]) > 0
        last_valid_ae = xs[idx_advexm][end]
    else
        last_valid_ae = ce.factual
    end
    return ce.search[:last_valid_ae] = last_valid_ae
end

"""
    isvalid(ce, model, data)

Checks if the label has been flipped.
"""
function isvalid(ce::AbstractCounterfactualExplanation, model, data)
    return argmax(vec(model(ce.counterfactual))) == argmax(vec(target_encoded(ce, data)))
end

"""
    protect_immutable!(
        samples::AbstractArray,
        counterfactuals::AbstractArray,
        mutability::Union{Nothing,AbstractArray},
    )

Protects immutable features from the contrastive divergence penalty.
"""
function protect_immutable!(
    samples::AbstractArray,
    counterfactuals::AbstractArray,
    mutability::Union{Nothing,AbstractArray},
)
    if !isnothing(mutability)
        direction_handlers = Dict(
            :both => (cf, s) -> s,
            :none => (cf, s) -> cf,
            :increase => (cf, s) -> cf > s ? cf : s,
            :decrease => (cf, s) -> cf < s ? cf : s,
        )

        for (i, ce) in enumerate(counterfactuals)
            for (j, allowed_direction) in enumerate(mutability)
                samples[i][j, :] = direction_handlers[allowed_direction](
                    counterfactuals[i][j, :], samples[i][j, :]
                )
            end
        end
    end
    return samples
end

"""
    setup_counterfactual_search(
        data,
        model,
        domain,
        input_encoder,
        mutability,
        nneighbours::Int64,
        nsamples::Union{Nothing,Int64},
    )

Sets up the counterfactual search.
"""
function setup_counterfactual_search(
    data,
    model,
    domain,
    input_encoder,
    mutability,
    nneighbours::Int64,
    nsamples::Union{Nothing,Int64},
)

    # Wrap training dataset in `CounterfactualData`:
    # NOTE: Using [1,...,n] for labels where n is the number of output classes. Exact label information is not necessary for training.
    X, y = unwrap(data)
    if isnothing(domain)
        domain = infer_domain_constraints(X)
    end
    counterfactual_data = CounterfactualData(
        X, y; domain=domain, input_encoder=input_encoder, mutability=mutability
    )
    # Wrap model:
    M = Models.Model(model, Models.FluxNN(); likelihood=:classification_multi)

    @assert nneighbours >= length(counterfactual_data.y_levels) "Number of neighbours must be greater than or equal to the number of output classes."

    # Set up counterfactual search:
    if isnothing(nsamples)
        nsamples = size(X, 2)
        # Use whole dataset:
        xs = [x[:, :] for x in eachcol(X)]                          # factuals
    else

        # Check that at least one counterfactual is generated for each batch:
        if nsamples < length(data)
            @warn "Need at least one counterfactual per batch. Setting `nsamples=$(nsamples)` to the total number of batches ($(length(data)))." maxlog =
                1
            nsamples = length(data)
        end

        # Use subset:
        idx_sub = sample(1:size(X, 2), nsamples; replace=false)
        Xsub = X[:, idx_sub]
        xs = [x[:, :] for x in eachcol(Xsub)]                       # factuals
    end

    # Factual label and targets:
    factual_labels = Vector{Int}(undef, nsamples)
    targets = Vector{Int}(undef, nsamples)
    all_labels = counterfactual_data.y_levels
    for (i, x) in enumerate(xs)
        factual_labels[i] = argmax(M.model(x))[1]                           # get factual label
        # targets[i] = rand(all_labels)                                       # choose a random target (including possibly the factual label)
        targets[i] = rand(all_labels[all_labels .!= factual_labels[i]])     # choose random target (excluding factual label)
    end
    factual_enc = Flux.onehotbatch(factual_labels, all_labels)

    return xs, factual_enc, targets, counterfactual_data, M
end
