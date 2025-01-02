using Base.Iterators
using CounterfactualExplanations
using CounterfactualExplanations: Convergence
using CounterfactualExplanations: Models
using CounterfactualExplanations: find_potential_neighbours
using Flux
using StatsBase
using TaijaParallel

"""
    generate!(
        model,
        data,
        generator;
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
    generator;
    nsamples::Union{Int,Nothing}=nothing,
    nneighbours::Int=100,
    convergence=Convergence.MaxIterConvergence(),
    parallelizer=nothing,
    input_encoder=nothing,
    verbose=1,
    domain=nothing,
)

    # Wrap training dataset in `CounterfactualData`:
    # NOTE: Using [1,...,n] for labels where n is the number of output classes. Exact label information is not necessary for training.
    X, y = unwrap(data)
    counterfactual_data = CounterfactualData(
        X, y; domain=domain, input_encoder=input_encoder
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
        Xsub = X[:, sample(1:size(X, 2), nsamples)]
        xs = [x[:, :] for x in eachcol(Xsub)]                       # factuals
    end

    # Factual label and targets:
    factual_labels = Vector{Int}(undef, nsamples)
    targets = Vector{Int}(undef, nsamples)
    all_labels = counterfactual_data.y_levels
    for (i, x) in enumerate(xs)
        factual_labels[i] = predict_label(M, counterfactual_data, x)[1]     # get factual label
        targets[i] = rand(all_labels[all_labels .!= factual_labels[i]])     # choose a random target that is not the factual label
    end
    factual_enc = Flux.onehotbatch(factual_labels, all_labels)

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
        return_flattened=true,
        verbose=verbose >= 1,
    )

    # Get neighbours in target class: find `nneighbours` potential neighbours than randomly choose one for each counterfactual.
    neighbours =
        (
            ce -> find_potential_neighbours(ce, counterfactual_data, nneighbours)[
                :, rand(1:end)
            ]
        ).(ces)
    counterfactuals = (ce -> ce.counterfactual).(ces)                               # get actual counterfactuals
    targets_enc = (ce -> target_encoded(ce, counterfactual_data)).(ces)             # encode targets as probabilities
    aversarial_targets = []
    targets_enc = []
    percent_valid = 0.0
    for (i, ce) in enumerate(ces)
        target_enc = target_encoded(ce, counterfactual_data)
        push!(targets_enc, target_enc)
        if argmax(vec(model(ce.counterfactual))) == argmax(vec(target_enc))
            # If model predicts target class, use target for adversarial loss:
            push!(aversarial_targets, target_enc)
            percent_valid += 1.0
        else
            # Otherwise, use factual label for adversarial loss:
            push!(aversarial_targets, factual_enc[:,i])
        end
    end

    # Partition data:
    n_total = length(counterfactuals)
    percent_valid = percent_valid / n_total
    group_indices = TaijaParallel.split_obs(1:n_total, length(data))

    dl = [
        (
            stack(hcat(counterfactuals[i]...)),
            stack(hcat(targets_enc[i]...)),
            stack(hcat(neighbours[i]...)),
            stack(hcat(aversarial_targets[i]...)),
        ) for i in group_indices
    ]
    @assert length(dl) == length(data)

    return dl, percent_valid
end
