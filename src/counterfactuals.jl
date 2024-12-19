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
    targets = rand(counterfactual_data.y_levels, nsamples)       # randomly generate targets

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
    # Unpacking:
    counterfactuals = (ce -> ce.counterfactual).(ces)                               # get actual counterfactuals
    targets_enc = (ce -> target_encoded(ce, counterfactual_data)).(ces)             # encode targets as probabilities

    # Partition data:
    n_total = length(counterfactuals)
    group_indices = TaijaParallel.split_obs(1:n_total, length(data))

    dl = [
        (
            stack(hcat(counterfactuals[i]...)),
            stack(hcat(targets_enc[i]...)),
            stack(hcat(neighbours[i]...)),
        ) for i in group_indices
    ]
    @assert length(dl) == length(data)

    return dl
end
