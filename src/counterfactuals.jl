using Base.Iterators
using CounterfactualExplanations
using CounterfactualExplanations: Convergence
using CounterfactualExplanations: Models
using Flux
using StatsBase
using TaijaParallel

"""
    generate!(
        input,
        model,
        data,
        generator;
        convergence=Convergence.MaxIterConvergence(),
        parallelizer=nothing,
        input_encoder=nothing,
        verbose=1,
        domain=nothing,
    )

This function generates counterfactual explanations for a given array of inputs `input`. It is supposed to be used inside of the mini-batch training loop. The model `model` is expected to be a Flux model that takes an input and returns a vector of predictions. The data `data` is expected to a training dataset compatible with Flux training. The generator `generator` is expected to be a function that generates new inputs. It is used to generate counterfactuals.
"""
function generate!(
    input,
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

    # Set up:
    # NOTE: Using [1,...,n] for labels where n is the number of output classes. Exact label information is not necessary for training.
    counterfactual_data = CounterfactualData(
        unwrap(data)...; domain=domain, input_encoder=input_encoder
    )
    M = Models.Model(model, Models.FluxNN(); likelihood=:classification_multi)

    # Set up counterfactual search:
    if isnothing(nsamples)
        nsamples = size(input, 2)
        # Use whole dataset:
        xs = [x[:, :] for x in eachcol(input)]                      # factuals
    else
        # Use subset:
        Xsub = input[:, sample(1:size(input, 2), nsamples)]
        xs = [x[:, :] for x in eachcol(Xsub)]                       # factuals
    end
    targets = rand(counterfactual_data.y_levels, nsamples)          # randomly generate targets

    ces = TaijaParallel.parallelize(
        parallelizer,
        CounterfactualExplanations.generate_counterfactual,
        xs,
        targets,
        counterfactual_data,
        M,
        generator;
        convergence=convergence,
        verbose=verbose >= 1,
    )
    counterfactuals = hcat(CounterfactualExplanations.counterfactual.(ces)...)

    # Get neighbours in target class:
    neighbours = (X -> X[:, 1])([
        CounterfactualExplanations.find_potential_neighbours(ce, 10) for ce in ces
    ])

    # Encoded targets:
    targets_enc = hcat((x -> x.target_encoded).(ces)...)

    # Target indices:
    target_indices = get_target_index.((counterfactual_data.y_levels,), targets)

    return counterfactuals, target_indices, targets_enc, neighbours
end

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
        verbose=verbose >= 1,
    )

    counterfactuals = CounterfactualExplanations.counterfactual.(ces)      # counterfactual inputs

    # Get neighbours in target class:
    neighbours = [
        CounterfactualExplanations.find_potential_neighbours(ce, nneighbours) for ce in ces
    ]

    # Encoded targets:
    targets_enc = (x -> x.target_encoded).(ces)

    # Target indices:
    target_indices = get_target_index.((counterfactual_data.y_levels,), targets)

    # Partition data:
    all_data = zip(counterfactuals, target_indices, targets_enc, neighbours)
    n_total = length(all_data)
    group_indices = TaijaParallel.split_obs(1:n_total, length(data))

    dl = [
        (
            hcat(counterfactuals[i]...),
            target_indices[i],
            hcat(targets_enc[i]...),
            neighbours[i],
        ) for i in group_indices
    ]
    @assert length(dl) == length(data)

    return dl
end
