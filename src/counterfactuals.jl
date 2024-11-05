using CounterfactualExplanations
using CounterfactualExplanations: Convergence
using CounterfactualExplanations: Models
using Flux
using StatsBase
using TaijaParallel

function generate!(
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

    # Wrap training dataset in `CounterfactualData`:
    X, y = unwrap(data)
    counterfactual_data = CounterfactualData(
        X, y; domain=domain, input_encoder=input_encoder
    )
    # Wrap model:
    M = Models.Model(model, Models.FluxNN(); likelihood=:classification_multi)

    # Set up counterfactual search:
    if isnothing(nsamples)
        nsamples = size(X, 2)
        # Use whole dataset:
        xs = [x[:, :] for x in eachcol(X)]                          # factuals
    else
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
        verbose=verbose > 1,
    )
    counterfactuals = hcat(CounterfactualExplanations.counterfactual.(ces)...)      # counterfactual inputs

    # Get neighbours in target class:
    neighbours = (X -> X[:, 1])([
        CounterfactualExplanations.find_potential_neighbours(ce, 10) for ce in ces
    ])

    # Encoded targets:
    targets_enc = hcat((x -> x.target_encoded).(ces)...)

    # Return data:
    bs = Int(round(size(counterfactuals, 2) / length(data)))
    dl = Flux.DataLoader(
        (counterfactuals, targets, targets_enc, neighbours); batchsize=bs, shuffle=false
    )

    return dl
end
