using CounterfactualExplanations
import CounterfactualTraining
using Flux

struct OmniscientGenerator <: AbstractGenerator end

function CounterfactualTraining.generate!(
    model,
    data,
    generator::OmniscientGenerator;
    nsamples::Union{Int,Nothing}=nothing,
    nneighbours::Int=100,
    convergence=Convergence.MaxIterConvergence(),
    parallelizer=nothing,
    input_encoder=nothing,
    verbose=1,
    domain=nothing,
)

    # Setup counterfactual search
    xs, factual_enc, targets, counterfactual_data, M = CounterfactualTraining.setup_counterfactual_search(
        data, model, domain, input_encoder, nneighbours, nsamples
    )

    # Get neighbours in target class and set counterfactuals to neighbours:
    targets_enc = (target -> Flux.onehotbatch([target], counterfactual_data.y_levels)).(targets)
    predicted_labels = [argmax(M.model(x)) for x in xs]
    in_target_class = (target -> findall(predicted_labels .== target)).(targets)
    neighbours = Vector(undef, length(xs))
    adversarial_targets = Vector(undef, length(xs))
    for (i, candidates) in enumerate(in_target_class)
        if length(candidates) > 0
            neighbours[i] = xs[rand(candidates)]
            adversarial_targets[i] = targets_enc[i]
        else
            neighbours[i] = xs[rand(1:length(xs))]
            adversarial_targets[i] = factual_enc[:, i]
        end
    end
    counterfactuals = neighbours
    percent_valid = 1.0

    # Partition data:
    n_total = length(counterfactuals)
    group_indices = TaijaParallel.split_obs(1:n_total, length(data))
    dl = [
        (
            stack(hcat(counterfactuals[i]...)),
            stack(hcat(targets_enc[i]...)),
            stack(hcat(neighbours[i]...)),
            stack(hcat(adversarial_targets[i]...)),
        ) for i in group_indices
    ]
    @assert length(dl) == length(data)

    return dl, percent_valid
end

