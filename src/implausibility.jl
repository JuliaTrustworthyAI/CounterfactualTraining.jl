using EnergySamplers: EnergySamplers
using LinearAlgebra

function implausibility(model, perturbed_input, samples, targets)
    return diag((model(samples) - model(perturbed_input))'targets)
end

function reg_loss(model, perturbed_input, samples, targets)
    return EnergySamplers.energy_penalty.(
        (model,), eachcol(perturbed_input), samples, targets
    )
end
