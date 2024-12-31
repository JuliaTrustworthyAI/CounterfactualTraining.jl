using EnergySamplers: EnergySamplers
using LinearAlgebra

function implausibility(model, perturbed_input, samples, targets)
    x = (model(samples) - model(perturbed_input))'targets
    return diag(x[:, :])
end

function reg_loss(model, perturbed_input, samples, targets)
    x = (abs2.(model(samples)) + abs2.(model(perturbed_input)))'targets
    return diag(x[:, :])
end
