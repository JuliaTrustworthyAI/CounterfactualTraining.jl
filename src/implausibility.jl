using EnergySamplers: EnergySamplers
using LinearAlgebra

@doc raw"""
    implausibility(model, perturbed_input, samples, targets)

Compute the implausibility (contrastive divergence) of the counterfactuals (`perturbed_input`) with respect to `samples` in the target class. This is computed as the difference between negative logits indexed at the target class for the `samples` and the `perturbed_input`.
"""
function implausibility(model, perturbed_input, samples, targets)
    E(x) = -model(x)                                    # energy
    x = ((E(samples)) - (E(perturbed_input)))'targets   # contrastive divergence
    return diag(x[:, :])
end

function reg_loss(model, perturbed_input, samples, targets)
    x = (abs2.(model(samples)) + abs2.(model(perturbed_input)))'targets
    return diag(x[:, :])
end
