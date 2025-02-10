using EnergySamplers: EnergySamplers
using LinearAlgebra

@doc raw"""
    implausibility(model, counterfactual, samples, targets)

Compute the implausibility (contrastive divergence) of the counterfactuals (`counterfactual`) with respect to `samples` in the target class. This is computed as the difference between negative logits indexed at the target class for the `samples` and the `counterfactual`.
"""
function implausibility(model, counterfactual, samples, targets)
    E(x) = -model(x)                                            # energy
    x = ((E(samples)) - (E(counterfactual)))[:,:]'targets       # contrastive divergence
    # x = ((E(counterfactual)) - (E(samples)))'targets   # contrastive divergence
    return diag(x[:, :])
end

function reg_loss(model, counterfactual, samples, targets)
    x = (abs2.(model(samples)) + abs2.(model(counterfactual)))'targets
    return diag(x[:, :])
end
