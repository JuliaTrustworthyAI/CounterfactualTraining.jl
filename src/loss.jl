using EnergySamplers: EnergySamplers
using Flux
using LinearAlgebra

@doc raw"""
    implausibility(model, counterfactual, samples, targets)

Compute the implausibility (contrastive divergence) of the counterfactuals (`counterfactual`) with respect to `samples` in the target class. This is computed as the difference between negative logits indexed at the target class for the `samples` and the `counterfactual`.
"""
function implausibility(model, counterfactual, samples, targets)
    E(x) = -model(x)                                        # energy
    x = ((E(samples)) - (E(counterfactual)))[:, :]'targets  # contrastive divergence
    return diag(x[:, :])
end

"""
    reg_loss(model, counterfactual, samples, targets)

Compute the regularization loss for the contrastice divergence.
"""
function reg_loss(model, counterfactual, samples, targets)
    x = (abs2.(model(samples)) + abs2.(model(counterfactual)))'targets
    return diag(x[:, :])
end

"""
    adv_loss(
        model, counterfactual, perturbations, targets; epsilon=2.0, p::Real=Inf, validities=nothing
    )

Adversarial loss function.
"""
function adv_loss(
    model,
    counterfactual,
    perturbations,
    targets;
    epsilon=0.5,
    p::Real=Inf,
    validities=nothing,
)
    # Identify adversarial examples
    idx_advexm = [
        isadvexm(perturbation, epsilon, p) for perturbation in eachcol(perturbations)
    ]
    if sum(idx_advexm) > 0
        println("Percent AE: $(sum(idx_advexm)/length(idx_advexm))")
        yhat_ce = model(counterfactual[:, idx_advexm])   # predictions
        return Flux.logitcrossentropy(yhat_ce, targets[:, idx_advexm])
    else
        return 0.0f0
    end
end

abstract type AbstractAECriterium end

Base.@kwdef struct NormBound <: AbstractAECriterium
    epsilon::AbstractFloat = 0.3
    p::Real = Inf
end

(nmb::NormBound)(perturbation::AbstractArray) = isadvexm(perturbation, nmb.epsilon, nmb.p)

isadvexm(perturbation, epsilon, p) = abs(norm(perturbation, p)) <= epsilon

global _global_ae_criterium = NormBound()

"""
    get_global_ae_criterium()

Get the global AE criterium.
"""
get_global_ae_criterium() = _global_ae_criterium

function set_global_ae_criterium(aecrit::AbstractAECriterium)
    global _global_ae_criterium = aecrit
    return _global_ae_criterium
end
