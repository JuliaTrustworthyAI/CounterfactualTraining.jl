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
    implausibility_and_reg_loss(model, counterfactual, samples, targets)

Computes both [`implausibility`](@ref) and [`reg_loss`](@ref) in a single pass,
sharing the forward passes through `model` for `samples` and `counterfactual`.
Returns `(implaus, regs)` — the same values that `implausibility(...)` and
`reg_loss(...)` would return separately.

This avoids redundant forward passes when both losses are needed (e.g. inside
the training loop's gradient tape).
"""
function implausibility_and_reg_loss(model, counterfactual, samples, targets)
    logits_samples = model(samples)
    logits_cf = model(counterfactual)
    # implausibility: (-logits_samples) - (-logits_cf) = logits_cf - logits_samples
    # x = ((E(samples)) - (E(counterfactual)))[:, :]'targets
    #   = (logits_cf - logits_samples)'targets
    implaus_x = (logits_cf .- logits_samples)[:, :]' * targets
    implaus = diag(implaus_x[:, :])
    # reg_loss: (abs2.(model(samples)) + abs2.(model(counterfactual)))'targets
    reg_x = (abs2.(logits_samples) .+ abs2.(logits_cf))' * targets
    regs = diag(reg_x[:, :])
    return implaus, regs
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
