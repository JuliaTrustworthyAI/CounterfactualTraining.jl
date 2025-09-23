using Flux
using Flux: logitcrossentropy
using StatsBase

"""
    generate_ae(model, x, y; attack_fun::Function=fgsm, eps::Real=0.3, kwrgs...)

Generate an adversarial example of `x` for `model` using attack `attack_fun`.
"""
function generate_ae(model, x, y; attack_fun::Function=fgsm, eps::Real=0.3, kwrgs...)
    return attack_fun(model, x, y; eps, kwrgs...)
end

"""
    fgsm(
        model,
        x,
        y;
        loss = logitcrossentropy,
        ϵ = 0.3,
        clamp_range::Union{Nothing,Tuple} = nothing,
    )
    
White-box Fast Gradient Sign Method (FGSM) attack by Goodfellow et al. (arxiv.org/abs/1412.6572). Code adapted from https://github.com/JuliaTrustworthyAI/AdversarialRobustness.jl/blob/main/src/attacks/fgsm/fgsm.jl.
"""
function fgsm(
    model, x, y; loss=logitcrossentropy, eps=0.3, clamp_range::Union{Nothing,Tuple}=nothing
)

    # ATTACK!
    grads = gradient(x -> loss(model(x), y), x)
    perturbations = (eps .* sign.(grads[1]))
    x .+= perturbations

    # Clamp
    if !isnothing(clamp_range)
        x = clamp.(x, clamp_range...)
    end

    return x
end

"""
    random_perturbation(
        model,
        x,
        y;
        loss=logitcrossentropy,
        eps::Real=0.3,
        clamp_range::Union{Nothing,Tuple}=nothing,
    )

Adding random noise to data.
"""
function random_perturbation(
    model,
    x,
    y;
    loss=logitcrossentropy,
    eps::Real=0.3,
    clamp_range::Union{Nothing,Tuple}=nothing,
)
    # ATTACK!
    delta = randn(size(x)...)
    x += eps * delta

    # Clamp
    if !isnothing(clamp_range)
        x = clamp.(x, clamp_range...)
    end

    return x
end
