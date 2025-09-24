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
    pgd(
        model,
        x,
        y;
        loss = logitcrossentropy,
        ε = 0.3,
        α = 0.01,
        num_steps = 40,
        random_start = true,
        clamp_range::Union{Nothing,Tuple} = nothing,
    )

Projected Gradient Descent (PGD) attack by Madry et al. (arxiv.org/abs/1706.06083).
This is an iterative attack that applies multiple small FGSM-like steps with projection
back to the ε-ball around the original input.

# Arguments

- `model`: The neural network model to attack
- `x`: Input data to perturb
- `y`: True labels
- `loss`: Loss function (default: logitcrossentropy)
- `ε`: Maximum perturbation magnitude (L∞ norm)
- `α`: Step size for each iteration
- `num_steps`: Number of PGD iterations
- `random_start`: Whether to start from a random point in the ε-ball
- `clamp_range`: Range to clamp final values (e.g., (0, 1) for images)
"""
function pgd(
    model,
    x,
    y;
    loss=logitcrossentropy,
    ε=0.3,
    α=0.01,
    num_steps=40,
    random_start=true,
    clamp_range::Union{Nothing,Tuple}=nothing,
)
    x_orig = copy(x)
    x_adv = copy(x)

    # Random start: begin from random point in ε-ball

    if random_start
        noise = (2 * rand(eltype(x), size(x)...) .- 1) .* ε
        x_adv .+= noise
        # Project back to ε-ball
        x_adv = x_orig .+ clamp.(x_adv .- x_orig, -ε, ε)
    end

    for i in 1:num_steps
        # Compute gradients
        grads = gradient(x -> loss(model(x), y), x_adv)

        # Take step in direction of gradient sign
        perturbation = α .* sign.(grads[1])
        x_adv .+= perturbation

        # Project back to ε-ball around original input
        x_adv = x_orig .+ clamp.(x_adv .- x_orig, -ε, ε)

        # Clamp to valid input range if specified
        if !isnothing(clamp_range)
            x_adv = clamp.(x_adv, clamp_range...)
        end
    end

    return x_adv
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
