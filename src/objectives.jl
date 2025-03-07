using CounterfactualExplanations
using Flux
using StatsBase

"Base type of training objectives."
abstract type AbstractObjective end

const default_energy_lambda = [0.5, 0.05]
const default_adversarial_lambda = 0.5

needs_counterfactuals(obj::AbstractObjective) = true

"""
    VanillaObjective <: AbstractObjective

The `VanillaObjective` is a concrete implementation of the `AbstractObjective` abstract type that optimizes for:

1. Standard classification objective (the discriminative task).
"""
struct VanillaObjective <: AbstractObjective
    class_loss::Function
    lambda::Vector{<:AbstractFloat}
    needs_ce::Bool
    function VanillaObjective(class_loss, lambda, needs_ce)
        @assert length(lambda) == 1 "Need exactly one values in lambda for the class loss."
        return new(class_loss, lambda, needs_ce)
    end
end

needs_counterfactuals(obj::VanillaObjective) = obj.needs_ce

"""
    VanillaObjective(class_loss, lambda; needs_ce=false)

Outer constructor to allow passing just `class_loss` and `lambda` as positional arguments.
"""
VanillaObjective(class_loss, lambda; needs_ce=false) =
    VanillaObjective(class_loss, lambda, needs_ce)

"""
    VanillaObjective(;
        class_loss::Function=Flux.Losses.logitcrossentropy,
        lambda::Vector{<:AbstractFloat}=[1.0],
        needs_ce::Bool=false,
    )

Outer constructor for the `VanillaObjective` type.
"""
function VanillaObjective(;
    class_loss::Function=Flux.Losses.logitcrossentropy,
    lambda::Vector{<:AbstractFloat}=[1.0],
    needs_ce::Bool=false,
)
    return VanillaObjective(class_loss, lambda, needs_ce)
end

"""
    (obj::VanillaObjective)(
        yhat,
        y,
        energy_differential::Vector{<:AbstractFloat}=[0.0f0],
        regularization::Vector{<:AbstractFloat}=[0.0f0],
        adversarial_loss::Union{AbstractFloat,Vector{<:AbstractFloat}}=0.0f0;
        agg=mean,
        kwrgs...,
    )

`obj::VanillaObjective` can be called directly on predictions `yhat` and labels `y`.
"""
function (obj::VanillaObjective)(
    yhat,
    y,
    energy_differential::Vector{<:AbstractFloat}=[0.0f0],
    regularization::Vector{<:AbstractFloat}=[0.0f0],
    adversarial_loss::Union{AbstractFloat,Vector{<:AbstractFloat}}=0.0f0;
    agg=mean,
    kwrgs...,
)

    # Compute the standard classification loss:
    class_loss = obj.class_loss(yhat, y; agg=agg, kwrgs...)

    return [class_loss]'obj.lambda
end

"""
    EnergyDifferentialObjective <: AbstractObjective

The `EnergyDifferentialObjective` is a concrete implementation of the `AbstractObjective` abstract type that optimizes for:

1. Standard classification objective (the discriminative task)
2. Energy differential between counterfactuals and observed data (the explainability task).
"""
struct EnergyDifferentialObjective <: AbstractObjective
    class_loss::Function
    lambda::Vector{<:AbstractFloat}
    function EnergyDifferentialObjective(class_loss, lambda)
        if length(lambda) < 3
            lambda = [obj.lambda..., 0.0]       # Add a dummy value of 0.0 for the reguilarization term.
        end
        @assert length(lambda) == 3 "Need exactly three values in lambda: for the class loss, energy differential, and regularization term, respectively and in that order."
        return new(class_loss, lambda)
    end
end

"""
    EnergyDifferentialObjective(;
        class_loss::Function=Flux.Losses.logitcrossentropy, 
        lambda::Vector{<:AbstractFloat}=$([1.0, default_energy_lambda...])
    )

Outer constructor for the `EnergyDifferentialObjective` type.
"""
function EnergyDifferentialObjective(;
    class_loss::Function=Flux.Losses.logitcrossentropy,
    lambda::Vector{<:AbstractFloat}=[1.0, default_energy_lambda...],
)
    return EnergyDifferentialObjective(class_loss, lambda)
end

"""
    (obj::EnergyDifferentialObjective)(
        yhat,
        y,
        energy_differential::Vector{<:AbstractFloat}=[0.0f0],
        regularization::Vector{<:AbstractFloat}=[0.0f0],
        adversarial_loss::Union{AbstractFloat,Vector{<:AbstractFloat}}=0.0f0;
        agg=mean,
        kwrgs...,
    )

If the `energy_differential` and `regularization` have been computed already, `obj::EnergyDifferentialObjective` can be called directly on predictions `yhat` and labels `y`. The different loss components are then added together with a weighting vector `lambda`.
"""
function (obj::EnergyDifferentialObjective)(
    yhat,
    y,
    energy_differential::Vector{<:AbstractFloat}=[0.0f0],
    regularization::Vector{<:AbstractFloat}=[0.0f0],
    adversarial_loss::Union{AbstractFloat,Vector{<:AbstractFloat}}=0.0f0;
    agg=mean,
    kwrgs...,
)

    # Compute the standard classification loss:
    class_loss = obj.class_loss(yhat, y; agg=agg, kwrgs...)

    # Energy differential:
    implausibility_loss = agg(Float32.(energy_differential))

    # Energy regularization:
    regularization_loss = agg(Float32.(regularization))

    # Aggregate:
    losses = [class_loss, implausibility_loss, regularization_loss]

    return losses'obj.lambda
end

"""
    AdversarialObjective <: AbstractObjective

The `AdversarialObjective` is a concrete implementation of the `AbstractObjective` abstract type that optimizes for:

1. Standard classification objective (the discriminative task).
2. Adversarial classification objective on the counterfactuals (the explainability task).
"""
struct AdversarialObjective <: AbstractObjective
    class_loss::Function
    lambda::Vector{<:AbstractFloat}
    function AdversarialObjective(class_loss, lambda)
        @assert length(lambda) == 2 "Need exactly two values in lambda: for the class loss and the adversarial loss, respectively and in that order."
        return new(class_loss, lambda)
    end
end

"""
    AdversarialObjective(;
        class_loss::Function=Flux.Losses.logitcrossentropy,
        lambda::Vector{<:AbstractFloat}=$([1.0,default_adversarial_lambda])
    )

Outer constructor for the `AdversarialObjective` type.
"""
function AdversarialObjective(;
    class_loss::Function=Flux.Losses.logitcrossentropy,
    lambda::Vector{<:AbstractFloat}=[1.0, default_adversarial_lambda],
)
    return AdversarialObjective(class_loss, lambda)
end

"""
    (obj::AdversarialObjective)(
        yhat,
        y,
        energy_differential::Vector{<:AbstractFloat}=[0.0f0],
        regularization::Vector{<:AbstractFloat}=[0.0f0],
        adversarial_loss::Union{AbstractFloat,Vector{<:AbstractFloat}}=0.0f0;
        agg=mean,
        kwrgs...,
    )

If the `adversarial_loss` has been computed already, `obj::AdversarialObjective` can be called directly on predictions `yhat` and labels `y`. The different loss components are then added together with a weighting vector `lambda`.
"""
function (obj::AdversarialObjective)(
    yhat,
    y,
    energy_differential::Vector{<:AbstractFloat}=[0.0f0],
    regularization::Vector{<:AbstractFloat}=[0.0f0],
    adversarial_loss::Union{AbstractFloat,Vector{<:AbstractFloat}}=0.0f0;
    agg=mean,
    kwrgs...,
)

    # Compute the standard classification loss:
    class_loss = obj.class_loss(yhat, y; agg=agg, kwrgs...)

    # Adversarial loss:
    adversarial_loss = agg(Float32.(adversarial_loss))

    return [class_loss, adversarial_loss]'obj.lambda
end

"""
    FullObjective <: AbstractObjective

The `FullObjective` is a concrete implementation of the `AbstractObjective` abstract type that optimizes for all three tasks:

1. Standard classification objective (the discriminative task)
2. Energy differential between counterfactuals and observed data (the explainability task).
3. Adversarial classification objective on the counterfactuals (the explainability task).
"""
struct FullObjective <: AbstractObjective
    class_loss::Function
    lambda::Vector{<:AbstractFloat}
    function FullObjective(class_loss, lambda)
        @assert length(lambda) == 4 "Need exactly four values in lambda: for the class loss, energy differential, regularization term and adversarial loss, respectively and in that order."
        return new(class_loss, lambda)
    end
end

"""
    FullObjective(;
        class_loss::Function=Flux.Losses.logitcrossentropy,
        energy_differential::PenaltyOrFun=EnergyDifferential(),
        lambda::Vector{<:AbstractFloat}=$([1.0,default_energy_lambda...,default_adversarial_lambda])
    )

Outer constructor for the `FullObjective` type.
"""
function FullObjective(;
    class_loss::Function=Flux.Losses.logitcrossentropy,
    lambda::Vector{<:AbstractFloat}=[
        1.0, default_energy_lambda..., default_adversarial_lambda
    ],
)
    return FullObjective(class_loss, lambda)
end

"""
    (obj::FullObjective)(
        yhat,
        y,
        energy_differential::Vector{<:AbstractFloat}=[0.0f0],
        regularization::Vector{<:AbstractFloat}=[0.0f0],
        adversarial_loss::Union{AbstractFloat,Vector{<:AbstractFloat}}=[0.0f0];
        agg=mean,
        kwrgs...,
    )

If the `adversarial_loss` has been computed already, `obj::FullObjective` can be called directly on predictions `yhat` and labels `y`. The different loss components are then added together with a weighting vector `lambda`.
"""
function (obj::FullObjective)(
    yhat,
    y,
    energy_differential::Vector{<:AbstractFloat}=[0.0f0],
    regularization::Vector{<:AbstractFloat}=[0.0f0],
    adversarial_loss::Union{AbstractFloat,Vector{<:AbstractFloat}}=[0.0f0];
    agg=mean,
    kwrgs...,
)

    # Compute the standard classification loss:
    class_loss = obj.class_loss(yhat, y; agg=agg, kwrgs...)

    # Energy differential:
    implausibility_loss = agg(Float32.(energy_differential))

    # Energy regularization:
    regularization_loss = agg(Float32.(regularization))

    # Adversarial loss:
    adversarial_loss = agg(Float32.(adversarial_loss))

    return [
        class_loss, implausibility_loss, regularization_loss, adversarial_loss
    ]'obj.lambda
end
