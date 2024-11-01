using CounterfactualExplanations
using Flux
using StatsBase

"Base type of training objectives."
abstract type AbstractObjective end

const default_energy_lambda = [0.5, 0001]
const default_adversarial_lambda = 1.0

"""
    EnergyDifferentialObjective <: AbstractObjective

The `EnergyDifferentialObjective` is a concrete implementation of the `AbstractObjective` abstract type that optimizes for:

1. Standard classification objective (the discriminative task)
2. Energy differential between counterfactuals and observed data (the explainability task).
"""
struct EnergyDifferentialObjective <: AbstractObjective
    class_loss::Function
    lambda::Vector{<:AbstractFloat}
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
    lambda::Vector{<:AbstractFloat}=[1.0, default_energy_lambda...]
)
    return EnergyDifferentialObjective(class_loss, lambda)
end

"""
    (obj::EnergyDifferentialObjective)(
        yhat,
        y;
        energy_differential::Vector{<:AbstractFloat},
        regularization::Vector{<:AbstractFloat}=[0.0f0],
        adversarial_loss::Union{AbstractFloat,Vector{<:AbstractFloat}},
        regularization::Vector{<:AbstractFloat}=[0.0f0],
        kwrgs...,
    )

If the `energy_differential` and `regularization` have been computed already, `obj::EnergyDifferentialObjective` can be called directly on predictions `yhat` and labels `y`. The different loss components are then added together with a weighting vector `lambda`.
"""
function (obj::EnergyDifferentialObjective)(
    yhat,
    y;
    energy_differential::Vector{<:AbstractFloat},
    regularization::Vector{<:AbstractFloat}=[0.0f0],
    adversarial_loss::Union{AbstractFloat,Vector{<:AbstractFloat}},
    kwrgs...,
)

    # Compute the standard classification loss:
    class_loss = obj.class_loss(yhat, y; kwrgs...)

    lambda = length(obj.lambda)==2 ? obj.lambda : [obj.lambda, 0.0]     # If there are two components to the loss, then we need to add a dummy component for the regularization term

    # Energy differential:
    implausibility_loss = agg(Float32.(energy_differential))

    # Energy regularization:
    regularization_loss = agg(Float32.(regularization))

    return [class_loss, implausibility_loss, regularization_loss]'lambda
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
    lambda::Vector{<:AbstractFloat}=[1.0,default_adversarial_lambda],
)
    return AdversarialObjective(class_loss, lambda)
end

"""
    (obj::AdversarialObjective)(
        yhat,
        y;
        energy_differential::Vector{<:AbstractFloat},
        regularization::Vector{<:AbstractFloat}=[0.0f0],
        adversarial_loss::Union{AbstractFloat,Vector{<:AbstractFloat}},
        kwrgs...,
    )

If the `adversarial_loss` has been computed already, `obj::AdversarialObjective` can be called directly on predictions `yhat` and labels `y`. The different loss components are then added together with a weighting vector `lambda`.
"""
function (obj::AdversarialObjective)(
    yhat,
    y;
    energy_differential::Vector{<:AbstractFloat},
    regularization::Vector{<:AbstractFloat}=[0.0f0],
    adversarial_loss::Union{AbstractFloat,Vector{<:AbstractFloat}},
    kwrgs...,
)

    # Compute the standard classification loss:
    class_loss = obj.class_loss(yhat, y; kwrgs...)

    # Adversarial loss:
    adversarial_loss = agg(Float32.(adversarial_loss))

    return [class_loss,adversarial_loss]'obj.lambda
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
    energy_differential::PenaltyOrFun
    lambda::Vector{<:AbstractFloat}
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
    energy_differential::PenaltyOrFun=EnergyDifferential(),
    lambda::Vector{<:AbstractFloat}=[1.0,default_energy_lambda...,default_adversarial_lambda]
)
    FullObjective(class_loss, energy_differential, lambda)
end

"""
    (obj::FullObjective)(
        yhat,
        y;
        energy_differential::Vector{<:AbstractFloat},
        regularization::Vector{<:AbstractFloat}=[0.0f0],
        adversarial_loss::Union{AbstractFloat,Vector{<:AbstractFloat}},
        kwrgs...,
    )

If the `adversarial_loss` has been computed already, `obj::FullObjective` can be called directly on predictions `yhat` and labels `y`. The different loss components are then added together with a weighting vector `lambda`.
"""
function (obj::FullObjective)(
    yhat,
    y;
    energy_differential::Vector{<:AbstractFloat},
    regularization::Vector{<:AbstractFloat}=[0.0f0],
    adversarial_loss::Union{AbstractFloat,Vector{<:AbstractFloat}},
    kwrgs...,
)

    # Compute the standard classification loss:
    class_loss = obj.class_loss(yhat, y; kwrgs...)

    # Energy differential:
    implausibility_loss = agg(Float32.(energy_differential))

    # Energy regularization:
    regularization_loss = agg(Float32.(regularization))

    # Adversarial loss:
    adversarial_loss = agg(Float32.(adversarial_loss))

    return [class_loss, implausibility_loss, regularization_loss, adversarial_loss]'obj.lambda
end