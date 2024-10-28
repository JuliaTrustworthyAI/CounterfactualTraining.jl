using CounterfactualExplanations
using CounterfactualExplanations.Convergence
using CounterfactualExplanations.DataPreprocessing
using TaijaParallel
using Flux

const Opt = Flux.Optimise.AbstractOptimiser

abstract type AbstractGeneratorType end

"Type for the ECCoGenerator."
struct ECCo end

"""
    get_generator(params::GeneratorParams, type::ECCo=params.type)::AbstractGenerator

Instantiates the `ECCoGenerator` with the given parameters.
"""
function get_generator(params::GeneratorParams, type::ECCo=params.type)::AbstractGenerator
    return ECCoGenerator(; opt=params.search_opt, λ=params.λ[1:2])
end

"Type for the REVISEGenerator."
struct REVISE end

"""
    get_generator(params::GeneratorParams, type::REVISE=params.type)::AbstractGenerator

Instantiates the `REVISEGenerator` with the given parameters.
"""
function get_generator(params::GeneratorParams, type::REVISE=params.type)::AbstractGenerator
    return REVISEGenerator(; opt=params.search_opt, λ=params.λ[1])
end

"Type for the GenericGenerator."
struct Generic end

"""
    get_generator(params::GeneratorParams, type::Generic=params.type)::AbstractGenerator

Instantiates a `GenericGenerator` with the given parameters.
"""
function get_generator(params::GeneratorParams, type::Generic=params.type)::AbstractGenerator
    return GenericGenerator(; opt=params.search_opt, λ=params.λ[1])
end

"""
    GeneratorParams

Mutable struct holding keyword arguments relevant to counterfactual generator.
"""
Base.@kwdef struct GeneratorParams
    type::AbstractGeneratorType = ECCo()
    search_opt::Opt = Descent(1.0f0)
    max_iter::Int = 50
    λ::AbstractVector{<:AbstractFloat} = [0.001f0, 5.0f0]
end

"""
    TrainingParams

Mutable struct holding keyword arguments relevant to counterfactual training.
"""
Base.@kwdef struct TrainingParams
    burnin::AbstractFloat = 0.0f0
    nepochs::Int = 100
    generator_params::GeneratorParams = GeneratorParams()
    nce::Int = 100
    conv::AbstractConvergence = Convergence.MaxIterConvergence(; max_iter=max_iter)
    training_opt::Opt = Adam()
    input_encoder::Union{Nothing,DataPreprocessing.InputTransformer} = nothing
    parallelizer::AbstractParallelizer = ThreadsParallelizer()
    verbose::Bool = true
end