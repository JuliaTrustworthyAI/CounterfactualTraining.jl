using CounterfactualExplanations
using CounterfactualExplanations.Convergence
using CounterfactualExplanations.DataPreprocessing
using TaijaParallel
using Flux

const Opt = Flux.Optimise.AbstractOptimiser

"""
    GeneratorParams

Mutable struct holding keyword arguments relevant to counterfactual generator.
"""
Base.@kwdef struct GeneratorParams <: AbstractGeneratorParams
    type::AbstractGeneratorType = ECCo()
    search_opt::Opt = Descent(1.0f0)
    maxiter::Int = 100
    λ::AbstractVector{<:AbstractFloat} = [0.001f0, 5.0f0]
end

"""
    get_generator(params::GeneratorParams)

Instantiates the generator according to the given parameters.
"""
get_generator(params::GeneratorParams) = get_generator(params, params.type)

"Type for the ECCoGenerator."
struct ECCo <: AbstractGeneratorType end

"""
    get_generator(params::GeneratorParams, type::ECCo=params.type)

Instantiates the `ECCoGenerator` with the given parameters.
"""
function get_generator(params::GeneratorParams, generator_type::ECCo)
    return ECCoGenerator(; opt=params.search_opt, λ=params.λ[1:2])
end

"Type for the REVISEGenerator."
struct REVISE <: AbstractGeneratorType end

"""
    get_generator(params::GeneratorParams, type::REVISE=params.type)

Instantiates the `REVISEGenerator` with the given parameters.
"""
function get_generator(params::GeneratorParams, generator_type::REVISE)
    return REVISEGenerator(; opt=params.search_opt, λ=params.λ[1])
end

"Type for the GenericGenerator."
struct Generic <: AbstractGeneratorType end

"""
    get_generator(params::GeneratorParams, type::Generic)

Instantiates a `GenericGenerator` with the given parameters.
"""
function get_generator(params::GeneratorParams, type::Generic)
    return GenericGenerator(; opt=params.search_opt, λ=params.λ[1])
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
    conv::AbstractConvergence = Convergence.MaxIterConvergence(; max_iter=generator_params.maxiter)
    training_opt::Opt = Adam()
    parallelizer::AbstractParallelizer = ThreadsParallelizer()
    verbose::Bool = true
end