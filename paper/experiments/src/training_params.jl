using CounterfactualExplanations
using CounterfactualExplanations.Convergence
using CounterfactualExplanations.DataPreprocessing
using MPI: MPI
using TaijaParallel
using Flux

const Opt = Flux.Optimise.AbstractOptimiser

function get_opt(params::AbstractConfiguration)
    # Adam:
    if params.opt == "adam"
        opt = Adam(params.lr)
    end

    # SGD:
    if params.opt == "sgd"
        opt = Descent(params.lr)
    end

    return opt
end

"""
    GeneratorParams

Mutable struct holding keyword arguments relevant to counterfactual generator.
"""
Base.@kwdef struct GeneratorParams <: AbstractGeneratorParams
    type::AbstractGeneratorType = ECCo()
    lr::AbstractFloat = 1.0
    opt::AbstractString = "sgd"
    maxiter::Int = 100
    penalty_strengths::AbstractVector{<:AbstractFloat} = [0.001, 5.0]
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
    return ECCoGenerator(; opt=get_opt(params), λ=params.penalty_strengths[1:2])
end

"Type for the REVISEGenerator."
struct REVISE <: AbstractGeneratorType end

"""
    get_generator(params::GeneratorParams, type::REVISE=params.type)

Instantiates the `REVISEGenerator` with the given parameters.
"""
function get_generator(params::GeneratorParams, generator_type::REVISE)
    return REVISEGenerator(; opt=get_opt(params), λ=params.penalty_strengths[1])
end

"Type for the GenericGenerator."
struct Generic <: AbstractGeneratorType end

"""
    get_generator(params::GeneratorParams, type::Generic)

Instantiates a `GenericGenerator` with the given parameters.
"""
function get_generator(params::GeneratorParams, type::Generic)
    return GenericGenerator(; opt=get_opt(params), λ=params.penalty_strengths[1])
end

"""
    TrainingParams

Mutable struct holding keyword arguments relevant to counterfactual training.
"""
Base.@kwdef struct TrainingParams <: AbstractConfiguration
    burnin::AbstractFloat = 0.0f0
    nepochs::Int = 100
    generator_params::GeneratorParams = GeneratorParams()
    nce::Int = 100
    conv::AbstractString = "max_iter"
    lr::AbstractFloat = 0.001
    opt::AbstractString = "adam"
    parallelizer::AbstractString = "threads"
    threaded::Bool = true
    verbose::Bool = true
end

function get_parallelizer(params::TrainingParams)

    # Multi-threading
    if params.parallelizer == "threads"
        pllr = ThreadsParallelizer()
    end

    # Multi-processing
    if params.parallelizer == "mpi"
        MPI.Init()
        pllr = MPIParallelizer(MPI.COMM_WORLD; threaded=params.threaded)
    end

    return pllr
end

function get_convergence(params::TrainingParams)
    # Maximum iterations:
    if params.conv == "max_iter"
        conv = Convergence.MaxIterConvergence(; max_iter=params.generator_params.maxiter)
    end

    # Decision threshold:
    if params.conv == "threshold"
        conv = Convergence.DecisionThresholdConvergence(; max_iter=params.generator_params.maxiter)
    end

    # Generator conditions:
    if params.conv == "gen_con"
        conv = Convergence.GeneratorConditionsConvergence(; max_iter=params.generator_params.maxiter)
    end
    return conv
end