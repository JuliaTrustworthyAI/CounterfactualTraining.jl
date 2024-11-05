using CounterfactualExplanations
using CounterfactualExplanations.Convergence
using CounterfactualExplanations.DataPreprocessing
using MPI: MPI
using TaijaParallel
using Flux

const Opt = Flux.Optimise.AbstractOptimiser

"Type for the ECCoGenerator."
struct ECCo <: AbstractGeneratorType end

"Type for the REVISEGenerator."
struct REVISE <: AbstractGeneratorType end

"Type for the GenericGenerator."
struct Generic <: AbstractGeneratorType end

"""
    generator_types

Catalogue of available generator types.
"""
const generator_types = Dict("ecco" => ECCo, "generic" => Generic, "revise" => REVISE)

"""
    get_generator_type(name::String)

Retrieves the generator type from the catalogue if available.
"""
function get_generator_type(s::String)
    s = lowercase(s)
    @assert s in keys(generator_types) "Unknown generator type: $s. Available types are $(keys(generator_types))"
    return generator_types[s]
end

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

"""
    get_generator(params::GeneratorParams, type::ECCo=params.type)

Instantiates the `ECCoGenerator` with the given parameters.
"""
function get_generator(params::GeneratorParams, generator_type::ECCo)
    return ECCoGenerator(; opt=get_opt(params), λ=params.penalty_strengths[1:2])
end

"""
    get_generator(params::GeneratorParams, type::REVISE=params.type)

Instantiates the `REVISEGenerator` with the given parameters.
"""
function get_generator(params::GeneratorParams, generator_type::REVISE)
    return REVISEGenerator(; opt=get_opt(params), λ=params.penalty_strengths[1])
end

"""
    get_generator(params::GeneratorParams, type::Generic)

Instantiates a `GenericGenerator` with the given parameters.
"""
function get_generator(params::GeneratorParams, type::Generic)
    return GenericGenerator(; opt=get_opt(params), λ=params.penalty_strengths[1])
end

"""
    class_losses

Catalouge of available class losses.
"""
const class_losses = Dict("logitcrossentropy" => Flux.Losses.logitcrossentropy)

"""
    get_classloss(s::String)

Retrieves the class loss from the catalogue if available.
"""
function get_class_loss(s::String)
    s = lowercase(s)
    @assert s in keys(class_losses) "Unknown class loss function: $s. Available types are $(keys(class_losses))"
    return class_losses[s]
end

"""
    TrainingParams

Mutable struct holding keyword arguments relevant to counterfactual training.
"""
Base.@kwdef struct TrainingParams <: AbstractConfiguration
    objective::AbstractString = "full"
    lambda_class_loss::AbstractFloat = 1.0
    lambda_energy_diff::AbstractFloat = CT.default_energy_lambda[1]
    lambda_energy_reg::AbstractFloat = CT.default_energy_lambda[2]
    lambda_adversarial::AbstractFloat = CT.default_adversarial_lambda
    class_loss::AbstractString = "logitcrossentropy"
    burnin::AbstractFloat = 0.0f0
    nepochs::Int = 100
    generator_params::GeneratorParams = GeneratorParams()
    nce::Int = 100
    conv::AbstractString = "max_iter"
    lr::AbstractFloat = 0.001
    opt::AbstractString = "adam"
    parallelizer::AbstractString = "threads"
    threaded::Bool = true
    verbose::Int = 2
end

"""
    objectives

Catalogue of available objective functions.
"""
const objectives = Dict(
    "full" => CT.FullObjective,
    "energy" => CT.EnergyDifferentialObjective,
    "adversarial" => CT.AdversarialObjective,
)

"""
    get_objective(s::String)

Retrieves the objective type from the catalogue if available.
"""
function get_objective(s::String)
    s = lowercase(s)
    @assert s in keys(objectives) "Unknown objective type: $s. Available types are $(keys(objectives))"
    return objectives[s]
end

function get_lambdas(obj::CT.FullObjective, params::TrainingParams)
    lambda = [
        params.lambda_class_loss,
        params.lambda_energy_diff,
        params.lambda_energy_reg,
        params.lambda_adversarial,
    ]
    return lambda
end

function get_lambdas(obj::CT.EnergyDifferentialObjective, params::TrainingParams)
    lambda = [params.lambda_class_loss, params.lambda_energy_diff, params.lambda_energy_reg]
    return lambda
end

function get_lambdas(obj::CT.AdversarialObjective, params::TrainingParams)
    lambda = [params.lambda_class_loss, params.lambda_adversarial]
    return lambda
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
        if MPI.Comm_rank(MPI.COMM_WORLD) != 0
            global_logger(NullLogger())
        else
            @info "Multi-processing using MPI. Disabling logging on non-root processes."
            if params.threaded
                @info "Multi-threading using $(Threads.nthreads()) threads."
                if Threads.threadid() != 1
                    global_logger(NullLogger())
                end
            end
        end
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
        conv = Convergence.DecisionThresholdConvergence(;
            max_iter=params.generator_params.maxiter
        )
    end

    # Generator conditions:
    if params.conv == "gen_con"
        conv = Convergence.GeneratorConditionsConvergence(;
            max_iter=params.generator_params.maxiter
        )
    end
    return conv
end
