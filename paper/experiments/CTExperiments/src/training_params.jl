using CounterfactualExplanations
using CounterfactualExplanations.Convergence
using CounterfactualExplanations.DataPreprocessing
using Logging
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

get_generator_name(gen::ECCo) = "ecco"
get_generator_name(gen::Generic) = "generic"
get_generator_name(gen::REVISE) = "revise"

"""
    generator_types

Catalogue of available generator types.
"""
const generator_types = Dict(
    get_generator_name(ECCo()) => ECCo,
    get_generator_name(Generic()) => Generic,
    get_generator_name(REVISE()) => REVISE,
)

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
    lambda_cost::AbstractFloat = 0.001
    lambda_energy::AbstractFloat = 5.0
end

get_generator_name(params::GeneratorParams) = get_generator_name(params.type)

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
    return ECCoGenerator(;
        opt=get_opt(params), λ=[params.lambda_cost, params.lambda_energy]
    )
end

"""
    get_generator(params::GeneratorParams, type::REVISE=params.type)

Instantiates the `REVISEGenerator` with the given parameters.
"""
function get_generator(params::GeneratorParams, generator_type::REVISE)
    return REVISEGenerator(; opt=get_opt(params), λ=params.lambda_cost)
end

"""
    get_generator(params::GeneratorParams, type::Generic)

Instantiates a `GenericGenerator` with the given parameters.
"""
function get_generator(params::GeneratorParams, type::Generic)
    return GenericGenerator(; opt=get_opt(params), λ=params.lambda_cost)
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
    lambda_energy_diff::AbstractFloat = CounterfactualTraining.default_energy_lambda[1]
    lambda_energy_reg::AbstractFloat = CounterfactualTraining.default_energy_lambda[2]
    lambda_adversarial::AbstractFloat = CounterfactualTraining.default_adversarial_lambda
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
    "full" => CounterfactualTraining.FullObjective,
    "energy" => CounterfactualTraining.EnergyDifferentialObjective,
    "adversarial" => CounterfactualTraining.AdversarialObjective,
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

function get_lambdas(obj::CounterfactualTraining.FullObjective, params::TrainingParams)
    lambda = [
        params.lambda_class_loss,
        params.lambda_energy_diff,
        params.lambda_energy_reg,
        params.lambda_adversarial,
    ]
    return lambda
end

function get_lambdas(obj::CounterfactualTraining.EnergyDifferentialObjective, params::TrainingParams)
    lambda = [params.lambda_class_loss, params.lambda_energy_diff, params.lambda_energy_reg]
    return lambda
end

function get_lambdas(obj::CounterfactualTraining.AdversarialObjective, params::TrainingParams)
    lambda = [params.lambda_class_loss, params.lambda_adversarial]
    return lambda
end

function get_parallelizer(pllr_type::String; threaded::Bool=true)
    # Multi-threading
    if pllr_type == "threads"
        pllr = ThreadsParallelizer()
    end

    # Multi-processing
    if pllr_type == "mpi"
        if !MPI.Initialized()
            MPI.Init()
        end
        pllr = MPIParallelizer(MPI.COMM_WORLD; threaded=threaded)
    end

    return pllr
end

function get_parallelizer(params::TrainingParams)
    return get_parallelizer(params.parallelizer; threaded=params.threaded)
end

"""
    objectives

Catalogue of available objective functions.
"""
const conv_catalogue = Dict(
    "max_iter" => Convergence.MaxIterConvergence,
    "threshold" => Convergence.DecisionThresholdConvergence,
    "gen_con" => Convergence.GeneratorConditionsConvergence,
)

function get_convergence(s::String, maxiter::Int)
    s = lowercase(s)
    @assert s in keys(conv_catalogue) "Unknown convergence type: $s. Available types are $(keys(conv_catalogue))"
    conv = conv_catalogue[s](; max_iter=maxiter)
    return conv
end

function get_convergence(params::TrainingParams)
    return get_convergence(params.conv, params.generator_params.maxiter)
end
