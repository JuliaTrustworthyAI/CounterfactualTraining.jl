using Accessors
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

"Type for the GravitationalGenerator."
struct Gravitational <: AbstractGeneratorType end

struct Omniscient <: AbstractGeneratorType end

get_generator_name(gen::ECCo; pretty::Bool=false) = pretty ? "ECCo" : "ecco"
get_generator_name(gen::Generic; pretty::Bool=false) = pretty ? "Generic" : "generic"
get_generator_name(gen::REVISE; pretty::Bool=false) = pretty ? "REVISE" : "revise"
function get_generator_name(gen::Gravitational; pretty::Bool=false)
    return pretty ? "Gravitational" : "gravi"
end
get_generator_name(gen::Omniscient; pretty::Bool=false) = pretty ? "Omniscient" : "omni"

"""
    generator_types

Catalogue of available generator types.
"""
const generator_types = Dict(
    get_generator_name(ECCo()) => ECCo,
    get_generator_name(Generic()) => Generic,
    get_generator_name(REVISE()) => REVISE,
    get_generator_name(Gravitational()) => Gravitational,
    get_generator_name(Omniscient()) => Omniscient,
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

const available_optimizers = Dict(
    "adam" => Flux.Optimise.Adam, "sgd" => Flux.Optimise.Descent
)

"""
    get_opt(params::AbstractConfiguration)
    
Retrieves the optimizer from the configuration.
"""
get_opt(params::AbstractConfiguration) = get_opt(params.opt)

function get_opt(s::String)
    s = lowercase(s)
    @assert s in keys(available_optimizers) "Unknown optimizer : $s. Available types are $(keys(available_optimizers))"
    return available_optimizers[s]()
end

"""
    GeneratorParams

Mutable struct holding keyword arguments relevant to counterfactual generator.
"""
Base.@kwdef struct GeneratorParams <: AbstractGeneratorParams
    type::AbstractGeneratorType = ECCo()
    lr::AbstractFloat = 1.0
    opt::AbstractString = "sgd"
    maxiter::Int = 50
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
    get_generator(params::GeneratorParams, generator_type::Generic)

Instantiates a `GenericGenerator` with the given parameters.
"""
function get_generator(params::GeneratorParams, generator_type::Generic)
    return GenericGenerator(; opt=get_opt(params), λ=params.lambda_cost)
end

"""
    get_generator(params::GeneratorParams, generator_type::Gravitational)

Instantiates a `GravitationalGenerator` with the given parameters.
"""
function get_generator(params::GeneratorParams, generator_type::Gravitational)
    return GravitationalGenerator(;
        opt=get_opt(params), λ=[params.lambda_cost, params.lambda_energy]
    )
end

"""
    get_generator(params::GeneratorParams, generator_type::Omniscient)

Instantiates an `OmniscientGenerator` with the given parameters.
"""
function get_generator(params::GeneratorParams, generator_type::Omniscient)
    return OmniscientGenerator()
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

    ## Fields

    - `objective`: The objective function to use for training. Options correspond to the keys of the [`CTExperiments.objectives`](@ref) dictionary.
    - `lambda_class_loss`: The weight of the class loss in the objective function.
    - `lambda_energy_diff`: The weight of the energy difference term in the objective function.
    - `lambda_energy_reg`: The weight of the energy regularization term in the objective function.
    - `lambda_adversarial`: The weight of the adversarial loss in the objective function.
    - `class_loss`: The class loss to use for training. Options correspond to the keys of the [`CTExperiments.class_losses`](@ref) dictionary.
    - `burnin`: The fraction of the training epochs to use for warm-up. During warm-up, only standard classification loss is used for training.
    - `nepochs`: The number of epochs to train for.
    - `generator_params`: The parameters for the generator to use during training.
    - `nce`: The number of counterfactuals to generate per epoch and per batch of training data.
    - `nneighbours`: The number of neighbours in the target class to compare counterfactuals against.
    - `conv`: The convergence type to use for the counterfactual search.
    - `lr`: The learning rate to use for training.
    - `opt`: The optimizer to use for training.
    - `parallelizer`: The parallelization strategy to use for training.
    - `threaded`: Whether to also use threading for training if working with MPI.
    - `verbose`: The level of verbosity to use for training.
"""
Base.@kwdef struct TrainingParams <: AbstractConfiguration
    objective::AbstractString = "full"
    lambda_class_loss::AbstractFloat = 1.0
    lambda_energy_diff::AbstractFloat = CounterfactualTraining.default_energy_lambda[1]
    lambda_energy_reg::AbstractFloat = CounterfactualTraining.default_energy_lambda[2]
    lambda_adversarial::AbstractFloat = CounterfactualTraining.default_adversarial_lambda
    class_loss::AbstractString = "logitcrossentropy"
    burnin::AbstractFloat = get_global_param("burnin", 0.0f0)
    nepochs::Int = get_global_param("nepochs", 50)
    generator_params::GeneratorParams = GeneratorParams()
    nce::Int = get_global_param("nce", 100)
    nneighbours::Int = 100
    conv::AbstractString = "max_iter"
    lr::AbstractFloat = 0.001
    opt::AbstractString = "adam"
    parallelizer::AbstractString = ""
    threaded::Bool = true
    verbose::Int = get_global_param("verbose", 1)
end

"""
    objectives

Catalogue of available objective functions.
"""
const objectives = Dict(
    "vanilla" => CounterfactualTraining.VanillaObjective,
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

function get_lambdas(obj::CounterfactualTraining.VanillaObjective, params::TrainingParams)
    lambda = [params.lambda_class_loss]
    return lambda
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

function get_lambdas(
    obj::CounterfactualTraining.EnergyDifferentialObjective, params::TrainingParams
)
    lambda = [params.lambda_class_loss, params.lambda_energy_diff, params.lambda_energy_reg]
    return lambda
end

function get_lambdas(
    obj::CounterfactualTraining.AdversarialObjective, params::TrainingParams
)
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
        # Active comm:
        pllr = MPIParallelizer(MPI.COMM_WORLD; threaded=threaded)
    end

    if pllr_type == ""
        pllr = nothing
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
