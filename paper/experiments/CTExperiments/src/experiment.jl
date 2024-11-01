using Accessors
import CounterfactualTraining as CT
using Flux
using UUIDs

include("training_params.jl")
include("model_and_data.jl")

"""
    MetaParams

Struct holding the meta parameters for the experiment.
"""
Base.@kwdef struct MetaParams <: AbstractConfiguration
    experiment_name::String = string(uuid1())
    data::String = "mnist"
    model_type::String = "mlp"
    generator_type::String = "ecco"
    dim_reduction::Bool = false
    config_file::String = joinpath(tempdir(), "experiment_config_$experiment_name.toml")
end

"""
    Experiment

A mutable struct that holds the data, model, training parameters, meta parameters and results of an experiment.
"""
mutable struct Experiment <: AbstractExperiment
    data::Dataset
    model_type::ModelType
    training_params::TrainingParams
    meta_params::MetaParams
end

"""
    Experiment(meta_params::MetaParams)

Sets up the experiment for the provided meta data. Keyword arguments for [`Dataset`](@ref), [`ModelType`](@ref), [`TrainingParams``](@ref) and [`GeneratorParams`](@ref) can be passed as named tuples to `data_params`, `model_params`, `training_params` and `generator_params`, respectively.
"""
function Experiment(
    meta_params::MetaParams=MetaParams();
    data_params::NamedTuple=(;),
    model_params::NamedTuple=(;),
    training_params::NamedTuple=(;),
    generator_params::NamedTuple=(;),
)

    # Load the configuration file and set up the experiment:
    if isfile(meta_params.config_file)
        config_dict = from_toml(meta_params.config_file)
        @info "Experiment configuration loaded from $(meta_params.config_file). Any parameters specified in the function call other than the meta params will be ignored."
        @assert meta_params == to_meta(config_dict) "Meta parameters do not match the configuration file."
        # Generate keyword containers from config file:
        data_params = to_ntuple(config_dict["data"])
        model_params = to_ntuple(config_dict["model_type"])
        _training_params = to_ntuple(config_dict["training_params"])
        _generator_params = _training_params.generator_params
        training_params = @delete $_training_params.generator_params
        generator_params = @delete $_generator_params.type
    end

    # Model and data:
    data = get_data(meta_params.data)(;data_params...)
    model_type = get_model_type(meta_params.model_type)(;model_params...)

    # Training parameters:
    generator_type = get_generator_type(meta_params.generator_type)
    generator_params = GeneratorParams(; type=generator_type(), generator_params...)
    training_params = TrainingParams(; generator_params=generator_params, training_params...)

    # Experiment:
    exper = Experiment(
        data,
        model_type,
        training_params,
        meta_params
    )

    return exper
end

function Experiment(fname::String)
    @assert isfile(fname) "Experiment file not found."
    meta = from_toml(fname) |> to_meta
    @assert meta.config_file == fname "Specified file name does not match the file name specified in the configuration file. Did you accidentally overwrite that parameter in the TOML file?"
    @info "Experiment loaded from $(fname)."
    return Experiment(meta)
end

"""
    setup(exp::Experiment) 

Sets up the experiment.
"""
setup(exp::AbstractExperiment) = setup(exp, exp.data, exp.model_type)

"""
    get_input_encoder(exp::Experiment)

Sets up the input encoder for the given experiment. This is dispatched over the dataset and generator type.
"""
function get_input_encoder(exp::AbstractExperiment)
    return get_input_encoder(
        exp,
        exp.data,
        exp.training_params.generator_params.type,
    )
end

"""
    get_input_encoder(
        exp::Experiment,
        data::Dataset,
        generator_type::AbstractGeneratorType,
    )

Sets up the input encoder for the given experiment, dataset and generator type.
"""
function get_input_encoder(
    exp::AbstractExperiment,
    data::Dataset,
    generator_type::AbstractGeneratorType,
)
    return nothing
end

"""
    run_trainging(exp::Experiment)

Trains the model on the given dataset with Counterfactual Training using the given training parameters and meta parameters.
"""
function run_training(exp::Experiment)

    # Counterfactual generator:
    generator = get_generator(exp.training_params.generator_params)
    model, train_set, input_encoder = setup(exp)
    conv = get_convergence(exp.training_params)
    domain = get_domain(exp.data)
    pllr = get_parallelizer(exp.training_params)

    # Optimizer and model:
    training_opt = get_opt(exp.training_params)
    opt_state = Flux.setup(training_opt, model)

    # Get objective:
    class_loss = get_class_loss(exp.training_params.class_loss)         # get classification loss function
    obj = get_objective(exp.training_params.objective)                  # get objective type
    obj = obj(class_loss, get_lambdas(obj(), exp.training_params))      # instantiate objective

    # Train:
    model, logs = CT.counterfactual_training(
        obj,
        model,
        generator,
        train_set,
        opt_state;
        parallelizer=pllr,
        verbose=exp.training_params.verbose,
        convergence=conv,
        nepochs=exp.training_params.nepochs,
        burnin=exp.training_params.burnin,
        nce=exp.training_params.nce,
        domain=domain,
        input_encoder=input_encoder,
    )

    return model, logs
end

