using Accessors
import Base: ==
import CounterfactualTraining as CT
using DataFrames
using Flux
using JLD2
using Logging
using UUIDs

include("training_params.jl")
include("model_and_data.jl")

"""
    MetaParams

Struct holding the meta parameters for the experiment.
"""
Base.@kwdef mutable struct MetaParams <: AbstractConfiguration
    experiment_name::String = string(uuid1())
    data::String = "mnist"
    model_type::String = "mlp"
    generator_type::String = "ecco"
    dim_reduction::Bool = false
    save_dir::String = mkpath(joinpath(tempdir(), experiment_name))
end

config_file(params::MetaParams) = joinpath(params.save_dir, "config.toml")

output_dir(params::MetaParams) = mkpath(joinpath(params.save_dir, "output"))

"""
    ==(x::MetaParams, y::MetaParams)

Convenience function to check for equality of all fields.
"""
function ==(x::MetaParams, y::MetaParams)::Bool
    relevant_fields = filter((x,) -> x != :save_dir, fieldnames(CTExperiments.MetaParams))
    return all(getfield(x, field) == getfield(y, field) for field in relevant_fields)
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
    if isfile(config_file(meta_params))
        config_dict = from_toml(config_file(meta_params))
        @info "Experiment configuration loaded from $(config_file(meta_params)). Any parameters specified in the function call other than the meta params will be ignored."
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
    data = get_data_set(meta_params.data)(; data_params...)
    model_type = get_model_type(meta_params.model_type)(; model_params...)

    # Training parameters:
    generator_type = get_generator_type(meta_params.generator_type)
    generator_params = GeneratorParams(; type=generator_type(), generator_params...)
    training_params = TrainingParams(;
        generator_params=generator_params, training_params...
    )

    # Experiment:
    exper = Experiment(data, model_type, training_params, meta_params)
    to_toml(exper)

    return exper
end

function Experiment(fname::String; new_save_dir::Union{Nothing,String}=nothing)
    @assert isfile(fname) "Experiment file not found."
    meta = to_meta(from_toml(fname))
    if !isnothing(new_save_dir)
        mkpath(new_save_dir)
        meta.save_dir = new_save_dir
        cp(fname, config_file(meta); force=true)
    end
    @info "Experiment loaded from $(fname)."
    exper = Experiment(meta)
    to_toml(exper, fname)
    return exper
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
    return get_input_encoder(exp, exp.data, exp.training_params.generator_params.type)
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
    exp::AbstractExperiment, data::Dataset, generator_type::AbstractGeneratorType
)
    return nothing
end

"""
    run_training(exp::Experiment; checkpoint_dir::Union{Nothing,String} = nothing)

Trains the model on the given dataset with Counterfactual Training using the given training parameters and meta parameters.
"""
function run_training(exp::Experiment; checkpoint_dir::Union{Nothing,String}=nothing)

    # Counterfactual generator:
    generator = get_generator(exp.training_params.generator_params)
    model, train_set, input_encoder, val_set = setup(exp)
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
    model, logs = CounterfactualTraining.counterfactual_training(
        obj,
        model,
        generator,
        train_set,
        opt_state;
        val_set=val_set,
        parallelizer=pllr,
        verbose=exp.training_params.verbose,
        convergence=conv,
        nepochs=exp.training_params.nepochs,
        burnin=exp.training_params.burnin,
        nce=exp.training_params.nce,
        domain=domain,
        input_encoder=input_encoder,
        checkpoint_dir=checkpoint_dir,
    )

    return model, logs
end

"""
    results_name(exper::Experiment)

Default name for the results file.
"""
function results_name(exper::Experiment)
    return joinpath(exper.meta_params.save_dir, "results.jld2")
end

"""
    save_results(exper::Experiment, model, logs)

Saves the results of an experiment to a file.
"""
function save_results(exper::Experiment, model, logs)
    save_name = results_name(exper)
    M = MLP(model; likelihood=:classification_multi)
    @info "Saving model and logs to $(save_name):"
    return jldsave(save_name; model, logs, M)
end

"""
    load_results(exper::Experiment)

Loads the results of an experiment from a file.
"""
function load_results(exper::Experiment)
    @assert has_results(exper) "No results found for experiment."
    save_name = results_name(exper)
    @info "Loading results from $(save_name):"
    model, logs, M = JLD2.load(save_name, "model", "logs", "M")
    return model, logs, M
end

"""
    load_results(grid::ExperimentGrid)

Overloads the function to load the results from all experiments in a grid.
"""
function load_results(grid::ExperimentGrid)
    exper_list = load_list(grid)
    @info "Loading results from $(length(exper_list)) experiments:"
    results = Logging.with_logger(Logging.NullLogger()) do
        load_results.(exper_list)
    end
    return results
end

"""
    has_results(exper::Experiment)

Checks if the results of an experiment are available in a file.
"""
function has_results(exper::Experiment)
    save_name = results_name(exper)
    return isfile(save_name)
end

"""
    get_logs(exper::Experiment)

Retrieves the logs from disk and returns a DataFrame with additional columns for the experiment `:id` and the path to the config file (`:config_file`).
"""
function get_logs(exper::Experiment)
    _, logs, _ = Logging.with_logger(Logging.NullLogger()) do
        load_results(exper)
    end
    df_logs = DataFrame(logs)
    df_logs.epoch .= 1:nrow(df_logs)
    df_logs.id .= exper.meta_params.experiment_name
    df_logs.config_file .= config_file(exper.meta_params)
    select!(df_logs, :id, :epoch, Not(:id, :epoch))
    return df_logs
end

"""
    get_logs(grid::ExperimentGrid)

Overloads the function for a grid of experiments.
"""
function get_logs(grid::ExperimentGrid)
    exper_list = Logging.with_logger(Logging.NullLogger()) do
        load_list(grid)
    end
    return vcat(get_logs.(exper_list)...)
end
