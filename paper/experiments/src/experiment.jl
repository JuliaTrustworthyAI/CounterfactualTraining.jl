using Flux

include("training_params.jl")
include("model_and_data.jl")

"""
    MetaParams

Mutable struct holding the meta parameters for the experiment.
"""
Base.@kwdef struct MetaParams
    dim_reduction::Bool = false
end

"""
    Experiment

A mutable struct that holds the data, model, training parameters, meta parameters and results of an experiment.
"""
mutable struct Experiment
    data::Dataset
    model_type::ModelType
    training_params::TrainingParams
    meta_params::MetaParams
end

"""
    setup(exp::Experiment) 

Sets up the experiment.
"""
function setup(exp::Experiment)

    # Setup the model and data:
    exp = setup(exp.data, exp.model_type)

    # Set up the input encoder for the given dataset, generator and meta parameters.
    set_input_encoder!(exp)

    return exp
end

"""
    set_input_encoder!(exp::Experiment)

Sets up the input encoder for the given experiment. This is dispatched over the dataset and generator type.
"""
function set_input_encoder!(exp::Experiment)
    return set_input_encoder!(
        exp,
        exp.data,
        exp.training_params.generator_params.type,
    )
end

"""
    set_input_encoder!(
        exp::Experiment,
        data::Dataset,
        generator_type::AbstractGeneratorType,
    )

Sets up the input encoder for the given experiment, dataset and generator type.
"""
function set_input_encoder!(
    exp::Experiment,
    data::Dataset,
    generator_type::AbstractGeneratorType,
)
    exp.training_params.input_encoder = nothing
    return exp
end

"""
    run_trainging(exp::Experiment)

Trains the model on the given dataset with Counterfactual Training using the given training parameters and meta parameters.
"""
function run_training(exp::Experiment)
    generator = get_generator(exp.training_params.generator_params)
    model, train_set = setup(exp)
    opt_state = Flux.setup(exp.training_params.training_opt, model)
    model, logs = counterfactual_training(
        loss,
        model,
        generator,
        train_set,
        opt_state;
        parallelizer=exp.training_params.parallelizer,
        verbose=exp.training_params.verbose,
        convergence=exp.training_params.conv,
        nepochs=exp.training_params.nepochs,
        burnin=exp.training_params.burnin,
        nce=exp.training_params.nce,
        domain=exp.data.domain
    )
    return model, logs
end

