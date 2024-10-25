
using Flux

abstract type Dataset end

abstract type ModelType end

include("model_and_data.jl")
include("training_params.jl")

mutable struct Experiment
    data::Dataset
    model_type::ModelType
    training_params::TrainingParams
end

function setup(exp::Experiment)
    exp = setup(exp.data, exp.model_type)
    set_input_encoder!(exp)
    return exp
end

function set_input_encoder!(exp::Experiment)
    exp.training_params.input_encoder = nothing
    return exp
end

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
        parallelizer=pllr,
        verbose=exp.training_params.verbose,
        convergence=exp.training_params.conv,
        nepochs=exp.training_params.nepochs,
        burnin=exp.training_params.burnin,
        nce=exp.training_params.nce,
        domain=exp.data.domain
    )
    return model, logs
end

