using CTExperiments
using CTExperiments.CounterfactualExplanations
using DotEnv
using Serialization

DotEnv.load!()

config_file = joinpath(ENV["EXPERIMENT_DIR"], "template_config.toml")
experiment = Experiment(config_file)
name = experiment.meta_params.experiment_name
@info "Running experiment: $(name)"
model, logs = run_training(experiment)
M = MLP(model; likelihood=:classification_multi)
fname = joinpath(ENV["OUTPUT_DIR"], "model_$(name).jls")
@info "Saving model to $(fname)"
serialize(fname, M)
