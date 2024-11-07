using CTExperiments
using CTExperiments.CounterfactualExplanations
using DotEnv
using Random
using Serialization

DotEnv.load!()

Random.seed!(2025)

config_file = joinpath(ENV["EXPERIMENT_DIR"], "run_model_config.toml")
experiment = Experiment(config_file)
name = experiment.meta_params.experiment_name
@info "Running experiment: $(name)"
checkpoint_dir = ENV["OUTPUT_DIR"]
model, logs = run_training(experiment; checkpoint_dir=checkpoint_dir)
M = MLP(model; likelihood=:classification_multi)
fname = joinpath(ENV["OUTPUT_DIR"], "$(name)_model.jls")
logs_name = joinpath(ENV["OUTPUT_DIR"], "$(name)_logs.jls")
@info "Saving model to $(fname)"
serialize(fname, M)
@info "Saving logs to $(logs_name)"
serialize(logs_name, logs)

