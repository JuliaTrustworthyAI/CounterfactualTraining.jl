using CTExperiments
using CTExperiments.CounterfactualExplanations
using DotEnv
using Random
using Serialization

DotEnv.load!()

Random.seed!(2025)

config_file = joinpath(ENV["EXPERIMENT_DIR"], "run_model_config.toml")
name = CTExperiments.from_toml(config_file)["meta_params"]["experiment_name"]
save_dir = joinpath(ENV["OUTPUT_DIR"], name)
experiment = Experiment(config_file; new_save_dir=save_dir)
@info "Running experiment: $(name)"
model, logs = run_training(experiment; checkpoint_dir=save_dir)
save_results(experiment, model, logs)
# M = MLP(model; likelihood=:classification_multi)
# fname = joinpath(save_dir, "$(name)_model.jls")
# logs_name = joinpath(save_dir, "$(name)_logs.jls")
# @info "Saving model to $(fname)"
# serialize(fname, M)
# @info "Saving logs to $(logs_name)"
# serialize(logs_name, logs)

