using CTExperiments
using CounterfactualExplanations
using DotEnv
using Random
using Serialization

DotEnv.load!()

Random.seed!(2025)

config_file = joinpath(ENV["EXPERIMENT_DIR"], "run_model_config.toml")
_name = CTExperiments.from_toml(config_file)["meta_params"]["experiment_name"]
save_dir = joinpath(ENV["OUTPUT_DIR"], _name)
experiment = Experiment(config_file; new_save_dir=save_dir)
@info "Running experiment: $(_name)"
model, logs = run_training(experiment; checkpoint_dir=save_dir)
save_results(experiment, model, logs)
