using CTExperiments
using CTExperiments.CounterfactualExplanations
using DotEnv

DotEnv.load!()

# Get config and set up grid:
config_file = joinpath(ENV["EXPERIMENT_DIR"], "run_grid_config.toml")
exper_grid = ExperimentGrid(config_file)