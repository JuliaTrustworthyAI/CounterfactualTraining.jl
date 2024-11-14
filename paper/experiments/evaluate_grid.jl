using CTExperiments
using CTExperiments.CounterfactualExplanations
using DotEnv

DotEnv.load!()

# Get config and set up grid:
eval_config = joinpath(ENV["EXPERIMENT_DIR"], "eval_config.toml") |> EvaluationConfig
exper_grid = ExperimentGrid(eval_config.grid_file)
exper_list = load_list(exper_grid)      # get list for reference