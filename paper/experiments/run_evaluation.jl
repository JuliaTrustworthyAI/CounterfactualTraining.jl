using CTExperiments
using CTExperiments.CounterfactualExplanations
using DotEnv

DotEnv.load!()

# Get config and set up grid:
eval_config = EvaluationConfig(joinpath(ENV["EXPERIMENT_DIR"], "run_evaluation_config.toml"))
exper_grid = ExperimentGrid(eval_config.grid_file)
exper_list = load_list(exper_grid)      # get list for reference
