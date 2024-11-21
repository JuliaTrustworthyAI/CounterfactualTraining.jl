using CTExperiments
using CTExperiments.DataFrames
using DotEnv

DotEnv.load!()

# Get config and set up grid:
eval_config = EvaluationConfig(
    joinpath(ENV["EXPERIMENT_DIR"], "run_evaluation_config.toml")
)
exper_grid = ExperimentGrid(eval_config.grid_file)

# Meta data:
df_meta = CTExperiments.expand_grid_to_df(exper_grid)

