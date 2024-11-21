using CTExperiments
using CTExperiments.DataFrames
using DotEnv

DotEnv.load!()

# Get config and set up grid:
eval_config = EvaluationConfig(
    joinpath(ENV["EXPERIMENT_DIR"], "evaluation_configs/2024-11-11.toml")
)
exper_grid = ExperimentGrid(eval_config.grid_file)
df_meta = CTExperiments.expand_grid_to_df(exper_grid)



