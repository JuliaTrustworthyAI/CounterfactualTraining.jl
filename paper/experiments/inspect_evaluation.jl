using CTExperiments
using CounterfactualExplanations
using CTExperiments.DataFrames
using DotEnv
using Logging
using PrettyTables

DotEnv.load!()

# Get config and set up grid:
eval_config = EvaluationConfig(
    joinpath(ENV["EXPERIMENT_DIR"], "run_evaluation_config.toml")
)

bmk = load_results(eval_config, "benchmark")