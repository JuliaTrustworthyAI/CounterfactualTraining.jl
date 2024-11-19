using CTExperiments
using CounterfactualExplanations
using CounterfactualExplanations.Evaluation
using CTExperiments.DataFrames
using DotEnv

DotEnv.load!()

# Get config and set up grid:
eval_config = EvaluationConfig(
    joinpath(ENV["EXPERIMENT_DIR"], "run_evaluation_config.toml")
)

# Collect benchmarks:
bmk = CTExperiments.collect_benchmarks(eval_config)