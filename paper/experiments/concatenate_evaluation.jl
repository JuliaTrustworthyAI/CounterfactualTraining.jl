using CTExperiments
using CounterfactualExplanations
using CounterfactualExplanations.Evaluation
using CTExperiments.DataFrames
using DotEnv

DotEnv.load!()

# Get config and set up grid:
eval_config = EvaluationConfig(
    joinpath(ENV["EXPERIMENT_DIR"], "evaluation_configs/2024-11-21_architecture.toml")
)

# Collect benchmarks:
bmk = CTExperiments.collect_benchmarks(eval_config)