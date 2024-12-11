using CTExperiments
using CounterfactualExplanations
using CounterfactualExplanations.Evaluation
using CTExperiments.DataFrames
using DotEnv

DotEnv.load!()

# Get config and set up grid:
eval_config = EvaluationConfig(get_config_from_args())

# Collect benchmarks:
bmk = CTExperiments.collect_benchmarks(eval_config)
