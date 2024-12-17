using CTExperiments
using CounterfactualExplanations
using CounterfactualExplanations.Evaluation
using CTExperiments.DataFrames
using DotEnv

DotEnv.load!()

# Get config and set up grid:
eval_grid = EvaluationGrid(get_config_from_args())
eval_list = setup_evaluations(eval_grid)

for (i, eval_config) in enumerate(eval_list)
    # Collect benchmarks:
    CTExperiments.collect_benchmarks(eval_config)
end
