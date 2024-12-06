using CTExperiments
using CounterfactualExplanations
using DotEnv

DotEnv.load!()

# Run grid:
include("run_grid.jl")

# Run evaluation:
include("run_evaluation_grid.jl")