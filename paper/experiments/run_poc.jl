using CTExperiments
using CounterfactualExplanations
using DotEnv

DotEnv.load!()

# Run grid:
ENV["config"] = joinpath(ENV["EXPERIMENT_DIR"], "poc.toml")
include("run_grid.jl")

# Run evaluation:
ENV["config"] = joinpath(ENV["EXPERIMENT_DIR"], "poc_evaluation_grid_config.toml")
include("run_evaluation_grid.jl")