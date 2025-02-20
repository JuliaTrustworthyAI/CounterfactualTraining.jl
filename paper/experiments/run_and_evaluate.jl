using CTExperiments
using CounterfactualExplanations
using DotEnv

DotEnv.load!()

# Run grid:
set_mpi_finalize(false)
include("run_grid_sequentially.jl")

# Run evaluation:
set_mpi_finalize(true)
include("run_evaluation_grid_sequentially.jl")

# Present:
include("present_results.jl")