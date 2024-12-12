using CTExperiments
using CounterfactualExplanations
using DotEnv

DotEnv.load!()

# Run grid:
set_mpi_finalize(false)
include("run_grid.jl")

# Run evaluation:
set_mpi_finalize(true)
include("run_evaluation_grid.jl")
