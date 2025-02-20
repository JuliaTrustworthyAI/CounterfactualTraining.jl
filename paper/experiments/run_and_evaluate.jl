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
global CTExperiments._colvar = "generator_type"
global CTExperiments._colorvar = "objective"
global CTExperiments._byvars_ce = "objective"
global CTExperiments._rowvar_ce = "lambda_energy_eval"
include("present_results.jl")