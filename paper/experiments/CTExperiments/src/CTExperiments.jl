module CTExperiments

using CounterfactualTraining

using CounterfactualExplanations
using CounterfactualExplanations.Evaluation
using CounterfactualExplanations.Objectives
using Logging
using Random
using TaijaData

abstract type AbstractConfiguration end
abstract type AbstractExperiment <: AbstractConfiguration end
abstract type AbstractGeneratorType <: AbstractConfiguration end
abstract type AbstractGeneratorParams <: AbstractConfiguration end
abstract type Dataset <: AbstractConfiguration end
abstract type ModelType <: AbstractConfiguration end

include("config.jl")
include("utils.jl")
include("omniscient.jl")
include("grid.jl")
include("experiment.jl")
include("evaluate.jl")
include("evaluation_grid.jl")
include("plotting.jl")

export Experiment, run_training
export make_dummy, remove_dummy!, isdummy
export ExperimentGrid, setup_experiments
export save_results, load_results, has_results
export load_list
export get_logs
export EvaluationConfig
export EvaluationGrid, setup_evaluations, ntasks
export test_performance, evaluate_counterfactuals
export generate_factual_target_pairs
export to_toml
export aggregate_logs, aggregate_ce_evaluation
export PlotParams, useful_byvars
export plot_errorbar_logs, boxplot_ce, plot_ce
export set_work_dir, get_work_dir, results_dir
export save_dir
export get_config_from_args
export mpi_should_finalize, set_mpi_finalize
export GMSC, MNIST, Moons, LinearlySeparable, Overlapping, Circles, CaliHousing, Adult
export get_data, get_ce_data, ntotal
export get_domain, get_mutability
export default_save_dir

"The default benchmarking measures."
const CE_MEASURES = [
    validity,
    plausibility_distance_from_target,
    plausibility_energy_differential,
    MMD(; compute_p=nothing),
    distance,
    redundancy,
]

end
