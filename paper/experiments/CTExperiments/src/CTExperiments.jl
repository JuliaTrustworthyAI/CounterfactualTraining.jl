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
export GMSC, MNIST, Moons
export get_data, get_ce_data

"The default benchmarking measures."
const CE_MEASURES = [
    validity,
    plausibility_cosine,
    plausibility_distance_from_target,
    plausibility_energy_differential,
    distance,
    redundancy,
]

"""
    generate_template(
        fname::String="paper/experiments/template_config.toml";
        experiment_name="template",
        overwrite=false,
        kwrgs...,
    )

Generates a template configuration file for experiments. This is useful for quickly setting up a new experiment by copying the generated template into your project directory.
"""
function generate_template(
    fname::String="paper/experiments/template_config.toml";
    experiment_name="template",
    overwrite=false,
    kwrgs...,
)
    write_file = !isfile(fname)     # don't write file if it exists
    if overwrite                    # unless specified
        @warn "File $fname already exists! Overwriting..."
        write_file = true
    end

    if write_file
        save_dir = joinpath(splitpath(fname)[1:(end - 1)])
        exper = Experiment(
            MetaParams(; experiment_name=experiment_name, save_dir=save_dir); kwrgs...
        )
        to_toml(exper, fname)
    else
        @warn "File already exists and not explicitly asked to overwrite it."
    end

    return fname
end

"""
    generate_grid_template(
        fname::String="paper/experiments/template_grid_config.toml"; overwrite=false, kwrgs...
    )

Generates a template configuration file for experiment grids. This is useful for quickly setting up a new grid of experiments by copying the generated template into your project directory.
"""
function generate_grid_template(
    fname::String="paper/experiments/template_grid_config.toml"; overwrite=false, kwrgs...
)
    write_file = !isfile(fname)     # don't write file if it exists
    if overwrite                    # unless specified
        @warn "File $fname already exists! Overwriting..."
        write_file = true
    end

    if write_file
        save_dir = joinpath(splitpath(fname)[1:(end - 1)])
        exper_grid = CTExperiments.ExperimentGrid(; save_dir=save_dir, kwrgs...)
        to_toml(exper_grid, fname)
    else
        @warn "File already exists and not explicitly asked to overwrite it."
    end

    return fname
end

"""
    generate_eval_template(
        fname::String="paper/experiments/template_eval_config.toml";
        overwrite=false,
        save_dir="paper/experiments/template_eval_dir",
    )

Generates a template configuration file for evaluation. This is useful for quickly setting up a new evaluation by copying the generated template into your project directory.
"""
function generate_eval_template(
    fname::String="paper/experiments/template_eval_config.toml";
    overwrite=false,
    save_dir="paper/experiments/template_eval_dir",
)
    write_file = !isfile(fname)     # don't write file if it exists
    if overwrite                    # unless specified
        @warn "File $fname already exists! Overwriting..."
        write_file = true
    end

    if write_file
        grid_file = Logging.with_logger(Logging.NullLogger()) do
            generate_grid_template()
        end
        exper_grid = CTExperiments.ExperimentGrid(grid_file)
        cfg = EvaluationConfig(exper_grid; grid_file=grid_file, save_dir=save_dir)
        to_toml(cfg, fname)
    else
        @warn "File already exists and not explicitly asked to overwrite it."
    end

    return fname
end

"""
    generate_eval_grid_template(
        fname::String="paper/experiments/template_eval_grid_config.toml";
        overwrite=false,
        save_dir="paper/experiments/template_eval_dir",
    )

Generates a template configuration file for evaluation grids. This is useful for quickly setting up a new evaluation grid by copying the generated template into your project directory.
"""
function generate_eval_grid_template(
    fname::String="paper/experiments/template_eval_grid_config.toml";
    overwrite=false,
    save_dir="paper/experiments/template_eval_dir",
)
    write_file = !isfile(fname)     # don't write file if it exists
    if overwrite                    # unless specified
        @warn "File $fname already exists! Overwriting..."
        write_file = true
    end

    if write_file
        grid_file = Logging.with_logger(Logging.NullLogger()) do
            generate_grid_template()
        end
        exper_grid = CTExperiments.ExperimentGrid(grid_file)
        cfg = EvaluationGrid(exper_grid; grid_file=grid_file, save_dir=save_dir)
        to_toml(cfg, fname)
    else
        @warn "File already exists and not explicitly asked to overwrite it."
    end

    return fname
end

export generate_template,
    generate_grid_template, generate_eval_template, generate_eval_grid_template

global _global_seed = 2025

function set_global_seed(seed::Int=_global_seed)
    global _global_seed = try
        parse(Int, ENV["GLOBAL_SEED"])
        @info "Found environment variable `ENV['GLOBAL_SEED']`. Setting global seed to it."
    catch
        seed
    end
    Random.seed!(_global_seed)
    @info "Global seed set to $_global_seed"
end

get_global_seed() = _global_seed

export set_global_seed, get_global_seed

end
