module CTExperiments

using CounterfactualTraining

using CounterfactualExplanations
using CounterfactualExplanations.Evaluation
using CounterfactualExplanations.Objectives
using Logging

abstract type AbstractConfiguration end
abstract type AbstractExperiment <: AbstractConfiguration end
abstract type AbstractGeneratorType <: AbstractConfiguration end
abstract type AbstractGeneratorParams <: AbstractConfiguration end
abstract type Dataset <: AbstractConfiguration end
abstract type ModelType <: AbstractConfiguration end

include("config.jl")
include("utils.jl")
include("grid.jl")
include("experiment.jl")
include("evaluate.jl")
include("plotting.jl")

export Experiment, run_training
export ExperimentGrid, setup_experiments
export save_results, load_results, has_results
export load_list
export get_logs
export EvaluationConfig
export test_performance, evaluate_counterfactuals
export to_toml
export aggregate_logs, aggregate_ce_evaluation
export plot_errorbar_logs, boxplot_ce

"The default benchmarking measures."
const CE_MEASURES = [
    validity, 
    plausibility_cosine,
    plausibility_distance_from_target,
    plausibility_energy_differential,
    distance,
    redundancy
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
        fname::String="paper/experiments/template_grid_config.toml"; overwrite=false
    )

Generates a template configuration file for experiment grids. This is useful for quickly setting up a new grid of experiments by copying the generated template into your project directory.
"""
function generate_grid_template(
    fname::String="paper/experiments/template_grid_config.toml"; overwrite=false
)
    write_file = !isfile(fname)     # don't write file if it exists
    if overwrite                    # unless specified
        @warn "File $fname already exists! Overwriting..."
        write_file = true
    end

    if write_file
        save_dir = joinpath(splitpath(fname)[1:(end - 1)])
        exper_grid = CTExperiments.ExperimentGrid(; save_dir=save_dir)
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

export generate_template, generate_grid_template, generate_eval_template

end
