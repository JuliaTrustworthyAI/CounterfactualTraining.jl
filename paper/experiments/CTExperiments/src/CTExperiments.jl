module CTExperiments

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

export Experiment, run_training
export ExperimentGrid, setup_experiments
export save_results, load_results, has_results
export load_list
export EvaluationConfig
export test_performance, evaluate_counterfactuals
export to_toml

function generate_template(
    fname::String="paper/experiments/template_config.toml";
    experiment_name="template",
    overwrite=false,
    kwrgs...,
)
    if overwrite && isfile(fname)
        @warn "File $fname already exists! Overwriting..."
        rm(fname)
    end
    save_dir = joinpath(splitpath(fname)[1:(end - 1)])
    exper = Experiment(
        MetaParams(; experiment_name=experiment_name, save_dir=save_dir); kwrgs...
    )
    to_toml(exper, fname)
    return fname
end

function generate_grid_template(
    fname::String="paper/experiments/template_grid_config.toml"; overwrite=false
)
    if overwrite && isfile(fname)
        @warn "File $fname already exists! Overwriting..."
        rm(fname)
    end

    save_dir = joinpath(splitpath(fname)[1:(end - 1)])
    exper_grid = CTExperiments.ExperimentGrid(;save_dir=save_dir)
    to_toml(exper_grid, fname)
    return fname
end

export generate_template, generate_grid_template

end
