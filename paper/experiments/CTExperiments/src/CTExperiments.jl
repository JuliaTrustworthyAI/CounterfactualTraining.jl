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

export Experiment, run_training

function generate_template(fname::String="paper/experiments/template_config.toml"; experiment_name="template", overwrite=true, kwrgs...)
    if overwrite && isfile(fname)
        @warn "File $fname already exists! Overwriting..."
        rm(fname)
    end
    exper = Experiment(MetaParams(; config_file=fname, experiment_name=experiment_name, kwrgs...))
    to_toml(exper)
    return fname
end

function generate_grid_template(
    fname::String="paper/experiments/grid_template_config.toml";
    overwrite=true,
    kwrgs...,
)
    if overwrite && isfile(fname)
        @warn "File $fname already exists! Overwriting..."
        rm(fname)
    end
   
    exper_grid = CTExperiments.ExperimentGrid(;
        data_params=Dict("batchsize" => [100, 1000], "n" => [10000, 30000])
    )
    to_toml(exper_grid)
    return fname
end

export generate_template, generate_grid_template

end