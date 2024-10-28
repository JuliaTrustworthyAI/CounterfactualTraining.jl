module CTExperiments

include("config.jl")

abstract type AbstractGeneratorType <: AbstractConfiguration end
abstract type AbstractGeneratorParams <: AbstractConfiguration end
abstract type Dataset <: AbstractConfiguration end
abstract type ModelType <: AbstractConfiguration end

include("utils.jl")
include("experiment.jl")

export Experiment, run_training

end