module CTExperiments

abstract type AbstractGeneratorType end
abstract type AbstractGeneratorParams end
abstract type AbstractExperiment end
abstract type Dataset end
abstract type ModelType end

include("utils.jl")
include("experiment.jl")

export Experiment, run_training

end