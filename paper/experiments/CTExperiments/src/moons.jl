using CounterfactualExplanations
using CounterfactualExplanations.DataPreprocessing: train_test_split, unpack_data
using Flux
using Flux.MLUtils
using MultivariateStats
using StatsBase
using TaijaData

"""
    Moons

Keyword container for the `Moons` data set. Can specify the number of samples `n`, the batch size `batchsize`.
"""
Base.@kwdef struct Moons <: Dataset
    n_train::Int = 1000
    batchsize::Int = 1
    n_validation::Int = 100
end

get_domain(d::Moons) = nothing

function get_ce_data(data::Moons, n_total::Int)
    return CounterfactualData(load_moons(n_total)...)
end

get_input_encoder(
    exp::AbstractExperiment, data::Moons, generator_type::AbstractGeneratorType
) = nothing