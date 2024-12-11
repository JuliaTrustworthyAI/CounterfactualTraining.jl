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
    n_train::Int = 3000
    batchsize::Int = 1
    n_validation::Int = 600
end

get_domain(d::Moons) = nothing

function get_ce_data(data::Moons, n_total::Int)
    return CounterfactualData(load_moons(n_total)...)
end

function get_input_encoder(
    exp::AbstractExperiment, data::Moons, generator_type::AbstractGeneratorType
)
    return nothing
end

"""
    get_data(data::Moons, test_set::Bool=false)

Load the Moons data set. Since data is synthetically generated, `test_set` has no effect.
"""
get_data(data::Moons, test_set::Bool=false) = load_moons()
