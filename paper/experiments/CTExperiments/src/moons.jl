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
    batchsize::Int = 30
    n_validation::Int = 600
end

get_domain(d::Moons) = nothing

function get_ce_data(data::Moons, n_total::Int)
    X, y = load_moons(n_total)
    X = Float32.(X)
    return CounterfactualData(X, y)
end

"""
    get_data(data::Moons, test_set::Bool=false)

Load the Moons data set. Since data is synthetically generated, `test_set` has no effect.
"""
function get_data(data::Moons, test_set::Bool=false)
    X, y = load_moons()
    X = Float32.(X)
    return X, y
end
