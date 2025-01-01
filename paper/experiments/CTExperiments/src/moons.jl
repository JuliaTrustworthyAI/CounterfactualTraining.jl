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
    train_test_ratio::Float32 = 0.8
    train_test_seed::Int = get_global_seed()
end

get_domain(d::Moons) = nothing

load_data(d::Moons, n::Int) = load_moons(n)
