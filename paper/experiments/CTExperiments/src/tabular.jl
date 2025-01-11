using CounterfactualExplanations
using CounterfactualExplanations.DataPreprocessing: train_test_split, unpack_data
using Flux
using Flux.MLUtils
using MultivariateStats
using Random
using StatsBase
using TaijaData

"""
    GMSC

Keyword container for the `GMSC` data set. Can specify the number of samples `n`, the batch size `batchsize`.
"""
Base.@kwdef struct GMSC <: Dataset
    n_train::Int = 12371
    batchsize::Int = 1024
    n_validation::Int = 1000
    train_test_ratio::Float32 = 0.8
    train_test_seed::Int = get_global_seed()
end

nmax(d::GMSC) = 16714

load_data(d::GMSC, n::Int) = load_gmsc(n)

"""
    CaliHousing

Keyword container for the `CaliHousing` (california housing) data set. Can specify the number of samples `n`, the batch size `batchsize`.
"""
Base.@kwdef struct CaliHousing <: Dataset
    n_train::Int = 15504
    batchsize::Int = 1024
    n_validation::Int = 1000
    train_test_ratio::Float32 = 0.8
    train_test_seed::Int = get_global_seed()
end

nmax(d::CaliHousing) = 20630

load_data(d::CaliHousing, n::Int) = load_california_housing(n)

"""
    Adult

Keyword container for the `Adult` data set.
"""
Base.@kwdef struct Adult <: Dataset
    n_train::Int = 25049
    batchsize::Int = 1024
    n_validation::Int = 1000
    train_test_ratio::Float32 = 0.8
    train_test_seed::Int = get_global_seed()
end

nmax(d::Adult) = 32561

load_data(d::Adult, n::Int) = load_uci_adult(n)