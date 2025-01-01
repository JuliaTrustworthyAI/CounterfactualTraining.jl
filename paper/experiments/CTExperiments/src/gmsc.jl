using CounterfactualExplanations
using CounterfactualExplanations.DataPreprocessing: train_test_split, unpack_data
using Flux
using Flux.MLUtils
using MultivariateStats
using Random
using StatsBase
using TaijaData

"""
    MNIST

Keyword container for the `MNIST` data set. Can specify the number of samples `n`, the batch size `batchsize`.
"""
Base.@kwdef struct GMSC <: Dataset
    n_train::Int = 15714
    batchsize::Int = 1000
    n_validation::Int = 1000
    train_test_ratio::Float32 = 0.8
    train_test_seed::Int = get_global_seed()
end

get_domain(d::GMSC) = nothing

load_data(d::GMSC, n::Int) = load_gmsc(n)