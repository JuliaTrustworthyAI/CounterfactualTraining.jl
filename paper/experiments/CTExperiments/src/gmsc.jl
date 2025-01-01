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
    n_train::Int = 16714
    batchsize::Int = 1000
    n_validation::Int = 1000
    train_test_ratio::Float32 = 0.8
    train_test_seed::Int = get_global_seed()
end

get_domain(d::GMSC) = nothing

function get_ce_data(data::GMSC, n_total::Int)
    X, y = load_gmsc(n_total)
    X = Float32.(X)
    return CounterfactualData(X, y)
end

"""
    get_data(data::GSMC, test_set::Bool=false)

Load the GMSC data set. The data is shuffled and split into training and test sets after setting the seed. If `test_set` is true, the function returns the test set.
"""
function get_data(data::GMSC, test_set::Bool=false)

    # Set seed and shuffle data:
    Random.seed!(data.train_test_seed)
    X, y = load_gmsc(nothing)
    X = Float32.(X)
    new_idx = randperm(size(X, 2))
    X = X[:, new_idx]
    y = y[new_idx]

    # Split data into training and test sets:
    ntrain = round(data.train_test_ratio * size(X, 2))
    if !test_set
        X = X[:, 1:ntrain]
        y = y[1:ntrain]
    else
        X = X[:, ntrain+1:end]
        y = y[ntrain+1:end]
    end

    return X, y
end
