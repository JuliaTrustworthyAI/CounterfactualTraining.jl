using CounterfactualExplanations
using CounterfactualExplanations.DataPreprocessing: train_test_split, unpack_data
using Flux
using Flux.MLUtils
using MultivariateStats
using StatsBase
using TaijaData

"""
    MNIST

Keyword container for the `MNIST` data set. Can specify the number of samples `n`, the batch size `batchsize`.
"""
Base.@kwdef struct MNIST <: Dataset
    n_train::Int = get_global_param("n_train", 10_000)
    batchsize::Int = 1000
    n_validation::Int = 1000
    train_test_ratio::Float32 = 0.8
    train_test_seed::Int = get_global_seed()
    mutability::Union{String,Vector{String}} = "none"
    domain::Union{String,Vector{<:Any}} = [-1.0f0, 1.0f0]
    datadir::String = get_global_dev_dir()
end

dname(d::MNIST) = "mnist"

"""
    get_data(data::MNIST, test_set::Bool=false)

Load the MNIST data set. If `test_set` is true, load the test set; otherwise, load the training set.
"""
function get_data(data::MNIST; n::Union{Nothing,Int}=nothing, test_set::Bool=false)

    # Get data:
    if test_set
        X, y = load_mnist_test()
    else
        X, y = load_mnist(Int(round(ntotal(data) * data.train_test_ratio)))
    end
    X = Float32.(X)

    # Subset:
    if !isnothing(n)
        X, y = take_subset(X, y, n; rng=get_rng(data))
    end

    return X, y
end
