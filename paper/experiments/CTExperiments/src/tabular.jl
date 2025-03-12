using CounterfactualExplanations
using CounterfactualExplanations.DataPreprocessing: train_test_split, unpack_data
using Flux
using Flux.MLUtils
using MultivariateStats
using Random
using StatsBase
using TaijaData

abstract type TabularData <: Dataset end

"""
    get_data(data::TabularData, test_set::Bool=false)


"""
function get_data(data::TabularData; n::Union{Nothing,Int}=nothing, test_set::Bool=false)

    if exceeds_max(data)
        @warn "Requesting more data than available (using oversampling)."
    end
    navailable = if isinf(nmax(data))
        100_000
    else
        nmax(data)
    end
    X, y, Xtest, ytest = load_data(data, navailable)    # load all available data

    # Get data:
    if test_set
        X, y = Xtest, ytest
    end
    X = Float32.(X)

    # Subset:
    if !isnothing(n)
        X, y = take_subset(X, y, n; rng=get_rng(data))
    end

    return X, y
end

"""
    GMSC

Keyword container for the `GMSC` data set. Can specify the number of samples `n`, the batch size `batchsize`.
"""
Base.@kwdef mutable struct GMSC <: TabularData
    n_train::Int = 12371
    batchsize::Int = 1000
    n_validation::Int = 1000
    train_test_ratio::Float32 = 0.8
    train_test_seed::Int = get_global_seed()
    mutability::Union{String,Vector{String}} = "none"
    domain::Union{String,Vector{<:Any}} = "none"
    datadir::String = get_global_dev_dir()
end

dname(d::GMSC) = "gmsc"

get_name(d::GMSC; pretty::Bool=false) = pretty ? "GMSC" : dname(d)

nmax(d::GMSC) = 16714

function load_data(d::GMSC, n::Int)
    data = Logging.with_logger(Logging.NullLogger()) do
        load_gmsc(n; train_test_split=d.train_test_ratio)
    end
    return data
end

"""
    CaliHousing

Keyword container for the `CaliHousing` (california housing) data set. Can specify the number of samples `n`, the batch size `batchsize`.
"""
Base.@kwdef mutable struct CaliHousing <: TabularData
    n_train::Int = 15504
    batchsize::Int = 1000
    n_validation::Int = 1000
    train_test_ratio::Float32 = 0.8
    train_test_seed::Int = get_global_seed()
    mutability::Union{String,Vector{String}} = "none"
    domain::Union{String,Vector{<:Any}} = "none"
    datadir::String = get_global_dev_dir()
end

dname(d::CaliHousing) = "cali"

get_name(d::CaliHousing; pretty::Bool=false) = pretty ? "California Housing" : dname(d)

nmax(d::CaliHousing) = 20630

function load_data(d::CaliHousing, n::Int)
    data = Logging.with_logger(Logging.NullLogger()) do
        load_california_housing(n; train_test_split=d.train_test_ratio)
    end
    return data
end

"""
    Adult

Keyword container for the `Adult` data set.
"""
Base.@kwdef mutable struct Adult <: TabularData
    n_train::Int = 25049
    batchsize::Int = 1000
    n_validation::Int = 1000
    train_test_ratio::Float32 = 0.8
    train_test_seed::Int = get_global_seed()
    mutability::Union{String,Vector{String}} = "none"
    domain::Union{String,Vector{<:Any}} = "none"
    datadir::String = get_global_dev_dir()
end

dname(d::Adult) = "adult"

get_name(d::Adult; pretty::Bool=false) = pretty ? "Adult" : dname(d)

nmax(d::Adult) = 32561

function load_data(d::Adult, n::Int)
    data = Logging.with_logger(Logging.NullLogger()) do
        load_uci_adult(n; train_test_split=d.train_test_ratio)
    end
    return data
end

function get_cats(d::Adult)
    cats = Logging.with_logger(Logging.NullLogger()) do
        load_uci_adult(; return_cats=true)[3]
    end
    return cats
end

"""
    Credit Default

Keyword container for the `Credit Default` data set.
"""
Base.@kwdef mutable struct Credit <: TabularData
    n_train::Int = 9617
    batchsize::Int = 1000
    n_validation::Int = 1000
    train_test_ratio::Float32 = 0.8
    train_test_seed::Int = get_global_seed()
    mutability::Union{String,Vector{String}} = "none"
    domain::Union{String,Vector{<:Any}} = "none"
    datadir::String = get_global_dev_dir()
end

dname(d::Credit) = "credit"

get_name(d::Credit; pretty::Bool=false) = pretty ? "Credit" : dname(d)

nmax(d::Credit) = 13272

function load_data(d::Credit, n::Int) 
    data = Logging.with_logger(Logging.NullLogger()) do 
        load_credit_default(n; train_test_split=d.train_test_ratio)
    end
    return data
end

function get_cats(d::Credit)
    cats = Logging.with_logger(Logging.NullLogger()) do 
        load_credit_default(; return_cats=true)[3] 
    end
    return cats
end
