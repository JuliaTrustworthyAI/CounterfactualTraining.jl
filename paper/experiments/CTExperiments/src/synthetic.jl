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
Base.@kwdef mutable struct Moons <: Dataset
    n_train::Int = 3000
    batchsize::Int = 30
    n_validation::Int = 600
    train_test_ratio::Float32 = 0.8
    train_test_seed::Int = get_global_seed()
    mutability::Union{String,Vector{String}} = "none"
    domain::Union{String,Vector{<:Any}} = "none"
    datadir::String = get_global_dev_dir()
end

dname(d::Moons) = "moons"

load_data(d::Moons, n::Int; seed=TaijaData.data_seed) = load_moons(n; seed=seed)

"""
    LinearlySeparable

Keyword container for the `LinearlySeparable` data set. Can specify the number of samples `n`, the batch size `batchsize`.
"""
Base.@kwdef mutable struct LinearlySeparable <: Dataset
    n_train::Int = 3000
    batchsize::Int = 30
    n_validation::Int = 600
    train_test_ratio::Float32 = 0.8
    train_test_seed::Int = get_global_seed()
    mutability::Union{String,Vector{String}} = "none"
    domain::Union{String,Vector{<:Any}} = "none"
    datadir::String = get_global_dev_dir()
end

dname(d::LinearlySeparable) = "lin_sep"

function load_data(d::LinearlySeparable, n::Int; seed=TaijaData.data_seed)
    return load_linearly_separable(n; seed=seed)
end

"""
    Overlapping

Keyword container for the `Overlapping` data set. Can specify the number of samples `n`, the batch size `batchsize`.
"""
Base.@kwdef mutable struct Overlapping <: Dataset
    n_train::Int = 3000
    batchsize::Int = 30
    n_validation::Int = 600
    train_test_ratio::Float32 = 0.8
    train_test_seed::Int = get_global_seed()
    mutability::Union{String,Vector{String}} = "none"
    domain::Union{String,Vector{<:Any}} = "none"
    datadir::String = get_global_dev_dir()
end

dname(d::Overlapping) = "over"

function load_data(d::Overlapping, n::Int; seed=TaijaData.data_seed)
    return load_overlapping(n; seed=seed)
end

"""
    Circles

Keyword container for the `Circles` data set. Can specify the number of samples `n`, the batch size `batchsize`.
"""
Base.@kwdef mutable struct Circles <: Dataset
    n_train::Int = 3000
    batchsize::Int = 30
    n_validation::Int = 600
    train_test_ratio::Float32 = 0.8
    train_test_seed::Int = get_global_seed()
    mutability::Union{String,Vector{String}} = "none"
    domain::Union{String,Vector{<:Any}} = "none"
    datadir::String = get_global_dev_dir()
end

dname(d::Circles) = "circles"

function load_data(d::Circles, n::Int; seed=TaijaData.data_seed)
    return load_circles(n; seed=seed)
end

get_data_name(str::String; pretty=true) = get_name(data_sets[str](); pretty)

get_name(d::Moons; pretty::Bool=false) = pretty ? "Moons" : "moons"
function get_name(d::LinearlySeparable; pretty::Bool=false)
    return pretty ? "Linearly Separable" : "lin_sep"
end
get_name(d::Circles; pretty::Bool=false) = pretty ? "Circles" : "circles"
get_name(d::Overlapping; pretty::Bool=false) = pretty ? "Overlapping" : "over"
