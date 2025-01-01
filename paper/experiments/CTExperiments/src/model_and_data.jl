using TaijaData
using MultivariateStats

get_domain(d::Dataset) = nothing

include("mnist.jl")
include("moons.jl")
include("gmsc.jl")

"""
    data_sets

Catalogue of available model types.
"""
const data_sets = Dict("mnist" => MNIST, "moons" => Moons)

"""
    get_data_set(s::String)

Retrieves the data set from the catalogue if available.
"""
function get_data_set(s::String)
    s = lowercase(s)
    @assert s in keys(data_sets) "Unknown data set: $s. Available sets are $(keys(data_sets))"
    return data_sets[s]
end

function get_data(data::Dataset; n::Union{Nothing,Int}=nothing, test_set::Bool=false)

    X, y = load_data(data, ntotal(data))    # load all available data

    # Set seed and shuffle data:
    Random.seed!(data.train_test_seed)
    X = Float32.(X)
    new_idx = randperm(size(X, 2))
    X = X[:, new_idx]
    y = y[new_idx]

    # Split data into training and test sets:
    ntrain = Int(round(data.train_test_ratio * size(X, 2)))
    if !test_set
        X = X[:, 1:ntrain]
        y = y[1:ntrain]
    else
        X = X[:, (ntrain + 1):end]
        y = y[(ntrain + 1):end]
    end

    # Subset:
    if !isnothing(n)
        X, y = take_subset(X, y, n)
    end

    return X, y
end

function take_subset(X, y, n)
    if n_total > n
        idx = sample(1:n_total, n; replace=false)

    elseif n_total < n
        idx = rand(1:n_total, n)
    else
        idx = 1:n_total
    end
    X = Float32.(X[:, idx])
    y = y[idx]

    return X, y
end

function get_ce_data(data::Dataset; test_set::Bool=false, train_only::Bool=false)
    data = CounterfactualData(get_data(data; test_set=test_set)...; domain=get_domain(data))
    if train_only
        _, _, data = train_val_split(data, data, data.n_validation / ntotal(data))
    end
    return data
end

ntotal(data::Dataset) = Int(round((data.n_train + data.n_validation) / data.train_test_ratio))

include("mlp.jl")
include("cnn.jl")

"""
    model_types

Catalogue of available model types.
"""
const model_types = Dict("mlp" => MLPModel, "lenet" => LeNetModel)

"""
    get_model_type(s::String)

Retrieves the model type from the catalogue if available.
"""
function get_model_type(s::String)
    s = lowercase(s)
    @assert s in keys(model_types) "Unknown model type: $s. Available types are $(keys(model_types))"
    return model_types[s]
end
