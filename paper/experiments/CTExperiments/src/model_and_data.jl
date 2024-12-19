using TaijaData
using MultivariateStats

get_domain(d::Dataset) = nothing

include("mnist.jl")
include("moons.jl")

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

"""
    get_data(data::Dataset; n::Union{Nothing,Int}=data.n_train, test_set::Bool=false)

Load dataset and return a subset of the data. If `n` is specified, it returns that number of samples, otherwise it returns all samples. If `test_set` is true, it loads the test set instead of the training set.
"""
function get_data(data::Dataset; n::Union{Nothing,Int}=data.n_train, test_set::Bool=false)
    X, y = get_data(data, test_set)
    n_total = size(X, 2)
    n = isnothing(n) ? n_total : n
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
