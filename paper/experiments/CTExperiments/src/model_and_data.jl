using MultivariateStats
using Random
using TaijaData

get_domain(d::Dataset) = nothing

include("mnist.jl")
include("synthetic.jl")
include("tabular.jl")

function get_rng(d::Dataset)
    return Xoshiro(d.train_test_seed)
end

"""
    data_sets

Catalogue of available model types.
"""
const data_sets = Dict(
    "lin_sep" => LinearlySeparable,
    "gmsc" => GMSC,
    "mnist" => MNIST,
    "moons" => Moons,
    "over" => Overlapping,
)

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
    X = Float32.(X)
    new_idx = randperm(get_rng(data), size(X, 2))
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
        X, y = take_subset(X, y, n; rng=get_rng(data))
    end

    return X, y
end

function take_subset(X, y, n; rng::AbstractRNG=Random.default_rng())
    n_total = size(X, 2)
    if n_total > n
        idx = sample(rng, 1:n_total, n; replace=false)
    elseif n_total < n
        idx = rand(rng, 1:n_total, n)
    else
        idx = 1:n_total
    end
    X = Float32.(X[:, idx])
    y = y[idx]

    return X, y
end

function get_ce_data(
    data::Dataset, n=nothing; test_set::Bool=false, train_only::Bool=false
)
    ce_data = CounterfactualData(get_data(data; n=n, test_set=test_set)...; domain=get_domain(data))
    if train_only
        _, _, ce_data = train_val_split(data, ce_data, data.n_validation / ntotal(data))
    end
    return ce_data
end

ntotal(data::Dataset) = Int(round((data.n_train + data.n_validation) / data.train_test_ratio))

include("linear.jl")
include("mlp.jl")
include("cnn.jl")

"""
    model_types

Catalogue of available model types.
"""
const model_types = Dict("linear" => LinearModel, "mlp" => MLPModel, "lenet" => LeNetModel)

"""
    get_model_type(s::String)

Retrieves the model type from the catalogue if available.
"""
function get_model_type(s::String)
    s = lowercase(s)
    @assert s in keys(model_types) "Unknown model type: $s. Available types are $(keys(model_types))"
    return model_types[s]
end
