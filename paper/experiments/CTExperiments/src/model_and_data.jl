using CounterfactualTraining
using MultivariateStats
using Random
using Serialization
using TaijaData

"""
    apply_inferred_domain!(d::Dataset)

Applies the domain constraints that would outherwise be inferred by `CounterfactualTraining`. This is to ensure that the same domain constraints are applied during training and evaluation.
"""
function apply_inferred_domain!(d::Dataset)
    if d.domain == "none"
        d.domain = get_data(d)[1] |> CounterfactualTraining.infer_domain_constraints
    end
    return d
end

nmax(d::Dataset) = Inf

exceeds_max(d::Dataset) = ntotal(d) > nmax(d)

include("mnist.jl")
include("synthetic.jl")
include("tabular.jl")

function get_rng(d::Dataset)
    return Xoshiro(d.train_test_seed)
end

function get_ce_measures(d::Dataset) end

"""
    data_sets

Catalogue of available model types.
"""
const data_sets = Dict(
    dname(LinearlySeparable()) => LinearlySeparable,
    dname(GMSC()) => GMSC,
    dname(MNIST()) => MNIST,
    dname(Moons()) => Moons,
    dname(Overlapping()) => Overlapping,
    dname(CaliHousing()) => CaliHousing,
    dname(Adult()) => Adult,
    dname(Circles()) => Circles,
    dname(Overlapping()) => Overlapping,
)

"""
    get_data_set(s::String)

Retrieves the data set from the catalogue if available.
"""
function get_data_set(s::String)
    s = lowercase(s)
    if s == ""
        @info "Dataset not specified. Using 'lin_sep'."
        s = "lin_sep"
    end
    @assert s in keys(data_sets) "Unknown data set: $s. Available sets are $(keys(data_sets))"
    return data_sets[s]
end

"""
    get_data(data::Dataset; n::Union{Nothing,Int}=nothing, test_set::Bool=false)

Loads the dataset `data`. By default, a total of [`ntotal(data)`] samples will be loaded. The output of [`ntotal`](@ref) depends on the parameters of the dataset. The keyword argument `n` can be specified to load only a subset of the dataset. 
"""
function get_data(data::Dataset; n::Union{Nothing,Int}=nothing, test_set::Bool=false)
    if exceeds_max(data)
        @warn "Requesting more data than available (using oversampling)."
    end
    navailable = if isinf(nmax(data))
        100_000
    else
        nmax(data)
    end
    X, y = load_data(data, navailable)  # load all available data

    # Set seed and shuffle data:
    X = Float32.(X)
    new_idx = randperm(get_rng(data), size(X, 2))
    X = X[:, new_idx]
    y = y[new_idx]

    # Split data into training and test sets:
    ntrain = data.n_train + data.n_validation
    ntest = ntotal(data) - ntrain
    @assert navailable >= ntrain + ntest
    if !test_set
        X = X[:, 1:ntrain]
        y = y[1:ntrain]
    else
        X = X[:, (end - ntest + 1):end]
        y = y[(end - ntest + 1):end]
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

function get_ce_data(data::Dataset, n=nothing; test_set::Bool=false, train_only::Bool=false)
    ce_data = CounterfactualData(
        get_data(data; n=n, test_set=test_set)...;
        domain=get_domain(data),
        mutability=get_mutability(data),
    )
    if train_only
        _, _, ce_data = train_val_split(data, ce_data, data.n_validation / ntotal(data))
    end
    return ce_data
end

function ntotal(data::Dataset)
    return Int(round((data.n_train + data.n_validation) / data.train_test_ratio))
end

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
    if s == ""
        @info "Model type not specified. Using 'mlp'."
        s = "mlp"
    end
    @assert s in keys(model_types) "Unknown model type: $s. Available types are $(keys(model_types))"
    return model_types[s]
end

"""
    input_dim(data::Dataset)

Helper function to get the dimension of the input data.
"""
input_dim(data::Dataset) = size(get_data(data; n=1)[1], 1)

"""
    get_mutability(data::Dataset)

Helper function to get the mutability constraints for the dataset. If `data.mutability` is a string, it converts it to a vector of symbols. If it's a vector of strings, it converts each string to a symbol.
"""
function get_mutability(data::Dataset)
    mtblty = data.mutability
    if mtblty isa String
        if mtblty == "none"
            mtblty = nothing
        else
            mtblty = fill(Symbol(mtblty), input_dim(data))
        end
    else
        mtblty = Symbol.(mtblty)
    end
    return mtblty
end

"""
    get_domain(data::Dataset)

Helper function to get the domain constraints for the dataset. If `data.domain` is a string other than "none", it throws an error. If it's a vector of two elements, it converts them to a tuple.
"""
function get_domain(d::Dataset)
    if d.domain isa String
        if d.domain == "none"
            domain = nothing
        else
            throw(ArgumentError("Domain must be a vector or 'none'."))
        end
    elseif typeof(d.domain) <: Vector{<:Real}
        domain = tuple(d.domain...)
    elseif typeof(d.domain) <: Vector{<:Tuple}
        domain = d.domain
    end
    return domain
end

"""
    load_vae(d::Dataset) 

Loads pre-trained VAE from the dataset directory. The file name is constructed using the dataset name.
"""
function load_vae(d::Dataset)
    return Serialization.deserialize(joinpath(d.datadir, "vae", "$(dname(d)).jls"))
end
