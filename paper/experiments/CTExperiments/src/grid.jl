using Base.Iterators

"""
    ExperimentGrid

A keyword dictionary that contains the parameters for experiments. It is used to generate a list of [`MetaParams`](@ref) objects, one for each unique combination of the fields (see [`setup_experiments`](@ref)).
"""
Base.@kwdef struct ExperimentGrid <: AbstractConfiguration
    data::Vector{<:AbstractString} = ["mnist"]
    model_type::Vector{<:AbstractString} = ["mlp"]
    generator_type::Vector{<:AbstractString} = ["ecco", "generic", "revise"]
    dim_reduction::Vector{<:Bool} = [false]
    data_params::Dict{String,Any} = Dict()
    model_params::Dict{String,Any} = Dict()
    training_params::Dict{String,Any} = Dict()
    generator_params::Dict{String,Any} = Dict()
end

"""
    setup_experiments(cfg::ExperimentGrid)

Generates a list of experiments to be run. The list contains one experiment for every combination of the fields in `cfg`.
"""
function setup_experiments(cfg::ExperimentGrid)

    # Store results in new dictionary with arrays of pairs (key, value):
    dict_array_of_pairs = to_kv_pair(cfg)

    # Filter out empty pairs:
    dict_array_of_pairs = filter(
        ((key, value),) -> length(value) > 0, dict_array_of_pairs
    )

    # For each combintation of parameters, create a new experiment:
    output = []
    for kwrgs in product(values(dict_array_of_pairs)...)
        _names = Symbol.([k for (k, _) in kwrgs])
        _values = [v for (_, v) in kwrgs]
        # Get inputs for MetaParams (e.g., data, model):
        idx_meta = .!(v -> typeof(v) <: NamedTuple).(_values)
        meta_kwrgs = (; zip(_names[idx_meta], _values[idx_meta])...)
        # Get other params:
        idx_other = .!(idx_meta)
        other_kwrgs = (; zip(_names[idx_other], _values[idx_other])...)
        exper = Experiment(MetaParams(; meta_kwrgs...); other_kwrgs...)
        push!(output, exper)
    end
    return output
end

to_kv_pair(x) = x

function to_kv_pair(k,vals::AbstractArray{<:Any}) 
    all_pairs = Pair[]
    for v in vals
        push!(all_pairs, k => v)
    end
    return all_pairs
end

function to_kv_pair(k, vals::Dict)
    all_pairs = Pair[]
    length(vals) == 0 && return all_pairs
    # Create a vectors of named tuples for each array in the dict:
    nt_vals = [[(; Symbol(k) => _v) for _v in v] for (k, v) in vals]
    all_combinations = vec([reduce(merge, x) for x in product(nt_vals...)])
    for v in all_combinations
        push!(all_pairs, k => v)
    end
    return all_pairs
end

function to_kv_pair(cfg::ExperimentGrid)
    dict_array_of_pairs = Dict()
    for (key, vals) in to_dict(cfg)
        dict_array_of_pairs[key] = to_kv_pair(key, vals)
    end
    return dict_array_of_pairs
end

