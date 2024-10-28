using Base.Iterators

Base.@kwdef struct ExperimentGrid <: AbstractConfiguration
    generator_type::Vector{<:AbstractString} = ["ECCo", "Generic", "REVISE"]
    model_type::Vector{<:AbstractString} = ["MLPModel"]
    dim_reduction::Vector{<:Bool} = [false]
end

function setup_experiments(cfg::ExperimentGrid)

    # Store results in new dictionary with arrays of pairs (key, value):
    dict_array_of_pairs = Dict()
    for (key, values) in to_dict(cfg)
        all_pairs = Pair[]
        for value in values
            push!(all_pairs, key => value)
        end
        dict_array_of_pairs[key] = all_pairs
    end

    # For each combintation of parameters, create a new experiment:
    output = []
    for kwrgs in product(values(dict_array_of_pairs)...)
        _names = Symbol.([k for (k, _) in kwrgs])
        _values = [v for (_, v) in kwrgs]
        kwrgs = (; zip(_names, _values)...)
        push!(output, MetaParams(;kwrgs...))
    end
    return output
end