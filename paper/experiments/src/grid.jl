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
end

"""
    setup_experiments(cfg::ExperimentGrid)

Generates a list of experiments to be run. The list contains one experiment for every combination of the fields in `cfg`.
"""
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
        exper = Experiment(MetaParams(; kwrgs...))
        push!(output, exper)
    end
    return output
end