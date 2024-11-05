using Base.Iterators

"""
    ExperimentGrid

A keyword dictionary that contains the parameters for experiments. It is used to generate a list of [`MetaParams`](@ref) objects, one for each unique combination of the fields (see [`setup_experiments`](@ref)).
"""
struct ExperimentGrid <: AbstractConfiguration
    data::String 
    model_type::String 
    generator_type::Vector{<:AbstractString}
    dim_reduction::Vector{<:Bool} 
    data_params::AbstractDict
    model_params::AbstractDict
    training_params::AbstractDict
    generator_params::AbstractDict
    function ExperimentGrid(
        data,
        model_type,
        generator_type,
        dim_reduction,
        data_params,
        model_params,
        training_params,
        generator_params,
    )
        
        # Data parameters
        append_params!(data_params, fieldnames(get_data(data)))

        # Model parameters
        append_params!(model_params, fieldnames(get_model_type(model_type)))

        # Training parameters
        append_params!(training_params, fieldnames(TrainingParams))

        # Generator parameters
        append_params!(generator_params, fieldnames(GeneratorParams))

        return new(
            data, 
            model_type, 
            generator_type, 
            dim_reduction, 
            data_params, 
            model_params, 
            training_params, 
            generator_params
        )
    end
end

function append_params!(params::AbstractDict, available_params)
    for x in available_params
        x = string(x)
        if  !(x in keys(params))
            params[x] = []
        end
    end
    return params
end


function ExperimentGrid(;
    data::String="mnist",
    model_type::String="mlp",
    generator_type::Vector{<:AbstractString}=["ecco", "generic", "revise"],
    dim_reduction::Vector{<:Bool}=[false],
    data_params::AbstractDict=Dict(),
    model_params::AbstractDict=Dict(),
    training_params::AbstractDict=Dict(),
    generator_params::AbstractDict=Dict(),
)
    return ExperimentGrid( 
        data,
        model_type, 
        generator_type, 
        dim_reduction, 
        data_params, 
        model_params, 
        training_params, 
        generator_params
     )
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

function to_kv_pair(k, vals::String)
    all_pairs = Pair[k => vals]
    return all_pairs
end

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

