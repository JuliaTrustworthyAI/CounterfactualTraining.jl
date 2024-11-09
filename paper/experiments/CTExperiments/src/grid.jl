using Base.Iterators
using JLD2
using UUIDs

"""
    ExperimentGrid

A keyword dictionary that contains the parameters for experiments. It is used to generate a list of [`MetaParams`](@ref) objects, one for each unique combination of the fields (see [`setup_experiments`](@ref)).
"""
struct ExperimentGrid <: AbstractConfiguration
    name::String
    data::String
    model_type::String
    generator_type::Vector{<:AbstractString}
    dim_reduction::Vector{<:Bool}
    data_params::Union{AbstractDict,NamedTuple}
    model_params::Union{AbstractDict,NamedTuple}
    training_params::Union{AbstractDict,NamedTuple}
    generator_params::Union{AbstractDict,NamedTuple}
    save_dir::String
    function ExperimentGrid(
        name,
        data,
        model_type,
        generator_type,
        dim_reduction,
        data_params,
        model_params,
        training_params,
        generator_params,
        save_dir,
    )

        # Data parameters
        data_params = append_params(data_params, fieldnames(get_data(data)))

        # Model parameters
        model_params = append_params(model_params, fieldnames(get_model_type(model_type)))

        # Training parameters
        training_params = append_params(training_params, fieldnames(TrainingParams))

        # Generator parameters
        generator_params = append_params(generator_params, fieldnames(GeneratorParams))

        return new(
            name,
            data,
            model_type,
            generator_type,
            dim_reduction,
            data_params,
            model_params,
            training_params,
            generator_params,
            save_dir,
        )
    end
end

function append_params(params::AbstractDict, available_params)
    for x in available_params
        x = string(x)
        if !(x in keys(params))
            params[x] = []
        end
    end
    return params
end

function append_params(params::NamedTuple, available_params)
    params = Dict(zip(string.(keys(params)), values(params)))
    params = append_params(params, available_params)
    return params
end

@doc raw"""
    ExperimentGrid(;
        name::String="grid_\$(string(uuid1()))",
        data::String="mnist",
        model_type::String="mlp",
        generator_type::Vector{<:AbstractString}=["ecco", "generic", "revise"],
        dim_reduction::Vector{<:Bool}=[false],
        data_params::Union{AbstractDict,NamedTuple}=Dict(),
        model_params::Union{AbstractDict,NamedTuple}=Dict(),
        training_params::Union{AbstractDict,NamedTuple}=Dict(),
        generator_params::Union{AbstractDict,NamedTuple}=Dict(),
        save_dir::String=mkpath(joinpath(tempdir(), name)),
    )

Outer constructor tailored for the `ExperimentGrid` type. It takes a number of keyword arguments, each one (except `data` and `model_type`) being a vector of possible values to be explored by the grid search. Calling [`setup_experiments(cfg::ExperimentGrid)`](@ref) on an instance of type `ExperimentGrid` will generate a list of experiments for all combinations of these vectors.
"""
function ExperimentGrid(;
    name::String="grid_$(string(uuid1()))",
    data::String="mnist",
    model_type::String="mlp",
    generator_type::Vector{<:AbstractString}=["ecco", "generic", "revise"],
    dim_reduction::Vector{<:Bool}=[false],
    data_params::Union{AbstractDict,NamedTuple}=Dict(),
    model_params::Union{AbstractDict,NamedTuple}=Dict(),
    training_params::Union{AbstractDict,NamedTuple}=Dict(),
    generator_params::Union{AbstractDict,NamedTuple}=Dict(),
    save_dir::String=mkpath(joinpath(tempdir(), name)),
)
    return ExperimentGrid(
        name,
        data,
        model_type,
        generator_type,
        dim_reduction,
        data_params,
        model_params,
        training_params,
        generator_params,
        save_dir
    )
end

function ExperimentGrid(fname::String; new_save_dir::Union{Nothing,String}=nothing)
    @assert isfile(fname) "Experiment file not found."
    dict = from_toml(fname)
    if !isnothing(new_save_dir)
        mkpath(new_save_dir)
        dict["save_dir"] = new_save_dir
        new_save_name = joinpath(new_save_dir, "grid_config.toml")
    end
    grid = (kwrgs -> ExperimentGrid(; kwrgs...))(CTExperiments.to_ntuple(dict))
    if !isnothing(new_save_dir) 
        to_toml(grid, new_save_name)        # store in new save directory
        to_toml(grid, fname)                # over-write old file with new config
    end
    return grid
end

"""
    setup_experiments(cfg::ExperimentGrid)

Generates a list of experiments to be run. The list contains one experiment for every combination of the fields in `cfg`.
"""
function setup_experiments(
    cfg::ExperimentGrid;
    experiment_name_prefix::Union{Nothing,String}="experiment"
)

    # Store results in new dictionary with arrays of pairs (key, value):
    dict_array_of_pairs = to_kv_pair(cfg)

    # Filter out empty pairs:
    dict_array_of_pairs = filter(
        ((key, value),) -> length(value) > 0 && !(key in ["name", "save_dir"]),
        dict_array_of_pairs,
    )

    # For each combintation of parameters, create a new experiment:
    exper_list = Vector{Experiment}()
    for (i, kwrgs) in enumerate(product(values(dict_array_of_pairs)...))

        # Experiment name:
        experiment_name = if isnothing(experiment_name_prefix)
            string(uuid1())
        else
            "$(experiment_name_prefix)_$(i)"
        end

        # Unpack:
        _names = Symbol.([k for (k, _) in kwrgs])
        _values = [v for (_, v) in kwrgs]
        # Get inputs for MetaParams (e.g., data, model):
        idx_meta = .!(v -> typeof(v) <: NamedTuple).(_values)
        meta_kwrgs = (; zip(_names[idx_meta], _values[idx_meta])...)
        # Get other params:
        idx_other = .!(idx_meta)
        other_kwrgs = (; zip(_names[idx_other], _values[idx_other])...)
        save_dir = mkpath(joinpath(cfg.save_dir, experiment_name))
        meta = MetaParams(;
            experiment_name=experiment_name, save_dir=save_dir, meta_kwrgs...
        )
        exper = Experiment(
            meta;
            other_kwrgs...,
        )
        push!(exper_list, exper)
    end

    # Store list of experiments:
    save_list(cfg, exper_list)

    return exper_list
end

function save_list(cfg::ExperimentGrid, exper_list::Vector{Experiment})
    save_dir = cfg.save_dir
    @info "Saving list of experiments to $(save_dir):"
    return jldsave(joinpath(save_dir, "exper_list.jld2"); exper_list)
end

function load_list(cfg::ExperimentGrid)
    save_dir = cfg.save_dir
    @info "Loading list of experiments from $(save_dir):"
    @assert isfile(joinpath(save_dir, "exper_list.jld2")) "No list of experiments found in $(save_dir). Did you accidentally delete it?"
    exper_list = JLD2.load(joinpath(save_dir, "exper_list.jld2"), "exper_list")
    return exper_list
end

to_kv_pair(x) = x

function to_kv_pair(k, vals::String)
    all_pairs = Pair[k => vals]
    return all_pairs
end

function to_kv_pair(k, vals::AbstractArray{<:Any})
    all_pairs = Pair[]
    for v in vals
        push!(all_pairs, k => v)
    end
    return all_pairs
end

function to_kv_pair(k, vals::Dict)
    all_pairs = Pair[]
    # If the dict is empty, return an empty array:
    length(vals) == 0 && return all_pairs
    # Filter out empty elements in the dict:
    vals = filter(((key, value),) -> length(value) > 0, vals)
    # Create a vectors of named tuples for each array in the dict:
    nt_vals = [[(; Symbol(k) => _v) for _v in v] for (k, v) in vals] 
    # If the array is empty, return an empty array:
    length(nt_vals) == 0 && return all_pairs
    # Create a vector of pairs from each array:
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
