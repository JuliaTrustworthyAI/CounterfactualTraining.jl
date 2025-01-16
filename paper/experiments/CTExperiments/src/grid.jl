using Base.Iterators
using JLD2
using UUIDs

abstract type AbstractGridConfiguration <: AbstractConfiguration end

"""
    ExperimentGrid

A keyword dictionary that contains the parameters for experiments. It is used to generate a list of [`MetaParams`](@ref) objects, one for each unique combination of the fields (see [`generate_list`](@ref)).

## Fields

- `name`: The name of the experiment.
- `data`: The name of the data set. Can be any of the keys in the [`CTExperiments.data_sets`](@ref) dictionary.
- `model_type`: The type of model to use. Can be any of the keys in the [`CTExperiments.model_types`](@ref) dictionary.
- `generator_type`: A vector of generator types to use. Can be any of the keys in the [`CTExperiments.generator_types`](@ref) dictionary.
- `dim_reduction`: A vector of boolean values indicating whether to apply dimensionality reduction to the input data.
- `data_params`: Additional parameters for the dataset, such as the number of samples to use and the batchsize.
- `model_params`: Parameters specific to the model architecture, such as number of layers or activation functions.
- `training_params`: Training parameters, such as learning rate and optimizer.
- `generator_params`: Parameters for the generator.
- `save_dir`: The directory where results will be saved.
"""
struct ExperimentGrid <: AbstractGridConfiguration
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
        data_params = append_params(data_params, get_data_set(data)())

        # Model parameters
        model_params = append_params(model_params, get_model_type(model_type)())

        # Training parameters
        training_params = append_params(training_params, TrainingParams())

        # Generator parameters
        generator_params = append_params(generator_params, GeneratorParams())

        # Instantiate grid: 
        grid = new(
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

        # Store grid config:
        if !isdir(save_dir)
            mkpath(save_dir)
        end
        if !isfile(default_grid_config_name(grid)) &&
            !isfile(joinpath(save_dir, "template_grid_config.toml")) && 
            !isnothing(grid.save_dir)
            to_toml(grid, default_grid_config_name(grid))
        end

        return grid
    end
end

function append_params(params::Union{NamedTuple,AbstractDict}, cfg::AbstractConfiguration)
    kvpairs = [k => getfield(cfg, k) for k in fieldnames(typeof(cfg))]
    return append_params(params, kvpairs)
end

"""
    append_params(params::AbstractDict, default_values::Vector{<:Pair})

Append default values to `params`. This is used for generating TOML files that can be easily filled out by the user.
"""
function append_params(params::AbstractDict, default_values::Vector{<:Pair})
    newparams = Dict()
    for (k,v) in default_values
        k = string(k)
        if !(k in keys(params)) || params[k] == []
            newparams[k] = [to_dict(v)]
        else
            newparams[k] = params[k]
        end
        if typeof(v) <: AbstractConfiguration
            newparams[k] = []

        end
    end
    return newparams
end

"""
    append_params(params::NamedTuple, default_values::Vector{<:Pair})

Extends the [`append_params`](@ref) function to work with `NamedTuple` objects.
"""
function append_params(params::NamedTuple, default_values::Vector{<:Pair})
    params = Dict(zip(string.(keys(params)), values(params)))
    params = append_params(params, default_values)
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

Outer constructor tailored for the `ExperimentGrid` type. It takes a number of keyword arguments, each one (except `data` and `model_type`) being a vector of possible values to be explored by the grid search. Calling [`generate_list(cfg::ExperimentGrid)`](@ref) on an instance of type `ExperimentGrid` will generate a list of experiments for all combinations of these vectors.
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
    save_dir::String=default_save_dir(tempdir(), name, data, model_type),
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
        save_dir,
    )
end

function default_save_dir(rootdir, name, data, model_type)
    return mkpath(joinpath(rootdir, name, model_type, data))
end

function default_save_dir(grid::ExperimentGrid; rootdir=tempdir())
    return default_save_dir(rootdir, grid.name, grid.model_type, grid.data)
end

"""
    ExperimentGrid(fname::String; new_save_dir::Union{Nothing,String}=nothing)

Load an experiment grid from a TOML file. The `fname` argument specifies the path to the TOML file. If `new_save_dir` is provided, it will be used as the save directory for all experiments generated by this grid.
"""
function ExperimentGrid(fname::String; new_save_dir::Union{Nothing,String}=nothing)
    @assert isfile(fname) "Experiment grid configuration file not found."
    dict = from_toml(fname)
    if !isnothing(new_save_dir)
        new_save_dir = default_save_dir(
            new_save_dir, dict["name"], dict["data"], dict["model_type"]
        )
        mkpath(new_save_dir)
        dict["save_dir"] = new_save_dir
    end
    grid = (kwrgs -> ExperimentGrid(; kwrgs...))(CTExperiments.to_ntuple(dict))
    if !isnothing(new_save_dir)
        to_toml(grid, default_grid_config_name(grid))   # store in new save directory
        to_toml(grid, fname)                            # over-write old file with new config
    end
    return grid
end

"""
    default_grid_config_name(grid::ExperimentGrid)

Returns the default name for the configuration file associated with this grid.
"""
function default_grid_config_name(grid::ExperimentGrid)
    return joinpath(grid.save_dir, "grid_config.toml")
end

"""
    generate_list(cfg::ExperimentGrid)

Generates a list of experiments to be run. The list contains one experiment for every combination of the fields in `cfg`.
"""
function generate_list(
    cfg::ExperimentGrid;
    name_prefix::Union{Nothing,String}="experiment",
    store_list::Bool=true,
)

    # Expand grid:
    expanded_grid, experiment_names = expand_grid(cfg; name_prefix=name_prefix)

    # For each combintation of parameters, create a new experiment:
    exper_list = Vector{Experiment}()
    for (i, kwrgs) in enumerate(expanded_grid)

        # Experiment name:
        experiment_name = experiment_names[i]

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
        exper = Experiment(meta; store=store_list, other_kwrgs...)
        push!(exper_list, exper)
    end

    # Store list of experiments:
    if store_list
        save_list(cfg, exper_list)
    end

    return exper_list
end

"""
    expand_grid(
        cfg::AbstractGridConfiguration;
        name_prefix::Union{Nothing,String}="experiment",
    )

Computes the Cartesian product across hyperparameters provided in the grid.
"""
function expand_grid(
    cfg::AbstractGridConfiguration; name_prefix::Union{Nothing,String}="experiment"
)

    # Store results in new dictionary with arrays of pairs (key, value):
    dict_array_of_pairs = to_kv_pair(cfg)

    # Ignore columns ...
    ingore_cols = ["name", "save_dir", "grid_file"]

    # Filter out empty pairs:
    dict_array_of_pairs = filter(
        ((key, value),) -> length(value) > 0 && !(key in ingore_cols), dict_array_of_pairs
    )

    # Generate Cartesian product:
    expanded_grid = product(values(dict_array_of_pairs)...)

    # Experiment names:
    experiment_names = String[]
    for i in 1:length(expanded_grid)
        experiment_name = if isnothing(name_prefix)
            string(uuid1())
        else
            "$(name_prefix)_$(i)"
        end
        push!(experiment_names, experiment_name)
    end

    return expanded_grid, experiment_names
end

function ntasks(grid::AbstractGridConfiguration; include_completed::Bool=false)
    if include_completed
        return length(expand_grid(grid)[2])
    else
        task_list = CTExperiments.generate_list(grid; store_list=false)
        return sum(.!has_results.(task_list))
    end
end

"""
    expand_grid_to_df(cfg::AbstractGridConfiguration)

Expands a grid of hyperparameters into a DataFrame. This is used to eventually merge evaluation results with grid configurations.
"""
function expand_grid_to_df(
    cfg::AbstractGridConfiguration; name_prefix::Union{Nothing,String}="experiment"
)
    df = DataFrame()

    # Expand grid:
    params, exper_names = expand_grid(cfg; name_prefix=name_prefix)

    for (params, _name) in zip(params, exper_names)

        # Step 1: Create a new tuple of Pairs from the original params tuple, 
        # excluding any Pairs with NamedTuple values
        new_params = Tuple(Pair(k, v) for (k, v) in params if !isa(v, NamedTuple))

        # Step 2: Iterate over the original params tuple again
        for (k, v) in params
            # Check if the value `v` is a NamedTuple
            if isa(v, NamedTuple)
                # If so, iterate over the key-value pairs in `v`
                for (k2, v2) in pairs(v)
                    # Create a new Pair with the key `k2` and value `v2`
                    if isa(v2, AbstractVector)
                        v2 = tuple(v2...)
                    end
                    new_params = vcat(new_params..., Pair(string(k2), v2))
                end
            end
        end

        # Collect:
        _df = DataFrame(new_params...; makeunique=true)
        _df.id .= _name
        df = vcat(df, _df)
    end

    select!(df, :id, Not(:id))

    return df
end

"""
    save_list(cfg::ExperimentGrid, exper_list::Vector{<:AbstractExperiment})

Saves the list of experiments corresponding to the experiment grid. This is used to save the list of experiments for later.
"""
function save_list(cfg::ExperimentGrid, exper_list::Vector{<:AbstractExperiment})
    save_dir = cfg.save_dir
    @info "Saving list of experiments to $(save_dir):"
    return jldsave(joinpath(save_dir, "exper_list.jld2"); exper_list)
end

"""
    load_list(cfg::ExperimentGrid)

Loads the list of experiments corresponding to the experiment grid. This is used to load the list of experiments when restarting a session.
"""
function load_list(cfg::ExperimentGrid)
    save_dir = cfg.save_dir
    @info "Loading list of experiments from $(save_dir):"
    @assert isfile(joinpath(save_dir, "exper_list.jld2")) "No list of experiments found in $(save_dir). Did you accidentally delete it?"
    exper_list = JLD2.load(joinpath(save_dir, "exper_list.jld2"), "exper_list")
    return exper_list
end

"""
    to_kv_pair(x)

When called on any object `x`, it returns `x` as-is.
"""
to_kv_pair(x) = x

"""
    to_kv_pair(k, vals::String)

When called on a string `vals`, it returns a single-element array of pairs where each pair consists of the key `k` and the value `vals`.
"""
function to_kv_pair(k, vals::String)
    all_pairs = Pair[k => vals]
    return all_pairs
end

"""
    to_kv_pair(k, vals::Bool)

When called on a bool `vals`, it returns a single-element array of pairs where each pair consists of the key `k` and the value `vals`.
"""
function to_kv_pair(k, vals::Bool)
    all_pairs = Pair[k => vals]
    return all_pairs
end

"""
    to_kv_pair(k, vals::AbstractArray{<:Any})

When called on an array `vals`, it returns an array of pairs where each pair consists of the key `k` and each element of the array `vals`.
"""
function to_kv_pair(k, vals::AbstractArray{<:Any})
    all_pairs = Pair[]
    for v in vals
        push!(all_pairs, k => v)
    end
    return all_pairs
end

"""
    to_kv_pair(k, vals::Dict)

When called on a dictionary `vals`, it returns an array of pairs where each pair consists of the key `k` and each element of the array `vals`.
"""
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

"""
    to_kv_pair(cfg::AbstractGridConfiguration)

Converts the experiment grid `cfg` into a dictionary of pairs. Each key in the dictionary is a string representing the name of an experiment parameter, and each value is an array of pairs where each pair consists of the key and the corresponding value for that parameter.
"""
function to_kv_pair(cfg::AbstractGridConfiguration)
    dict_array_of_pairs = Dict()
    for (key, vals) in to_dict(cfg)
        dict_array_of_pairs[key] = to_kv_pair(key, vals)
    end
    return dict_array_of_pairs
end

"""
    default_evaluation_dir(grid::ExperimentGrid)

Returns the path to the evaluation directory for the experiment grid `grid`. The evaluation directory is created under the `save_dir` of the grid and named "evaluation". If the directory does not exist, it is created.
"""
function default_evaluation_dir(grid::ExperimentGrid)
    return mkpath(joinpath(grid.save_dir, "evaluation"))
end
