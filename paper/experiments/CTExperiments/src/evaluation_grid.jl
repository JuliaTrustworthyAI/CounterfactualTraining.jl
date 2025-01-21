using Base.Iterators
using JLD2
using UUIDs

global _default_generator_params_eval_grid = (
    lambda_cost=[0.0],
    lambda_energy=[0.1, 0.5, 1.0, 5.0, 10.0, 20.0],
)

"""
    EvaluationGrid

A configuration object for the evaluation grid. The `generator_params` if the `EvaluationGrid` are inherited from the `ExperimentGrid`, but additional values can be supplied to be merged with the inherited values. 

## Fields

- `grid_file::String`: The path to the TOML file containing the evaluation grid configuration.
- `save_dir::String`: The directory where the results will be saved.
- `counterfactual_params::Union{AbstractDict,NamedTuple}`: A dictionary or named tuple of parameters for counterfactual generation.
- `generator_params::Union{AbstractDict,NamedTuple}`: A dictionary or named tuple of parameters for the generator.
- `test_time::Bool`: Whether the evaluation is run at test time or validation.
"""
struct EvaluationGrid <: AbstractGridConfiguration
    grid_file::String
    save_dir::String
    counterfactual_params::Union{AbstractDict,NamedTuple}
    generator_params::Union{AbstractDict,NamedTuple}
    test_time::Bool
    inherit::Bool
    function EvaluationGrid(
        grid_file, save_dir, counterfactual_params, generator_params, test_time, inherit
    )

        # Counterfactual params:
        counterfactual_params = append_params(
            counterfactual_params, CounterfactualParams()
        )

        # Generator parameters:
        generator_params = append_params(generator_params, GeneratorParams())
        if inherit
            inherited_generator_params = CTExperiments.from_toml(grid_file)["generator_params"]
            merged_params = Dict{String,Any}()
            for (k, v) in generator_params
                merged_values = unique([inherited_generator_params[k]..., v...])
                merged_params[k] = sort(merged_values)
            end
            generator_params = merged_params
        end

        # Instantiate grid: 
        grid = new(grid_file, save_dir, counterfactual_params, generator_params, test_time, inherit)

        # If grid file exists already, return that one:
        if isfile(default_grid_config_name(grid))
            @info "Using existing config file: $(default_grid_config_name(grid))."
            grid = EvaluationGrid(default_grid_config_name(grid))
            return grid
        end

        # Store grid config:
        if !isdir(save_dir)
            mkpath(save_dir)
        end  
        if !isfile(default_grid_config_name(grid)) &&
            !isfile(joinpath(save_dir, "template_eval_grid_config.toml"))
            to_toml(grid, default_grid_config_name(grid))
        end

        return grid
    end
end

const EvalConfigOrGrid = Union{AbstractEvaluationConfig,EvaluationGrid}

"""
    EvaluationGrid(
        grid::ExperimentGrid;
        grid_file::Union{Nothing,String}=nothing,
        save_dir::Union{Nothing,String}=nothing,
        counterfactual_params::NamedTuple=(;),
        generator_params::NamedTuple=(;),
        test_time::Bool=false,
    )

Outer constructor for evaluation grid dispatched over `grid::ExperimentGrid`. 
"""
function EvaluationGrid(
    grid::ExperimentGrid;
    grid_file::Union{Nothing,String}=nothing,
    save_dir::Union{Nothing,String}=nothing,
    counterfactual_params::NamedTuple=(;),
    generator_params::NamedTuple=_default_generator_params_eval_grid,
    test_time::Bool=false,
    inherit::Bool=get_global_param("inherit", true),
)
    save_dir = if isnothing(save_dir)
        default_evaluation_dir(grid)
    else
        save_dir
    end
    grid_file = isnothing(grid_file) ? default_grid_config_name(grid) : grid_file
    return EvaluationGrid(
        grid_file, save_dir, counterfactual_params, generator_params, test_time, inherit
    )
end

"""
    EvaluationGrid(;
        grid_file::String,
        save_dir::String,
        counterfactual_params::Union{AbstractDict,NamedTuple},
        generator_params::Union{AbstractDict,NamedTuple},
        test_time::Bool,
    )

Outer constructor that accepts all fields as keyword arguments. 
"""
function EvaluationGrid(;
    grid_file::String,
    save_dir::String,
    counterfactual_params::Union{AbstractDict,NamedTuple},
    generator_params::Union{AbstractDict,NamedTuple},
    test_time::Bool,
    inherit::Bool,
)
    return EvaluationGrid(
        grid_file, save_dir, counterfactual_params, generator_params, test_time, inherit
    )
end

"""
    EvaluationGrid(fname::String; new_save_dir::Union{Nothing,String}=nothing)

Outer constructor dispatched over `fname::String`.
"""
function EvaluationGrid(
    fname::String;
    new_save_dir::Union{Nothing,String}=nothing,
    inherit::Bool=get_global_param("inherit", true),
)
    @assert isfile(fname) "Evaluation grid configuration file not found."
    dict = from_toml(fname)
    if !haskey(dict, "name")
        if !isnothing(new_save_dir)
            mkpath(new_save_dir)
            dict["save_dir"] = new_save_dir
        end
        eval_grid = (kwrgs -> EvaluationGrid(; kwrgs...))(CTExperiments.to_ntuple(dict))
    else
        @info "Supplied file path to `ExperimentGrid`. Using default parameters for `EvaluationGrid`."
        eval_grid = (exper_grid -> EvaluationGrid(exper_grid; inherit=inherit))(ExperimentGrid(fname))
    end
    if !isnothing(new_save_dir)
        to_toml(eval_grid, default_grid_config_name(eval_grid))     # store in new save directory
        to_toml(eval_grid, fname)                                   # over-write old file with new config
    end
    return eval_grid
end

"""
    default_grid_config_name(grid::EvaluationGrid)

Returns the default name for the configuration file associated with this grid.
"""
function default_grid_config_name(grid::EvaluationGrid)
    return joinpath(grid.save_dir, "evaluation_grid_config.toml")
end

"""
    generate_list(cfg::EvaluationGrid)

Generates a list of evaluations to be run. The list contains one evaluation for every combination of the fields in `cfg`.
"""
function generate_list(
    cfg::EvaluationGrid;
    name_prefix::Union{Nothing,String}="evaluation",
    store_list::Bool=true,
)

    # Expand grid:
    expanded_grid, evaluation_names = expand_grid(cfg; name_prefix=name_prefix)

    # For each combintation of parameters, create a new experiment:
    eval_list = Vector{EvaluationConfig}()
    for (i, kwrgs) in enumerate(expanded_grid)

        # Evaluation name:
        evaluation_name = evaluation_names[i]
        save_dir = mkpath(joinpath(cfg.save_dir, evaluation_name))

        # Unpack:
        dont_include = [
            "inherit",
        ]
        _names = Symbol.([k for (k, _) in kwrgs if !(k in dont_include)])
        _values = [v for (k, v) in kwrgs if !(k in dont_include)]

        evaluation = EvaluationConfig(;
            grid_file=cfg.grid_file, save_dir=save_dir, (; zip(_names, _values)...)...
        )
        push!(eval_list, evaluation)
    end

    # Store list of experiments:
    if store_list
        save_list(cfg, eval_list)
    end

    return eval_list
end

"""
    set_work_dir(grid::EvaluationGrid, cfg::EvaluationConfig, eval_work_root::String)

A working directory for evaluation grid results.
"""
function set_work_dir(
    grid::EvaluationGrid,
    cfg::EvaluationConfig,
    eval_work_root::String,
    output_work_root::String,
)

    work_dir = get_work_dir(grid, cfg, eval_work_root, output_work_root)

    # Evaluation specific:
    if !isfile(joinpath(work_dir, "eval_config.toml"))
        to_toml(cfg, joinpath(work_dir, "eval_config.toml"))
    end
    @info "Working directory for this evaluation is $work_dir. Use this folder to store script for presenting results (plots, tables, etc.)."

    # Root directory for all evaluations:
    root_dir = joinpath(splitpath(work_dir)[1:(end - 1)]...)
    if !isfile(joinpath(root_dir, "grid_config.toml"))
        cp(cfg.grid_file, joinpath(root_dir, "grid_config.toml"))
    end
    if !isfile(joinpath(root_dir, "evaluation_grid_config.toml"))
        to_toml(grid, joinpath(root_dir, "evaluation_grid_config.toml"))
    end

    return work_dir
end

"""
    get_work_dir(grid::EvaluationGrid, cfg::EvaluationConfig, eval_work_root::String)

Get the working directory for evaluation grid results.
"""
function get_work_dir(
    grid::EvaluationGrid,
    cfg::EvaluationConfig,
    eval_work_root::String,
    output_work_root::String,
)
    _root = get_work_dir(grid, eval_work_root, output_work_root)
    return mkpath(joinpath(_root, splitpath(cfg.save_dir)[end]))
end

"""
    get_work_dir(grid::EvaluationGrid, eval_work_root::String, output_work_root::String)

Get the root working directory for evaluation grid results.
"""
function get_work_dir(grid::EvaluationGrid, eval_work_root::String, output_work_root::String)
    dir = replace(grid.save_dir, output_work_root => eval_work_root)
    return mkpath(dir)
end

results_dir(grid::EvaluationGrid) = joinpath(grid.save_dir, "results")

"""
    save_list(grid::EvaluationGrid, exper_list::Vector{<:AbstractExperiment})

Saves the list of evaluations corresponding to the evaluation grid for easy down-stream usage.
"""
function save_list(grid::EvaluationGrid, eval_list::Vector{<:EvaluationConfig})
    save_dir = grid.save_dir
    @info "Saving list of evaluations to $(save_dir):"
    return jldsave(joinpath(save_dir, "eval_list.jld2"); eval_list)
end

"""
    load_list(grid::EvaluationGrid)

Loads the list of evaluations corresponding to the evaluation grid.
"""
function load_list(grid::EvaluationGrid)
    save_dir = grid.save_dir
    @info "Loading list of evaluations from $(save_dir):"
    @assert isfile(joinpath(save_dir, "eval_list.jld2")) "No list of evaluations found in $(save_dir). Did you accidentally delete it?"
    eval_list = JLD2.load(joinpath(save_dir, "eval_list.jld2"), "eval_list")
    return eval_list
end

"""
    load_ce_evaluation(grid::EvaluationGrid)

Loads the benchmark. 
"""
function load_ce_evaluation(grid::EvaluationGrid)

    # Load list:
    eval_list = load_list(grid)
    evals = DataFrame[]

    for (i, cfg) in enumerate(eval_list)
        evaluation = load_results(cfg, Benchmark, default_bmk_name(cfg)).evaluation
        evaluation.evaluation .= splitpath(cfg.save_dir)[end]
        push!(evals, evaluation)
    end

    # Combine:
    evaluation = reduce(vcat, evals)
    if "model" in names(evaluation) && !("id" in names(evaluation))
        rename!(evaluation, :model => :id)
    end
    select!(evaluation, :evaluation, :id, Not(:evaluation, :id))

    return evaluation
end

get_data_set(grid::EvalConfigOrGrid) = get_data_set(ExperimentGrid(grid.grid_file).data)

function get_data_seed(grid::EvalConfigOrGrid)
    _seed = ExperimentGrid(grid.grid_file).data_params["train_test_seed"] |>
        unique
    @assert length(_seed) == 1 "Did you specify multiple seeds?"
    return _seed[1]
end

function get_data_params(grid::EvalConfigOrGrid, param::String)
    data_params = ExperimentGrid(grid.grid_file).data_params
    @assert param in keys(data_params)
    val = data_params[param] |> unique
    println(val)
    @assert length(val) == 1 "Did you specify multiple values for $param?"
    return val[1]
end

function get_ce_data(cfg::AbstractEvaluationConfig)
    return (dt -> CounterfactualData(dt...))(get_data(cfg))
end

function get_data(cfg::AbstractEvaluationConfig)
    # Get data:
    data = (
        dataset_type -> (get_data(
            dataset_type(;
                train_test_seed=get_data_params(cfg, "train_test_seed"),
            );
            n=nothing,
            test_set=cfg.test_time,
        ))
    )(
        get_data_set(cfg)
    )

    return data
end