using Base.Iterators
using JLD2
using UUIDs

"""
    EvaluationGrid

A configuration object for the evaluation grid.

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
    function EvaluationGrid(
        grid_file, 
        save_dir,
        counterfactual_params,
        generator_params,
        test_time,
    )

        # Counterfactual params:
        counterfactual_params = append_params(counterfactual_params, fieldnames(CounterfactualParams))

        # Generator parameters
        generator_params = append_params(generator_params, fieldnames(GeneratorParams))

        # Instantiate grid: 
        grid = new(
            grid_file,
            save_dir,
            counterfactual_params,
            generator_params,
            test_time,
        )

        # Store grid config:
        if !isdir(save_dir)
            mkpath(save_dir)
        end

        if !isfile(default_grid_config_name(grid))
            to_toml(grid, default_grid_config_name(grid))
        end

        return grid
    end
end

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
    generator_params::NamedTuple=(;),
    test_time::Bool=false,
)
    save_dir = if isnothing(save_dir)
        default_evaluation_dir(grid)
    else
        save_dir
    end
    grid_file = isnothing(grid_file) ? default_grid_config_name(grid) : grid_file
    return EvaluationGrid(grid_file, save_dir, counterfactual_params, generator_params, test_time)
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
)
    return EvaluationGrid(
        grid_file,
        save_dir,
        counterfactual_params,
        generator_params,
        test_time,
    )
end

"""
    EvaluationGrid(fname::String; new_save_dir::Union{Nothing,String}=nothing)

Outer constructor dispatched over `fname::String`.
"""
function EvaluationGrid(fname::String; new_save_dir::Union{Nothing,String}=nothing)
    @assert isfile(fname) "Evaluation grid configuration file not found."
    dict = from_toml(fname)
    if !isnothing(new_save_dir)
        mkpath(new_save_dir)
        dict["save_dir"] = new_save_dir
    end
    grid = (kwrgs -> EvaluationGrid(; kwrgs...))(CTExperiments.to_ntuple(dict))
    if !isnothing(new_save_dir)
        to_toml(grid, default_grid_config_name(grid))   # store in new save directory
        to_toml(grid, fname)                            # over-write old file with new config
    end
    return grid
end

"""
    default_grid_config_name(grid::EvaluationGrid)

Returns the default name for the configuration file associated with this grid.
"""
function default_grid_config_name(grid::EvaluationGrid)
    return joinpath(grid.save_dir, "evaluation_grid_config.toml")
end

"""
    setup_evaluations(cfg::EvaluationGrid)

Generates a list of evaluations to be run. The list contains one evaluation for every combination of the fields in `cfg`.
"""
function setup_evaluations(
    cfg::EvaluationGrid; name_prefix::Union{Nothing,String}="evaluation"
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
        _names = Symbol.([k for (k, _) in kwrgs])
        _values = [v for (_, v) in kwrgs]

        evaluation = EvaluationConfig(;
            grid_file=cfg.grid_file, save_dir=save_dir, (; zip(_names, _values)...)...
        )
        push!(eval_list, evaluation)
    end

    # Store list of experiments:
    # save_list(cfg, exper_list)

    return eval_list
end