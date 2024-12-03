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
struct EvaluationGrid <: AbstractConfiguration
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

Outer constructor for evaluation grid.
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
    default_grid_config_name(grid::EvaluationGrid)

Returns the default name for the configuration file associated with this grid.
"""
function default_grid_config_name(grid::EvaluationGrid)
    return joinpath(grid.save_dir, "evaluation_grid_config.toml")
end