using Accessors
using CounterfactualExplanations
using CounterfactualExplanations.Evaluation
using CSV
using DataFrames
using Logging
using StatisticalMeasures

abstract type AbstractEvaluationConfig <: AbstractConfiguration end

include("evaluate_counterfactuals.jl")

"""
    EvaluationConfig <: AbstractEvaluationConfig

Configuration for evaluation of counterfactuals. This configuration includes the path to a grid file, a directory for saving results, parameters for generating counterfactuals, and a Boolean indicating whether the evaluation happens at test time or during the hyperparameter tuning stage. 
    
The constructor checks whether the grid file exists and saves the configuration to a TOML file in the specified directory (unless it already exists).
"""
struct EvaluationConfig <: AbstractEvaluationConfig
    grid_file::String
    save_dir::String
    counterfactual_params::CounterfactualParams
    test_time::Bool
    function EvaluationConfig(grid_file, save_dir, counterfactual_params, test_time)
        @assert isfile(grid_file) "Grid file not found: $grid_file"
        config = new(grid_file, save_dir, counterfactual_params, test_time)
        if !isfile(joinpath(save_dir, "eval_config.toml"))
            to_toml(config, joinpath(save_dir, "eval_config.toml"))
        end
        return config
    end
end

"""
    EvaluationConfig(
        grid::ExperimentGrid;
        grid_file::Union{Nothing,String}=nothing,
        save_dir::Union{Nothing,String}=nothing,
        counterfactual_params::NamedTuple=(;),
        test_time::Bool=false,
    )

Outer constructor for `EvaluationConfig`. It takes an `ExperimentGrid` and optional parameters for the grid file, save directory, counterfactual parameters, and whether it's a test time evaluation. If no grid file is provided, it defaults to using the name of the experiment grid. The constructor checks if the grid file exists and saves the configuration to a TOML file in the specified directory (unless it already exists). 
"""
function EvaluationConfig(
    grid::ExperimentGrid;
    grid_file::Union{Nothing,String}=nothing,
    save_dir::Union{Nothing,String}=nothing,
    counterfactual_params::NamedTuple=(;),
    test_time::Bool=false,
)
    save_dir = if isnothing(save_dir)
        default_evaluation_dir(grid)
    else
        save_dir
    end
    counterfactual_params = CounterfactualParams(; counterfactual_params...)
    grid_file = isnothing(grid_file) ? default_grid_config_name(grid) : grid_file
    return EvaluationConfig(grid_file, save_dir, counterfactual_params, test_time)
end

"""
    EvaluationConfig(;
        grid_file::String,
        save_dir::String,
        counterfactual_params::NamedTuple=(;),
        test_time::Bool=false,
    )

Outer constructor where all inputs are passed as keyword arguments.
"""
function EvaluationConfig(;
    grid_file::String,
    save_dir::String,
    counterfactual_params::NamedTuple=(;),
    test_time::Bool=false,
    generator_params::Union{Nothing,NamedTuple}=nothing,
)
    if !isnothing(generator_params)
        # append generator params to counterfactual params:
        counterfactual_params = @insert counterfactual_params.generator_params =
            generator_params
    end
    counterfactual_params = CounterfactualParams(; counterfactual_params...)
    return EvaluationConfig(grid_file, save_dir, counterfactual_params, test_time)
end

"""
    EvaluationConfig(fname::String)

Outer constructor that reads a TOML file and returns an `EvaluationConfig` object. The TOML file should contain the necessary configuration parameters for the evaluation.
"""
function EvaluationConfig(fname::String)
    @assert isfile(fname) "Experiment file not found."
    dict = from_toml(fname)
    return (kwrgs -> EvaluationConfig(; kwrgs...))(CTExperiments.to_ntuple(dict))
end

"""
    default_eval_config_name(eval_config::EvaluationConfig)

Returns the default name for the evaluation configuration file. This is typically used to save the configuration in a specific directory.
"""
function default_eval_config_name(eval_config::EvaluationConfig)
    return joinpath(eval_config.save_dir, "eval_config.toml")
end

"""
    to_toml(eval_config::EvaluationConfig)

Dispatches the `to_toml` function over the `EvaluationConfig` type. This function is used to save the configuration in a TOML file.
"""
function to_toml(eval_config::EvaluationConfig)
    return to_toml(eval_config, default_eval_config_name(eval_config))
end

"""
    test_performance(
        exper::Experiment; measure=[accuracy, multiclass_f1score], n::Union{Nothing,Int}=nothing
    )

Tests the performance of a trained model on the test set and returns the evaluation metrics. The `measure` parameter specifies which metric(s) to evaluate, and `n` can be used to limit the number of samples used for testing.
"""
function test_performance(
    exper::Experiment; measure=[accuracy, multiclass_f1score], n::Union{Nothing,Int}=nothing
)
    _, _, M = load_results(exper)

    # Get test data:
    Xtest, ytest = get_data(exper.data; n=n, test_set=true)
    yhat = predict_label(M, CounterfactualData(Xtest, ytest), Xtest)

    # Evaluate the model:
    results = [measure(ytest, yhat) for measure in measure]

    return results
end

"""
    test_performance(grid::ExperimentGrid; kwrgs...)

Tests the performance of a trained model on the test set for each experiment in an `ExperimentGrid` and returns the evaluation metrics. The `measure` parameter specifies which metric(s) to evaluate.
"""
function test_performance(grid::ExperimentGrid; kwrgs...)
    exper_list = load_list(grid)
    results = Logging.with_logger(Logging.NullLogger()) do
        test_performance.(exper_list; kwrgs...)
    end
    return results
end

"""
    adv_performance(exper::Experiment; measure=[accuracy, multiclass_f1score])

Tests the performance of a trained model on adversarial examples and returns the evaluation metrics. The `measure` parameter specifies which metric(s) to evaluate.
"""
function adv_performance(exper::Experiment; measure=[accuracy, multiclass_f1score])
    model, logs, M = load_results(exper)

    # Get test data:
    Xtest, ytest = get_data(exper.data; test_set=true)

    # Generate adversarial examples:

    # Evaluate the model:
    yhat = predict_label(M, CounterfactualData(Xtest, ytest), Xtest)
    results = [measure(ytest, yhat) for measure in measure]

    return results
end

"""
    save_results(cfg::EvaluationConfig, data::DataFrame, fname::String)

Saves the evaluation results to a CSV and JLD2 file.
"""
function save_results(cfg::EvaluationConfig, data::DataFrame, fname::String)
    csv_file = joinpath(cfg.save_dir, fname * ".csv")
    CSV.write(csv_file, data)
    jld2_file = joinpath(cfg.save_dir, fname * ".jld2")
    return jldsave(jld2_file; data)
end

"""
    load_results(cfg::EvaluationConfig, fname::String)

Loads the evaluation results from a CSV.
"""
function load_results(cfg::EvaluationConfig, fname::String)
    return CSV.read(joinpath(cfg.save_dir, fname * ".csv"), DataFrame)
end

"""
    set_work_dir(cfg::EvaluationConfig, eval_work_root::String)

A working directory for evaluation results.
"""
function set_work_dir(cfg::EvaluationConfig, eval_work_root::String)
    work_dir = get_work_dir(cfg, eval_work_root)
    if !isfile(joinpath(work_dir, "eval_config.toml"))
        to_toml(cfg, joinpath(work_dir, "eval_config.toml"))
    end
    @info "Working directory for this evaluation is $work_dir. Use this folder to store script for presenting results (plots, tables, etc.)."
    if !isfile(joinpath(work_dir, "grid_config.toml"))
        cp(cfg.grid_file, joinpath(work_dir, "grid_config.toml"))
    end
    return work_dir
end

function get_work_dir(cfg::EvaluationConfig, eval_work_root::String)
    return mkpath(joinpath(eval_work_root, splitpath(cfg.save_dir)[end]))
end

results_dir(cfg::EvaluationConfig) = joinpath(cfg.save_dir, "results")

"""
    make_dummy(cfg::EvaluationConfig)

Modify the configurations parameters to create a dummy version of it. This is used in the context of multi-processing to ensure that each process receives the same number of tasks.   
"""
function make_dummy(cfg::EvaluationConfig, suffix1, suffix2)
    old_save_dir = cfg.save_dir
    @reset cfg.save_dir = mkpath(
        joinpath(
            splitpath(old_save_dir)[1:(end - 1)]..., "dummy_eval_$(suffix1)_$(suffix2)"
        ),
    )
    return cfg
end

function isdummy(cfg::EvaluationConfig)::Bool
    foldername = splitpath(cfg.save_dir)[end]
    return contains(lowercase(foldername), "dummy")
end

function remove_dummy!(cfg::EvaluationConfig)
    if isdummy(cfg)
        rm(cfg.save_dir; recursive=true)
        @info "Removed dummy experiment: $(cfg.save_dir)"
    end
end
