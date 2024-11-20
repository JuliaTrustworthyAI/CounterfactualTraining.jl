using CounterfactualExplanations
using CounterfactualExplanations.Evaluation
using CSV
using DataFrames
using Logging
using StatisticalMeasures

abstract type AbstractEvaluationConfig <:AbstractConfiguration end

include("evaluate_counterfactuals.jl")

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

function EvaluationConfig(;
    grid_file::String, save_dir::String, counterfactual_params::NamedTuple=(;), test_time::Bool=false,
)
    counterfactual_params = CounterfactualParams(; counterfactual_params...)
    return EvaluationConfig(grid_file, save_dir, counterfactual_params, test_time)
end

function EvaluationConfig(fname::String)
    @assert isfile(fname) "Experiment file not found."
    dict = from_toml(fname)
    return (kwrgs -> EvaluationConfig(; kwrgs...))(CTExperiments.to_ntuple(dict))
end

function default_eval_config_name(eval_config::EvaluationConfig)
    return joinpath(eval_config.save_dir, "eval_config.toml")
end

function to_toml(eval_config::EvaluationConfig)
    return to_toml(eval_config, default_eval_config_name(eval_config))
end

function test_performance(
    exper::Experiment; measure=[accuracy, multiclass_f1score], n::Union{Nothing,Int}=nothing
)
    model, logs, M = load_results(exper)

    # Get test data:
    Xtest, ytest = get_data(exper.data; n=n, test_set=true)
    yhat = predict_label(M, CounterfactualData(Xtest, ytest), Xtest)

    # Evaluate the model:
    results = [measure(ytest, yhat) for measure in measure]

    return results
end

function test_performance(grid::ExperimentGrid; kwrgs...)
    exper_list = load_list(grid)
    results = Logging.with_logger(Logging.NullLogger()) do
        test_performance.(exper_list; kwrgs...)
    end
    return results
end

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

function save_results(cfg::EvaluationConfig, data::DataFrame, fname::String)
    csv_file = joinpath(cfg.save_dir, fname * ".csv")
    CSV.write(csv_file, data)
    jld2_file = joinpath(cfg.save_dir, fname * ".jld2")
    jldsave(jld2_file; data)
end

function load_results(cfg::EvaluationConfig, fname::String)
    return CSV.read(joinpath(cfg.save_dir, fname * ".csv"), DataFrame)
end
