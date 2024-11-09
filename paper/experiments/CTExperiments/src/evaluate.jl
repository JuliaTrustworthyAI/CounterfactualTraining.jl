using CounterfactualExplanations
using CounterfactualExplanations.Evaluation
using StatisticalMeasures

function test_performance(exper::Experiment; measure=[accuracy, multiclass_f1score])
    model, logs, M = load_results(exper)
    
    # Get test data:
    Xtest, ytest = get_test_data(exper.data)
    yhat = predict_label(M, CounterfactualData(Xtest, ytest), Xtest)

    # Evaluate the model:
    results = [measure(ytest, yhat) for measure in measure]

    return results
end

function adv_performance(exper::Experiment; measure=[accuracy, multiclass_f1score])
    model, logs, M = load_results(exper)

    # Get test data:
    Xtest, ytest = get_test_data(exper.data)

    # Generate adversarial examples:

    # Evaluate the model:
    yhat = predict_label(M, CounterfactualData(Xtest, ytest), Xtest)
    results = [measure(ytest, yhat) for measure in measure]

    return results
end

function evaluate_counterfactuals(
    grid::ExperimentGrid;
    generators::Union{Nothing,Dict{Symbol,AbstractGenerator}}=nothing,
    measure::Vector{<:PenaltyOrFun}=[validity, plausibility],
)
    exper_list = load_list(grid)

    # Get all available test data:
    dataset = get_data(grid.data) |>
        dataset_type -> get_test_data(dataset_type; n=nothing) |>
        dt -> CounterfactualData(dt...)

    # Get models:
    models = Dict(
        [
            exper.meta_params.experiment_name => load_results(exper)[3] for
            exper in exper_list
        ]...,
    )

    # Counterfactual generators:
    if isnothing(generators)
        generators = Dict(
            :ecco => ECCoGenerator(),
            :wachter => GenericGenerator(),
        )
    end

    # Generate and benchmark counterfactuals:
    pllr = get_parallelizer(exper_list[1].training_params)
    return benchmark(
        dataset; models=models, generators=generators, measure=measure, parallelizer=pllr
    )

end