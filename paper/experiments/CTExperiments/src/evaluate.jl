using StatisticalMeasures

function test_performance(exper::Experiment; measure=[accuracy, multiclass_f1score])
    model, logs, M = load_results(exper)
    
    # Get test data:
    Xtest, ytest = get_test_data(exper, exper.data)
    yhat = predict_label(M, CounterfactualData(Xtest, ytest), Xtest)

    # Evaluate the model:
    results = [measure(ytest, yhat) for measure in measure]

    return results
end

function adv_performance(exper::Experiment; measure=[accuracy, multiclass_f1score])
    model, logs, M = load_results(exper)

    # Get test data:
    Xtest, ytest = get_test_data(exper, exper.data)

    # Generate adversarial examples:

    # Evaluate the model:
    yhat = predict_label(M, CounterfactualData(Xtest, ytest), Xtest)
    results = [measure(ytest, yhat) for measure in measure]

    return results
end

function evaluate(grid::ExperimentGrid)
end