using Plots
using Random
using TaijaData: load_mnist_test

"""
    plot_all_mnist(gen, model, data = load_mnist_test(); img_height = 150, seed = 123)

Plots counterfactuals for all factual and target labels for a specified model and generator.
"""
function plot_all_mnist(
    gen,
    model,
    data=CounterfactualData(load_mnist_test()...);
    convergence=DecisionThresholdConvergence(),
    img_height=150,
    seed=123,
)

    # Set seed:
    if !isnothing(seed)
        Random.seed!(seed)
    end

    targets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    factuals = targets
    plts = []

    for factual in factuals
        chosen = rand(findall(data.output_encoder.labels .== factual))
        x = select_factual(data, chosen)
        for target in targets
            if factual != target
                @info "Generating counterfactual for $(factual) -> $(target)"
                ce = generate_counterfactual(
                    x,
                    target,
                    data,
                    model,
                    gen;
                    initialization=:identity,
                    convergence=convergence,
                )
                plt = Plots.plot(
                    convert2mnist(CounterfactualExplanations.counterfactual(ce));
                    axis=([], false),
                    size=(img_height, img_height),
                    title="$factual → $target",
                )
            else
                plt = Plots.plot(
                    convert2mnist(x);
                    axis=([], false),
                    size=(img_height, img_height),
                    title="Factual",
                )
            end
            push!(plts, plt)
        end
    end

    plt = Plots.plot(
        plts...;
        layout=(length(factuals), length(targets)),
        size=(img_height * length(targets), img_height * length(factuals)),
        dpi=300,
    )

    return plt
end

"""
    convert2mnist(x)

Converts a vector to a 28x28 grey image.
"""
function convert2mnist(x)
    x = (x -> (x -> Gray.(x))(permutedims(reshape(x, 28, 28))))((x .+ 1) ./ 2)
    return x
end

function avg(x::Vector{Float32})
    return sum(x) / length(x)
end

function loss(yhat, y, implausibility; λ=1.0, agg=avg)
    class_loss = Flux.Losses.logitcrossentropy(yhat, y)
    return class_loss + λ * avg(Float32.(implausibility))
end