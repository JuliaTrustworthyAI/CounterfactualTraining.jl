using Plots
using Printf: @printf
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
function convert2mnist(x; blue::Bool=false)
    if !blue
        x = (x -> (x -> Gray.(x))(permutedims(reshape(x, 28, 28))))((x .+ 1) ./ 2)
    else
        x = (x -> (x -> RGB.(0, 0, x))(permutedims(reshape(x, 28, 28))))((x .+ 1) ./ 2)
    end
    return x
end

function avg(x::Vector{Float32})
    return sum(x) / length(x)
end

function loss(
    yhat,
    y,
    implausibility,
    regularization,
    validity;
    penalty_strengths=[0.5, 0.001],
    agg=avg,
)

    # Standard classification loss:
    class_loss = Flux.Losses.logitcrossentropy(yhat, y)

    # Implausibility loss (counterfactual):
    implausibility_loss = penalty_strengths[1] * agg(Float32.(implausibility))

    # Regularization loss:
    regularization_loss = penalty_strengths[2] * agg(Float32.(regularization))

    # Total loss:
    ℒ = class_loss + validity + implausibility_loss + regularization_loss

    return ℒ
end

"""
    get_config_from_args()

Retrieves the config file name from the command line arguments. This is used for scripting.
"""
function get_config_from_args()
    haskey(ENV, "config") && return ENV["config"]
    if isinteractive() && !any((x -> contains(x, "--config=")).(ARGS))
        println("Specify the path to your config file.")
        input = readline()
        println("Using config file: $input")
        push!(ARGS, "--config=$input")
        ENV["config"] = input
    end
    config_arg = ARGS[(x -> contains(x, "--config=")).(ARGS)]
    @assert length(config_arg) == 1 "Please provide exactly one config file name."
    fname = replace(config_arg[1], "--config=" => "")
    @assert isfile(fname) "Config file not found: $fname"
    return fname
end


function meminfo_julia()
    # @printf "GC total:  %9.3f MiB\n" Base.gc_total_bytes(Base.gc_num())/2^20
    # Total bytes (above) usually underreports, thus I suggest using live bytes (below)
    @printf "GC live:   %9.3f MiB\n" Base.gc_live_bytes() / 2^20
    @printf "JIT:       %9.3f MiB\n" Base.jit_total_bytes() / 2^20
    @printf "Max. RSS:  %9.3f MiB\n" Sys.maxrss() / 2^20
end