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
function get_config_from_args(; new_save_dir::Union{Nothing,String}=nothing, return_adjusted::Bool=true)

    # Interactive sessions:
    if isinteractive() &&
        !any((x -> contains(x, "--config=")).(ARGS)) &&
        !haskey(ENV, "CONFIG")
        println("Specify the path to your config file.")
        input = readline()
        println("Using config file: $input")
        push!(ARGS, "--config=$input")
        ENV["CONFIG"] = input
    end

    # Command line:
    if any((x -> contains(x, "--config=")).(ARGS))
        config_arg = ARGS[(x -> contains(x, "--config=")).(ARGS)]
        @assert length(config_arg) == 1 "Please provide exactly one config file name."
        fname = replace(config_arg[1], "--config=" => "")
        @assert isfile(fname) "Config file not found: $fname"
    else
        fname = ENV["CONFIG"]
    end

    # Do not return adjust path:
    if !return_adjusted
        return fname
    end

    if isnothing(new_save_dir)
        # Use old directory:
        new_save_dir = ""
    end

    # Load config:
    cfg = CTExperiments.from_toml(fname)

    if any((x -> contains(x, "--data=")).(ARGS))
        requested_dataset =
            ARGS[(x -> contains(x, "--data=")).(ARGS)] |>
            x -> replace(x[1], "--data=" => "")
        @assert requested_dataset in collect(keys(CTExperiments.data_sets)) "Requested dataset not available: $requested_data. Available datasets are $(collect(keys(CTExperiments.data_sets)))."
        if haskey(cfg, "data") && cfg["data"] != requested_dataset
            @info "Using existing config with new dataset: '$requested_dataset' (was '$(cfg["data"])')."
            cfg["data"] = requested_dataset
        end
    end

    if any((x -> contains(x, "--model=")).(ARGS))
        requested_model =
            ARGS[(x -> contains(x, "--model=")).(ARGS)] |>
            x -> replace(x[1], "--model=" => "")
        @assert requested_model in collect(keys(CTExperiments.model_types)) "Requested model type not available: $requested_model. Available models are $(collect(keys(CTExperiments.model_types)))."
        if haskey(cfg, "model_type") && cfg["model_type"] != requested_model
            @info "Using existing config with new model type: '$requested_model' (was '$(cfg["model_type"])')."
            cfg["model_type"] = requested_model
        end
    end

    # Save copy:
    if haskey(cfg, "name")
        cfg["model_type"] = cfg["model_type"] == "" ? "mlp" : cfg["model_type"]
        cfg["data"] = cfg["data"] == "" ? "lin_sep" : cfg["data"]
        cfg["save_dir"] = default_save_dir(new_save_dir, cfg["name"], cfg["data"], cfg["model_type"])
        rootdir, fonly = (dirname(fname), splitdir(fname)[end])
        fname = joinpath(default_save_dir(rootdir, cfg["name"], cfg["data"], cfg["model_type"]), fonly)
        CTExperiments.to_toml(cfg, fname)
    end

    return fname
end

const _mpi_finalize = true

mpi_should_finalize() = _mpi_finalize

function set_mpi_finalize(finalize::Bool)
    global _mpi_finalize = finalize
    return _mpi_finalize
end

"""
    generate_template(
        fname::String="paper/experiments/template_config.toml";
        experiment_name="template",
        overwrite=false,
        kwrgs...,
    )

Generates a template configuration file for experiments. This is useful for quickly setting up a new experiment by copying the generated template into your project directory.
"""
function generate_template(
    fname::String="paper/experiments/template_config.toml";
    experiment_name="template",
    overwrite=false,
    kwrgs...,
)
    write_file = !isfile(fname)     # don't write file if it exists
    if overwrite                    # unless specified
        @warn "File $fname already exists! Overwriting..."
        write_file = true
    end

    if write_file
        save_dir = joinpath(splitpath(fname)[1:(end - 1)])
        exper = Experiment(
            MetaParams(; experiment_name=experiment_name, save_dir=save_dir); kwrgs...
        )
        to_toml(exper, fname)
    else
        @warn "File already exists and not explicitly asked to overwrite it."
    end

    return fname
end

"""
    generate_grid_template(
        fname::String="paper/experiments/template_grid_config.toml"; overwrite=false, kwrgs...
    )

Generates a template configuration file for experiment grids. This is useful for quickly setting up a new grid of experiments by copying the generated template into your project directory.
"""
function generate_grid_template(
    fname::String="paper/experiments/template_grid_config.toml"; overwrite=false, kwrgs...
)
    write_file = !isfile(fname)     # don't write file if it exists
    if overwrite                    # unless specified
        @warn "File $fname already exists! Overwriting..."
        write_file = true
    end

    if write_file
        save_dir = joinpath(splitpath(fname)[1:(end - 1)])
        exper_grid = CTExperiments.ExperimentGrid(; save_dir=save_dir, kwrgs...)
        to_toml(exper_grid, fname)
    else
        @warn "File already exists and not explicitly asked to overwrite it."
    end

    return fname
end

"""
    generate_eval_template(
        fname::String="paper/experiments/template_eval_config.toml";
        overwrite=false,
        save_dir="paper/experiments/template_eval_dir",
    )

Generates a template configuration file for evaluation. This is useful for quickly setting up a new evaluation by copying the generated template into your project directory.
"""
function generate_eval_template(
    fname::String="paper/experiments/template_eval_config.toml";
    overwrite=false,
    save_dir="paper/experiments/template_eval_dir",
)
    write_file = !isfile(fname)     # don't write file if it exists
    if overwrite                    # unless specified
        @warn "File $fname already exists! Overwriting..."
        write_file = true
    end

    if write_file
        grid_file = Logging.with_logger(Logging.NullLogger()) do
            generate_grid_template()
        end
        exper_grid = CTExperiments.ExperimentGrid(grid_file)
        cfg = EvaluationConfig(exper_grid; grid_file=grid_file, save_dir=save_dir)
        to_toml(cfg, fname)
    else
        @warn "File already exists and not explicitly asked to overwrite it."
    end

    return fname
end

"""
    generate_eval_grid_template(
        fname::String="paper/experiments/template_eval_grid_config.toml";
        overwrite=false,
        save_dir="paper/experiments/template_eval_dir",
    )

Generates a template configuration file for evaluation grids. This is useful for quickly setting up a new evaluation grid by copying the generated template into your project directory.
"""
function generate_eval_grid_template(
    fname::String="paper/experiments/template_eval_grid_config.toml";
    overwrite=false,
    save_dir="paper/experiments/template_eval_dir",
)
    write_file = !isfile(fname)     # don't write file if it exists
    if overwrite                    # unless specified
        @warn "File $fname already exists! Overwriting..."
        write_file = true
    end

    if write_file
        grid_file = Logging.with_logger(Logging.NullLogger()) do
            generate_grid_template()
        end
        exper_grid = CTExperiments.ExperimentGrid(grid_file)
        cfg = EvaluationGrid(exper_grid; grid_file=grid_file, save_dir=save_dir)
        to_toml(cfg, fname)
    else
        @warn "File already exists and not explicitly asked to overwrite it."
    end

    return fname
end

export generate_template,
    generate_grid_template, generate_eval_template, generate_eval_grid_template

global _global_seed = 2025

function set_global_seed(seed::Union{Nothing,Int}=_global_seed)
    global _global_seed = if isnothing(seed)
        parse(Int, ENV["GLOBAL_SEED"])
        @info "Found environment variable `ENV['GLOBAL_SEED']`. Setting global seed to it."
    else
        seed
    end
    Random.seed!(_global_seed)
    @info "Global seed set to $_global_seed"
end

get_global_seed() = _global_seed

global _global_dev_dir = "paper/experiments/dev"

function set_global_dev_dir(dir::Union{Nothing,String}=_global_dev_dir)
    global _global_dev_dir = if isnothing(dir)
        ENV["DEV_DIR"]
        @info "Found environment variable `ENV['DEV_DIR']`. Setting global dev dir to it."
    else
        dir
    end
    @info "Global dev dir set to $_global_dev_dir"
    return _global_dev_dir
end

get_global_dev_dir() = _global_dev_dir

export set_global_seed, get_global_seed
