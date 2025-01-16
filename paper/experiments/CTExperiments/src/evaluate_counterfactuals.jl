using CounterfactualExplanations
using CounterfactualExplanations.Evaluation
using DataFrames
using JLD2
using Serialization
using TaijaParallel

"""
    CounterfactualParams

Struct holding keyword arguments relevant to the evaluation of counterfactual explanations for fitted models.

## Fields

- `generator_params`: A `GeneratorParams` struct containing parameters for the counterfactual generation process.
- `n_individuals`: The number of individuals to generate counterfactual explanations for (per model and generator).
- `n_runs`: The number of runs to perform for each model and generator.
- `conv`: The convergence criterion used to stop the optimization process. Can be any of the options specified by [`CTExperiments.conv_catalogue`](@ref).
- `maxiter`: The maximum number of iterations allowed during the optimization process.
- `vertical_splits`: The number of vertical splits to use when generating counterfactual explanations. This can be used to reduce peak memory (higher values mean less memory usage).
- `store_ce`: A boolean indicating whether to store the counterfactual explanations for each individual and run.
- `parallelizer`: The parallelization strategy to use. Can be either `"threads"` or `"mpi"`.
- `threaded`: A boolean indicating whether to also use multi-threading when using `"mpi"` for parallelization.
- `concatenate_output`: A boolean indicating whether to concatenate the output of multiple runs into a single evaluation. Setting this to `false` can help to avoid out-of-memory errors when dealing with large benchmarks.
- `verbose`: A boolean indicating whether to print verbose output during the evaluation process.
"""
Base.@kwdef struct CounterfactualParams <: AbstractConfiguration
    generator_params::GeneratorParams = GeneratorParams()
    n_individuals::Int = get_global_param("n_individuals", 100)
    n_runs::Int = 5
    conv::AbstractString = "max_iter"
    maxiter::Int = 100
    vertical_splits::Int = get_global_param("vertical_splits", 100)
    store_ce::Bool = false
    parallelizer::AbstractString = "mpi"
    threaded::Bool = false
    concatenate_output::Bool = get_global_param("concatenate_output", true)
    verbose::Bool = true
    ndiv::Int = 100
    function CounterfactualParams(
        generator_params,
        n_individuals,
        n_runs,
        conv,
        maxiter,
        vertical_splits,
        store_ce,
        parallelizer,
        threaded,
        concatenate_output,
        verbose,
        ndiv
    )
        if generator_params isa NamedTuple
            if haskey(generator_params, :type) && generator_params.type isa String
                generator_type = get_generator_type(generator_params.type)
                generator_params = @delete $generator_params.type          # remove type
                generator_params = GeneratorParams(;
                    type=generator_type(), generator_params...
                )
            else
                generator_params = GeneratorParams(; generator_params...)
            end
        end

        return new(
            generator_params,
            n_individuals,
            n_runs,
            conv,
            maxiter,
            vertical_splits,
            store_ce,
            parallelizer,
            threaded,
            concatenate_output,
            verbose,
            ndiv
        )
    end
end

function get_parallelizer(cfg::CounterfactualParams)
    return get_parallelizer(cfg.parallelizer; threaded=cfg.threaded)
end

get_convergence(cfg::CounterfactualParams) = get_convergence(cfg.conv, cfg.maxiter)

"""
    evaluate_counterfactuals(
        cfg::AbstractEvaluationConfig,
        data::CounterfactualData,
        models::AbstractDict,
        generators::AbstractDict;
        measure::Vector{<:PenaltyOrFun}=get_ce_measures(),
    )

Evaluate the counterfactuals using the provided `models` and `generators`.

# Arguments

- `cfg`: The evaluation configuration.
- `data`: The counterfactual data containing the generated counterfactuals.
- `models`: A dictionary of models, where keys are model names and values are model objects.
- `generators`: A dictionary of generators, where keys are generator names and values are generator objects.
- `measure`: An array of penalty or function measures to evaluate.

# Returns

- A dataframe containing the evaluation results for each counterfactual.
"""
function evaluate_counterfactuals(
    cfg::AbstractEvaluationConfig,
    data::CounterfactualData,
    models::AbstractDict,
    generators::AbstractDict;
    measure::Vector{<:PenaltyOrFun}=get_ce_measures(),
)

    grid = ExperimentGrid(cfg.grid_file)
    exper_list = load_list(grid)

    # Get parallelizer:
    pllr = get_parallelizer(cfg.counterfactual_params)
    conv = get_convergence(cfg.counterfactual_params)
    interim_storage_path = interim_ce_path(cfg)
    vertical_splits = if cfg.counterfactual_params.vertical_splits == 0
        nothing
    else
        cfg.counterfactual_params.vertical_splits
    end

    if cfg.counterfactual_params.store_ce == true || CounterfactualExplanations.Evaluation.includes_divergence_metric(measure)
        @warn "Setting `_ce_transform` to `flatten` to avoid storing entire `CounterfactualExplanation` object."
        transformer = ExplicitCETransformer(CounterfactualExplanations.flatten)
        global_ce_transform(transformer)
    end

    # Generate and benchmark counterfactuals:
    rng = get_data_set(cfg)() |> get_rng
    bmk =
        benchmark(
            data;
            models=models,
            generators=generators,
            measure=measure,
            parallelizer=pllr,
            suppress_training=true,
            initialization=:identity,
            n_individuals=cfg.counterfactual_params.n_individuals,
            n_runs=cfg.counterfactual_params.n_runs,
            convergence=conv,
            store_ce=cfg.counterfactual_params.store_ce,
            storage_path=interim_storage_path,
            vertical_splits=vertical_splits,
            concatenate_output=cfg.counterfactual_params.concatenate_output,
            verbose=cfg.counterfactual_params.verbose,
        ) 
    if Evaluation.includes_divergence_metric(measure)
        bmk = compute_divergence(bmk, measure, data; rng=rng, nsamples=cfg.counterfactual_params.ndiv)
    end

    rm(interim_storage_path; recursive=true)
    
    # Remove counterfactuals to save memory:
    if !cfg.counterfactual_params.store_ce && "ce" ∈ names(bmk.evaluation)
        @info "Removing counterfactuals from evaluation"
        df = select(bmk.evaluation, Not(:ce))
        return Benchmark(df)
    else
        return bmk
    end
    
end

"""
    interim_ce_path(cfg::AbstractEvaluationConfig)

If path to store interim counterfactual data is not set, it creates a new directory at `save_dir`/interim_counterfactuals. Returns the path to this directory.
"""
function interim_ce_path(cfg::AbstractEvaluationConfig)
    return mkpath(joinpath(cfg.save_dir, "interim_counterfactuals"))
end

"""
    evaluate_counterfactuals(
        cfg::AbstractEvaluationConfig;
        measure::Vector{<:PenaltyOrFun}=get_ce_measures(),
    )

Generate and evaluate counterfactuals based on the provided configuration. This function generates counterfactuals and evaluates them using the specified measures. It returns a DataFrame containing the results of the evaluation.

# Arguments
- `cfg::AbstractEvaluationConfig`: Configuration for evaluation.
- `measure::Vector{<:PenaltyOrFun}=get_ce_measures()`: Measures to evaluate the counterfactuals with.

# Returns
- A DataFrame containing the results of the evaluation.
"""
function evaluate_counterfactuals(
    cfg::AbstractEvaluationConfig; measure::Vector{<:PenaltyOrFun}=get_ce_measures()
)
    data, models, generators = load_data_models_generators(cfg)

    # Evaluate counterfactuals:
    bmk = evaluate_counterfactuals(cfg, data, models, generators; measure=measure)

    return bmk
end

"""
    evaluate_counterfactuals(
        cfg::AbstractEvaluationConfig,
        comm::MPI.Comm;
        measure::Vector{<:PenaltyOrFun}=get_ce_measures(),
    )

Generate and evaluate counterfactuals based on the provided configuration. This method of `evaluate_counterfactuals` is dispatched for parallel evaluation using MPI (`comm::MPI.Comm`). The generation and evaluation of counterfactuals are distributed across the MPI processes.

# Arguments
- `cfg::AbstractEvaluationConfig`: Configuration for evaluation.
- `comm::MPI.Comm`: MPI communicator.
- `measure::Vector{<:PenaltyOrFun}=get_ce_measures()`: Measures to evaluate the counterfactuals with.

# Returns
- A DataFrame containing the results of the evaluation. 
"""
function evaluate_counterfactuals(
    cfg::AbstractEvaluationConfig,
    comm::MPI.Comm;
    measure::Vector{<:PenaltyOrFun}=get_ce_measures(),
)

    # Initialize MPI:
    rank = MPI.Comm_rank(comm)
    _size = MPI.Comm_size(comm)

    # Root process loads all data
    if rank == 0
        data, all_models, generators = load_data_models_generators(cfg)
        # Convert models dict to array for distribution
        model_keys = collect(keys(all_models))
        model_values = collect(values(all_models))
    else
        data = nothing
        all_models = nothing
        generators = nothing
        model_keys = nothing
        model_values = nothing
    end

    # Broadcast data and generators to all processes
    data = MPI.bcast(data, 0, comm)
    generators = MPI.bcast(generators, 0, comm)

    # Distribute models across processes
    if rank == 0
        # Calculate chunks for each process
        chunks_keys = TaijaParallel.split_obs(model_keys, _size)
        chunks_values = TaijaParallel.split_obs(model_values, _size)
    else
        chunks_keys = nothing
        chunks_values = nothing
    end

    # Scatter model chunks to processes
    local_keys = MPI.scatter(chunks_keys, comm; root=0)
    local_values = MPI.scatter(chunks_values, comm; root=0)

    # Create local models dictionary
    local_models = Dict(zip(local_keys, local_values))

    # Evaluate counterfactuals on local models
    local_results = evaluate_counterfactuals(
        cfg, data, local_models, generators; measure=measure
    )

    MPI.Barrier(comm)  # Ensure all processes have completed local evaluation before gathering results

    # Gather results from all processes (if needed)
    if cfg.counterfactual_params.concatenate_output
        # Gather results from all processes
        all_results = MPI.gather(local_results, comm; root=0)

        # Combine results on root process
        if rank == 0
            combined_results = reduce(vcat, all_results)
            return combined_results
        end
    else
        @info "Not concatenating results as configured."
    end

    return nothing
end

"""
    load_data_models_generators(cfg::AbstractEvaluationConfig)

Load data, models, and generators from the configuration.
"""
function load_data_models_generators(cfg::AbstractEvaluationConfig)

    # Load grid and experiment list:
    grid = ExperimentGrid(cfg.grid_file)
    exper_list = load_list(grid)

    # Get data:
    data = get_ce_data(cfg)

    # Get models:
    models = Logging.with_logger(Logging.NullLogger()) do
        Dict(
            [
                exper.meta_params.experiment_name => load_results(exper)[3] for
                exper in exper_list
            ]...,
        )
    end

    # Counterfactual generators:
    gen_params = cfg.counterfactual_params.generator_params
    _gen_name = get_generator_name(gen_params)
    _generator = get_generator(gen_params)
    generators = Dict(_gen_name => _generator)

    return data, models, generators
end

"""
    collect_benchmarks(cfg::AbstractEvaluationConfig)

Uses the `Evaluation.get_benchmark_files` function to collect all benchmarks from the specified storage path.
"""
function collect_benchmarks(cfg::AbstractEvaluationConfig; kwrgs...)
    if isfile(default_bmk_name(cfg))
        @info "Benchmark file $(default_bmk_name(cfg)) already exists. Skipping."
        bmk = Serialization.deserialize(default_bmk_name(cfg))
        return collect_benchmarks(cfg, bmk; kwrgs...)
    end

    bmk_files = Evaluation.get_benchmark_files(interim_ce_path(cfg))
    bmks = Vector{Benchmark}(undef, length(bmk_files))

    Threads.@threads for i in eachindex(bmk_files)
        bmks[i] = Serialization.deserialize(bmk_files[i])
    end
    bmk = reduce(vcat, bmks)

    return collect_benchmarks(cfg, bmk; kwrgs...)
end

"""
    collect_benchmarks(
        cfg::AbstractEvaluationConfig,
        bmk::Benchmark;
        save_bmk::Bool=true,
        remove_interim::Bool=true,
    )

Saves the `Benchmark` object to disk if requested.
"""
function collect_benchmarks(
    cfg::AbstractEvaluationConfig,
    bmk::Benchmark;
    save_bmk::Bool=true,
    remove_interim::Bool=true,
)

    # Save results to file if requested
    if save_bmk && !isfile(default_bmk_name(cfg))
        @info "Saving benchmark results ..."
        save_results(cfg, bmk.evaluation, default_ce_evaluation_name(cfg))
        save_results(cfg, bmk)
    end

    # Remove interim files if requested
    if remove_interim
        @info "Removing interim files ..."
        rm(interim_ce_path(cfg); recursive=true)
    end

    return bmk
end

"""
    save_results(
        cfg::AbstractEvaluationConfig, bmk::Benchmark; fname::Union{Nothing,String}=nothing
    )

Saves the `Benchmark` object to disk using `Serialization.serialize`. The location is specified by `cfg.save_dir`, unless `fname` is provided.
"""
function save_results(
    cfg::AbstractEvaluationConfig, bmk::Benchmark; fname::Union{Nothing,String}=nothing
)
    fname = if isnothing(fname)
        default_bmk_name(cfg)
    else
        fname
    end
    return Serialization.serialize(fname, bmk)
end

"""
    load_results(cfg::AbstractEvaluationConfig, fname::String)

Loads the benchmark. 
"""
function load_results(
    cfg::AbstractEvaluationConfig, bmk::Type{Benchmark}, fname::String=default_bmk_name(cfg)
)
    return Serialization.deserialize(fname)
end

"""
    load_ce_evaluation(
        cfg::AbstractEvaluationConfig; fname::Union{Nothing,String}=nothing
    )

Load the counterfactual evaluation results.
"""
function load_ce_evaluation(
    cfg::AbstractEvaluationConfig; fname::Union{Nothing,String}=nothing
)
    fname = if isnothing(fname)
        default_bmk_name(cfg)
    else
        fname
    end

    bmk = load_results(cfg, Benchmark, fname)
    df = bmk.evaluation
    if "model" in names(df) && !("id" in names(df))
        rename!(df, :model => :id)
    end

    return df
end

default_bmk_name(cfg::AbstractEvaluationConfig) = joinpath(cfg.save_dir, "bmk.jls")

default_ce_evaluation_name(cfg::AbstractEvaluationConfig) = "bmk_evaluation"

"""
    generate_factual_target_pairs(cfg::AbstractEvaluationConfig)

Dispatches the `generate_factual_target_pairs` on `cfg::AbstractEvaluationConfig`. The data, models and generators are loaded according the configuration `cfg`.
"""
function generate_factual_target_pairs(
    cfg::AbstractEvaluationConfig;
    fname::Union{Nothing,String}=nothing,
    overwrite::Bool=false,
)
    fname = if isnothing(fname)
        default_factual_target_pairs_name(cfg)
    end
    if isfile(fname) && !overwrite
        @info "Loading factual target pairs from $fname ..."
        output = load_results(cfg, Benchmark, fname)
    else
        data, models, generators = load_data_models_generators(cfg)
        output = generate_factual_target_pairs(cfg, data, models, generators)
        @info "Saving results to $fname."
        save_results(cfg, output; fname=fname)
    end
    return output
end

function default_factual_target_pairs_name(cfg::AbstractEvaluationConfig)
    return joinpath(cfg.save_dir, "factual_target_pairs.jls")
end

"""
    generate_factual_target_pairs(
        cfg::AbstractEvaluationConfig,
        data::CounterfactualData,
        models::AbstractDict,
        generators::AbstractDict,
    )

Generates counterfactuals for each target and factual pair. The function randomly chooses a sample in the factual class based on the `data` and uses it for all `models` and `generators` to allow for comparisons between models and generators.
"""
function generate_factual_target_pairs(
    cfg::AbstractEvaluationConfig,
    data::CounterfactualData,
    models::AbstractDict,
    generators::AbstractDict,
)

    # Targets and factuals:
    targets = sort(data.y_levels)
    factuals = targets

    # Store only counterfactual, not whole CE object:
    @warn "Setting `_ce_transform` to `counterfactual` to avoid storing entire `CounterfactualExplanation` object."
    transformer = ExplicitCETransformer(CounterfactualExplanations.counterfactual)
    global_ce_transform(transformer)

    output = Benchmark[]

    for factual in factuals
        chosen = rand(findall(data.output_encoder.labels .== factual))
        x = select_factual(data, chosen)
        for target in targets
            if cfg.counterfactual_params.verbose
                @info "Generating counterfactual for $(factual) -> $(target)"
            end

            # If factual is equal to target, don't search:
            if factual == target
                convergence = MaxIterConvergence(0)
            else
                convergence = get_convergence(cfg.counterfactual_params)
            end

            # Generate and benchmark counterfactuals:
            bmk = benchmark(
                x,
                target,
                data;
                models=models,
                generators=generators,
                measure=validity,
                parallelizer=nothing,
                initialization=:identity,
                convergence=convergence,
                store_ce=true,
                verbose=cfg.counterfactual_params.verbose,
            )

            DataFrames.transform!(bmk.evaluation, :model => ByRow(x -> x[1]) => :model)
            DataFrames.transform!(
                bmk.evaluation, :generator => ByRow(x -> x[1]) => :generator
            )

            # Add factual values:
            bmk.counterfactuals.x .= [x]

            push!(output, bmk)
        end
    end

    output = reduce(vcat, output)

    return output
end
