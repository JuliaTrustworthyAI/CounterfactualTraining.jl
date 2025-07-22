using CounterfactualExplanations
using CounterfactualExplanations.Evaluation
using DataFrames
using IntegratedGradients
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
    n_runs::Int = get_global_param("n_runs", 5)
    conv::AbstractString = "threshold"
    decision_threshold::AbstractFloat = get_global_param("tau_eval", 0.95)
    maxiter::Int = get_global_param("maxiter_eval", 50)
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
        decision_threshold,
        maxiter,
        vertical_splits,
        store_ce,
        parallelizer,
        threaded,
        concatenate_output,
        verbose,
        ndiv,
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
            decision_threshold,
            maxiter,
            vertical_splits,
            store_ce,
            parallelizer,
            threaded,
            concatenate_output,
            verbose,
            ndiv,
        )
    end
end

function get_parallelizer(cfg::CounterfactualParams)
    return get_parallelizer(cfg.parallelizer; threaded=cfg.threaded)
end

function get_convergence(cfg::CounterfactualParams)
    return get_convergence(cfg.conv, cfg.maxiter, cfg.decision_threshold)
end

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

    # Get parallelizer:
    pllr = get_parallelizer(cfg.counterfactual_params)
    conv = get_convergence(cfg.counterfactual_params)
    interim_storage_path = interim_ce_path(cfg)
    vertical_splits = if cfg.counterfactual_params.vertical_splits == 0
        nothing
    else
        cfg.counterfactual_params.vertical_splits
    end

    if cfg.counterfactual_params.store_ce == true ||
        CounterfactualExplanations.Evaluation.includes_divergence_metric(measure)
        @warn "Setting `_ce_transform` to `flatten` to avoid storing entire `CounterfactualExplanation` object."
        transformer = ExplicitCETransformer(CounterfactualExplanations.flatten)
        global_ce_transform(transformer)
    end

    # Generate and benchmark counterfactuals:
    bmk = benchmark(
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
        rng = get_data_set(cfg)() |> get_rng    # ensure that same test set columns are chosen for MMD (for reproducibility)
        bmk = compute_divergence(
            bmk, measure, data; rng=rng, nsamples=cfg.counterfactual_params.ndiv
        )
    end

    # In case `feature_sensitivity` was computed, store index of feature:
    add_sensitivity!(bmk)

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
    input_list = load_data_models_generators(cfg)

    # Evaluate counterfactuals:
    bmk = Benchmark[]
    for (data, models, generators) in input_list
        _bmk = evaluate_counterfactuals(cfg, data, models, generators; measure=measure)
        push!(bmk, _bmk)
    end
    bmk = reduce(vcat, bmk)

    return bmk
end

"""
    load_data_models_generators(cfg::AbstractEvaluationConfig)

Load data, models, and generators from the configuration.
"""
function load_data_models_generators(cfg::AbstractEvaluationConfig)

    # Load grid and experiment list:
    grid = ExperimentGrid(cfg.grid_file)
    exper_list = load_list(grid)

    # Counterfactual generators (same for each exper):
    gen_params = cfg.counterfactual_params.generator_params
    _gen_name = get_name(gen_params)
    _generator = get_generator(gen_params)
    generators = Dict(_gen_name => _generator)

    # Get data:
    datasets = [exper.data for exper in exper_list]
    unique_dataset = unique(x -> CTExperiments.to_dict(x), datasets)

    out_list = Tuple[]

    for dataset in unique_dataset
        exper_with_this_dataset = [
            exper for exper in exper_list if
            CTExperiments.to_dict(exper.data) == CTExperiments.to_dict(dataset)
        ]

        # Data:
        data = get_ce_data(dataset; test_set=cfg.test_time)

        # Get models:
        models = Logging.with_logger(Logging.NullLogger()) do
            Dict(
                [
                    exper.meta_params.experiment_name => load_results(exper)[3] for
                    exper in exper_with_this_dataset
                ]...,
            )
        end

        push!(out_list, (data, models, generators))
    end

    return out_list
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
    nce::Int=1,
)
    fname = if isnothing(fname)
        default_factual_target_pairs_name(cfg)
    end
    if isfile(fname) && !overwrite
        @info "Loading factual target pairs from $fname ..."
        output = load_results(cfg, Benchmark, fname)
    else
        input_list = load_data_models_generators(cfg)
        output = Benchmark[]
        for (data, models, generators) in input_list
            _output = generate_factual_target_pairs(cfg, data, models, generators; nce=nce)
            push!(output, _output)
        end
        output = reduce(vcat, output)
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
    generators::AbstractDict;
    nce::Int=1,
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
        chosen = rand(findall(data.output_encoder.labels .== factual), nce)
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


using MPI
using Random

# Main function with optional comm parameter
function integrated_gradients(cfg::Experiment; n=1000, test_set=true, max_entropy::Bool=true, 
                            nrounds::Int=10, verbose::Bool=false, baseline_type="random", 
                            comm::Union{Nothing,MPI.Comm}=nothing, kwrgs...)
    
    model, _, _ = load_results(cfg)
    X, y = get_data(cfg.data; n, test_set)

    # Bootstrap:
    idx = rand(1:size(X,2),n)
    X = X[:,idx]
    y = y[idx]
    
    return integrated_gradients(model, X, y, comm; max_entropy, nrounds, verbose, baseline_type, kwrgs...)
end

# Sequential version (original implementation)
function integrated_gradients(model, X, y, comm::Nothing; max_entropy::Bool=true, 
                            nrounds::Int=10, verbose::Bool=false, baseline_type="random", kwrgs...)
    
    igs = Matrix{<:AbstractFloat}[]
    for i in 1:nrounds
        if verbose
            @info "Round $i/$nrounds"
        end
        if max_entropy
            x = randn(size(X,1),1)  # random starting point (Gaussian init)
            if verbose
                @info "Computing maximum entropy baseline ..."
            end
            bl = IntegratedGradients.get_maximum_entropy_baseline(model, x)[1]
            ig = calculate_average_contributions(model, X, y; baseline=bl, kwrgs...)
        else
            ig = calculate_average_contributions(model, X, y; baseline_type, kwrgs...)
        end
        push!(igs, ig)
    end
    return igs
end

# MPI distributed version
function integrated_gradients(model, X, y, comm::MPI.Comm; max_entropy::Bool=true, 
                            nrounds::Int=10, verbose::Bool=false, baseline_type="random", kwrgs...)
    
    rank = MPI.Comm_rank(comm)
    size_comm = MPI.Comm_size(comm)
    
    # Distribute rounds across processes
    rounds_per_proc = div(nrounds, size_comm)
    remainder = nrounds % size_comm
    
    # Calculate which rounds this process will handle
    start_round = rank * rounds_per_proc + 1 + min(rank, remainder)
    end_round = start_round + rounds_per_proc - 1 + (rank < remainder ? 1 : 0)
    local_nrounds = end_round - start_round + 1
    
    if verbose && rank == 0
        @info "Distributing $nrounds rounds across $size_comm processes"
    end
    
    # Each process computes its assigned rounds
    local_igs = Matrix{<:AbstractFloat}[]
    for i in 1:local_nrounds
        global_round = start_round + i - 1
        if verbose
            @info "Process $rank: Round $global_round/$nrounds (local: $i/$local_nrounds)"
        end
        
        if max_entropy
            # Use a seed based on global round number for reproducibility
            Random.seed!(global_round)
            x = randn(size(X,1),1)  # random starting point (Gaussian init)
            if verbose
                @info "Process $rank: Computing maximum entropy baseline for round $global_round..."
            end
            bl = IntegratedGradients.get_maximum_entropy_baseline(model, x)[1]
            ig = calculate_average_contributions(model, X, y; baseline=bl, kwrgs...)
        else
            ig = calculate_average_contributions(model, X, y; baseline_type, kwrgs...)
        end
        push!(local_igs, ig)
    end
    
    # Gather all results on root process
    all_igs = MPI.gather(local_igs, comm; root=0)
    
    if rank == 0
        # Reconstruct results in correct order
        igs = Matrix{<:AbstractFloat}[]
        
        # Create a vector to hold results in correct order
        ordered_results = Vector{Matrix{<:AbstractFloat}}(undef, nrounds)
        
        # Place each process's results in the correct positions
        for proc_rank in 0:(size_comm-1)
            proc_rounds_per_proc = div(nrounds, size_comm)
            proc_remainder = nrounds % size_comm
            proc_start = proc_rank * proc_rounds_per_proc + 1 + min(proc_rank, proc_remainder)
            
            for (local_idx, ig) in enumerate(all_igs[proc_rank + 1])
                global_idx = proc_start + local_idx - 1
                ordered_results[global_idx] = ig
            end
        end
        
        if verbose
            @info "Collected and ordered results from all processes"
        end
        
        return ordered_results
    else
        return nothing  # Non-root processes return nothing
    end
end
