"""
    CounterfactualParams

Struct holding keyword arguments relevant to the evaluation of counterfactual explanations for fitted models.
"""
Base.@kwdef struct CounterfactualParams <: AbstractConfiguration
    generator_params::GeneratorParams = GeneratorParams()
    n_individuals::Int = 100
    n_runs::Int = 10
    conv::AbstractString = "max_iter"
    maxiter::Int = 100
    vertical_splits::Int = 0
    store_ce::Bool = false
    parallelizer::AbstractString = "threads"
    threaded::Bool = true
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
    )
        if generator_params isa NamedTuple
            if generator_params.type isa String
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
        measure::Vector{<:PenaltyOrFun}=CE_MEASURES,
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
    measure::Vector{<:PenaltyOrFun}=CE_MEASURES,
)
    # Get parallelizer:
    pllr = get_parallelizer(cfg.counterfactual_params)
    conv = get_convergence(cfg.counterfactual_params)
    interim_storage_path = mkpath(joinpath(cfg.save_dir, "interim_counterfactuals"))
    vertical_splits = if cfg.counterfactual_params.vertical_splits == 0
        nothing
    else
        cfg.counterfactual_params.vertical_splits
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
    )()

    rename!(bmk, :model => :id)

    return bmk
end

"""
    evaluate_counterfactuals(
        cfg::AbstractEvaluationConfig;
        measure::Vector{<:PenaltyOrFun}=CE_MEASURES,
    )

Generate and evaluate counterfactuals based on the provided configuration. This function generates counterfactuals and evaluates them using the specified measures. It returns a DataFrame containing the results of the evaluation.

# Arguments
- `cfg::AbstractEvaluationConfig`: Configuration for evaluation.
- `measure::Vector{<:PenaltyOrFun}=CE_MEASURES`: Measures to evaluate the counterfactuals with.

# Returns
- A DataFrame containing the results of the evaluation.
"""
function evaluate_counterfactuals(
    cfg::AbstractEvaluationConfig;
    measure::Vector{<:PenaltyOrFun}=CE_MEASURES,
)
    
    data, models, generators = load_data_models_generators(cfg)

    # Evaluate counterfactuals:
    bmk = evaluate_counterfactuals(
        cfg,
        data,
        models,
        generators;
        measure=measure,
    )

    return bmk  
end

function evaluate_counterfactuals(
    cfg::AbstractEvaluationConfig,
    comm::MPI.Comm;
    measure::Vector{<:PenaltyOrFun}=CE_MEASURES,
)

    # Initialize MPI:
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

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
        chunk_size = ceil(Int, length(model_keys) / size)
        chunks_keys = [
            model_keys[i:min(i + chunk_size - 1, end)] for
            i in 1:chunk_size:length(model_keys)
        ]
        chunks_values = [
            model_values[i:min(i + chunk_size - 1, end)] for
            i in 1:chunk_size:length(model_values)
        ]
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

    # Gather results from all processes
    all_results = MPI.gather(local_results, comm; root=0)

    # Combine results on root process
    if rank == 0
        combined_results = vcat(all_results...)
        MPI.Finalize()
        return combined_results
    end

    MPI.Finalize()
    return nothing
end

function load_data_models_generators(cfg::AbstractEvaluationConfig)

    # Load grid and experiment list:
    grid = ExperimentGrid(cfg.grid_file)
    exper_list = load_list(grid)

    # Get all available test data:
    data = (
        dataset_type -> (dt -> CounterfactualData(dt...))(
            get_data(dataset_type(); n=nothing, test_set=cfg.test_time)
        )
    )(
        get_data_set(grid.data)
    )

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
