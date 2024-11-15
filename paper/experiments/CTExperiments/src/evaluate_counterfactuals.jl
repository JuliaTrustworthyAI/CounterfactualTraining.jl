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

function evaluate_counterfactuals(
    cfg::AbstractEvaluationConfig; measure::Vector{<:PenaltyOrFun}=CE_MEASURES
)
    grid = ExperimentGrid(cfg.grid_file)
    exper_list = load_list(grid)

    # Get all available test data:
    dataset = (
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
        dataset;
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
