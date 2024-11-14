"""
    CounterfactualParams

Struct holding keyword arguments relevant to the evaluation of counterfactual explanations for fitted models.
"""
Base.@kwdef struct CounterfactualParams <: AbstractConfiguration
    generators::Vector{<:AbstractString} = String["ecco"]
    generator_params::Union{AbstractDict,NamedTuple} = (;)
    n_individuals::Int = 100
    n_runs::Int = 10
    conv::AbstractString = "max_iter"
    maxiter::Int = 100
    vertical_splits::Int = 0
    store_ce::Bool = false
    parallelizer::AbstractString = "threads"
    threaded::Bool = true
end

function get_parallelizer(cfg::CounterfactualParams)
    return get_parallelizer(cfg.parallelizer; threaded=cfg.threaded)
end

get_convergence(cfg::CounterfactualParams) = get_convergence(cfg.conv, cfg.maxiter)

function evaluate_counterfactuals(
    cfg::AbstractEvaluationConfig; measure::Vector{<:PenaltyOrFun}=[validity, plausibility]
)
    grid = from_toml(cfg.grid_file)
    exper_list = load_list(grid)

    # Get all available test data:
    dataset = (dataset_type ->
        (dt -> CounterfactualData(dt...))(get_test_data(dataset_type(); n=nothing)))(get_data(
        grid.data
    ))

    # Get models:
    models = Dict(
        [
            exper.meta_params.experiment_name => load_results(exper)[3] for
            exper in exper_list
        ]...,
    )

    # Counterfactual generators:
    generators = Dict{Symbol,AbstractGenerator}()
    for gen_name in cfg.counterfactual_params.generators
        generator_type = get_generator_type(gen_name)
        _gen_params = cfg.counterfactual_params.generator_params        # get key words
        generator_params = @delete $_gen_params.type                    # remove type
        generator_params = GeneratorParams(; type=generator_type(), generator_params...)
        generator[Symbol(gen_name)] = get_generator(generator_params)
    end

    pllr = get_parallelizer(cfg.counterfactual_params)
    conv = get_convergence(cfg.counterfactual_params)
    interim_storage_path = mkpath(joinpath(cfg.save_dir, "interim_counterfactuals"))

    # Generate and benchmark counterfactuals:
    return benchmark(
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
    )
end