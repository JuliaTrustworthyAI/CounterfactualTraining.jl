const default_axis = (; width=225, height=225)

const default_mnist = (; width=150, height=150)

const default_ce = (; width=400, height=400)

const default_facet = (; linkyaxes=:minimal, linkxaxes=:minimal)

Base.@kwdef struct PlotParams
    x::Union{Nothing,String} = nothing
    byvars::Union{Nothing,String,Vector{String}} = nothing
    colorvar::Union{Nothing,String} = get_global_param("colorvar", nothing)
    rowvar::Union{Nothing,String} = get_global_param("rowvar", nothing)
    colvar::Union{Nothing,String} = get_global_param("colvar", nothing)
end

function (params::PlotParams)()
    return (;
        x=params.x,
        byvars=params.byvars,
        colorvar=params.colorvar,
        rowvar=params.rowvar,
        colvar=params.colvar,
    )
end

"""
    plot_errorbar_logs(
        cfg::EvaluationConfig;
        y::String="acc_val",
        byvars::Union{Nothing,String,Vector{String}}=nothing,
        colorvar::Union{Nothing,String}=nothing,
        rowvar::Union{Nothing,String}=nothing,
        colvar::Union{Nothing,String}=nothing,
        facet=(; linkyaxes=:minimal, linkxaxes=:minimal),
        axis=(width=225, height=225),
    )

Plot error bars for aggregated logs from an experiment grid.

## Arguments

- `cfg::EvalConfigOrGridg`: Configuration object containing grid file path and other settings.
- `y`: Variable in the logs to aggregate and plot (default: "acc_val").
- `byvars`: Columns to group data by for aggregation (default: nothing).
- `colorvar`, `rowvar`, `colvar`: Variables to use as color, row, and column facets respectively.
- `facet`: Facet options for the plot (default: `(; linkyaxes=:minimal, linkxaxes=:minimal)`.
- `axis`: Axis options for the plot (default: `(width=225, height=225)`).

## Returns

A Makie plot object displaying error bars for aggregated logs.
"""
function plot_errorbar_logs(
    cfg::EvalConfigOrGrid;
    x::Nothing=nothing,
    y::String="acc_val",
    byvars::Union{Nothing,String,Vector{String}}=nothing,
    colorvar::Union{Nothing,String}=nothing,
    rowvar::Union{Nothing,String}=nothing,
    colvar::Union{Nothing,String}="generator_type",
    facet=default_facet,
    axis=default_axis,
)
    byvars = gather_byvars(byvars, colorvar, rowvar, colvar)

    # Aggregate logs:
    df_agg = aggregate_logs(cfg; y=y, byvars=byvars)
    use_line_plot = all(isnan.(df_agg.std))

    # Plotting:
    plt = data(df_agg)
    if use_line_plot
        plt = plt * mapping(:epoch => "Epoch", :mean => "Value") * visual(Lines)
    else
        plt = plt * mapping(:epoch => "Epoch", :mean => "Value", :std) * visual(Errorbars)
    end
    if !isnothing(colorvar)
        plt = plt * mapping(; color=colorvar => nonnumeric)
    end
    if !isnothing(rowvar)
        plt = plt * mapping(; row=rowvar => nonnumeric)
    end
    if !isnothing(colvar)
        plt = plt * mapping(; col=colvar => nonnumeric)
    end

    plt = draw(plt; facet=facet, axis=axis)
    return plt, df_agg
end

function boxplot_ce(
    df::DataFrame,
    df_meta::DataFrame,
    df_eval::DataFrame;
    x::Union{Nothing,String}="generator_type",
    y::String="plausibility_distance_from_target",
    byvars::Union{Nothing,String,Vector{String}}=nothing,
    kwrgs...
)
    x = isnothing(x) ? "generator_type" : x

    byvars = gather_byvars(byvars, colorvar, rowvar, colvar, x)

    # Aggregate:
    df_agg = aggregate_ce_evaluation(df, df_meta, df_eval; y=y, byvars=byvars)

    # Plotting:
    plt = boxplot_ce(df_agg,x;kwrgs...)

    return plt, df_agg
end

function boxplot_ce(
    df_agg::DataFrame,
    x::Union{Nothing,String}="generator_type";
    colorvar::Union{Nothing,String}=nothing,
    rowvar::Union{Nothing,String}=nothing,
    colvar::Union{Nothing,String}=nothing,
    facet=default_facet,
    axis=default_axis,
)
    # Plotting:
    plt = data(df_agg) * mapping(Symbol(x), :mean => "Value") * visual(BoxPlot)
    if !isnothing(colorvar)
        plt = plt * mapping(; color=colorvar => nonnumeric)
    end
    if !isnothing(rowvar)
        plt = plt * mapping(; row=rowvar => nonnumeric)
    end
    if !isnothing(colvar)
        plt = plt * mapping(; col=colvar => nonnumeric)
    end

    plt = draw(plt; facet=facet, axis=axis)

    return plt
end

function plot_ce(cfg::EvalConfigOrGrid; save_dir=nothing, kwrgs...)
    return plot_ce(CTExperiments.get_data_set(cfg)(), cfg; save_dir=save_dir, kwrgs...)
end

function plot_ce(dataset::Dataset, eval_grid::EvaluationGrid; save_dir=nothing, kwrgs...)
    eval_list = load_list(eval_grid)
    plt_list = []

    for eval_config in eval_list
        if !isnothing(save_dir)
            local_save_dir = mkpath(
                joinpath(save_dir, splitpath(eval_config.save_dir)[end])
            )
        else
            local_save_dir = nothing
        end
        plt = plot_ce(dataset, eval_config; save_dir=local_save_dir, kwrgs...)
        push!(plt_list, plt)
    end
    return plt_list
end

function plot_ce(
    dataset::Dataset,
    eval_config::EvaluationConfig;
    byvars::Union{Nothing,String,Vector{String}}=nothing,
    axis=default_axis,
    dpi=300,
    save_dir=nothing,
)

    # Aggregate:
    df_agg = aggregate_counterfactuals(eval_config; byvars=byvars)

    # Plotting:
    generators = sort(unique(df_agg.generator_type))
    factuals = sort(unique(df_agg.factual))
    targets = sort(unique(df_agg.target))

    if isa(byvars, String)
        byvars = [byvars]
    end

    if isnothing(byvars)
        byvars = [byvars]
    else
        plot_dict = Dict(k => Dict() for k in byvars)
    end

    # Plotting:
    for variable in byvars
        vals = if isnothing(variable)
            [nothing]
        else
            sort(unique(df_agg[!, variable]))
        end
        for val in vals
            full_plts = _plot_over_generators(
                dataset,
                generators,
                factuals,
                targets,
                df_agg,
                variable,
                val;
                save_dir,
                axis,
                dpi,
            )
            if isnothing(val)
                plot_dict = full_plts
            else
                plot_dict[variable][val] = full_plts
            end
        end
    end

    return plot_dict
end

function _plot_over_generators(
    dataset,
    generators,
    factuals,
    targets,
    df_agg,
    variable,
    val;
    save_dir=nothing,
    axis=default_axis,
    dpi=300,
)
    full_plts = Dict()
    for (i, generator) in enumerate(generators)
        if !isnothing(variable)
            df_local = df_agg[
                df_agg.generator_type .== generator .&& [
                    x == val for x in df_agg[!, variable]
                ],
                :,
            ]
        else
            df_local = df_agg[df_agg.generator_type .== generator, :]
        end
        plts = []
        for factual in factuals
            for target in targets
                plt = plot_ce(dataset, df_local, factual, target)
                push!(plts, plt)
            end
        end
        full_plt = Plots.plot(
            plts...;
            layout=(length(factuals), length(targets)),
            size=(values(axis)[1] * length(targets), values(axis)[2] * length(factuals)),
            dpi=dpi,
        )
        if !isnothing(save_dir)
            fname = if isnothing(variable)
                "ce_$(generator).png"
            else
                "ce_$(generator)_$(variable)=$(val).png"
            end
            Plots.savefig(full_plt, joinpath(save_dir, fname))
        end
        full_plts[generator] = full_plt
    end
    return full_plts
end

function plot_ce(data::MNIST, df::DataFrame, factual::Int, target::Int; axis=default_mnist)
    x = filter(x -> x.factual .== factual && x.target .== target, df).mean
    if length(x) == 0
        plt = Plots.plot(; axis=([], false), size=values(axis))
    else
        @assert length(x) == 1 "Expected 1 value, got $(length(x))."
        x = x[1]
        if target == factual
            title = "Factual"
            blue = true
        else
            title = "$factual→$target"
            blue = false
        end
        plt = Plots.plot(
            convert2mnist(x; blue=blue); axis=([], false), size=values(axis), title=title
        )
    end
    return plt
end

function plot_ce(data::Dataset, df::DataFrame, factual::Int, target::Int; axis=default_ce)
    data = get_ce_data(data, 200)
    plt = Plots.plot(data)
    x = filter(x -> x.factual .== factual && x.target .== target, df).mean
    if length(x) > 0
        @assert length(x) == 1 "Expected 1 value, got $(length(x))."
        x = x[1]
        @assert size(x, 1) == 2 "Can only plot 2-D data."
        if target == factual
            title = ""
            Plots.scatter!(
                plt,
                [x[1]],
                [x[2]];
                label="Factual",
                size=values(axis),
                title=title,
                ms=5,
                color=:yellow,
            )
        else
            title = "$factual→$target"
            Plots.scatter!(
                plt,
                [x[1]],
                [x[2]];
                label="Counterfactual",
                size=values(axis),
                title=title,
                ms=10,
                shape=:star,
                color=:yellow,
            )
        end
    end
    return plt
end

"""
    plot_ce(
        exper::Experiment,
        params::Union{CounterfactualParams,TrainingParams};
        target::Union{Nothing,Int}=nothing,
        nsamples::Int=9,
    )

Generate and plot counterfactual explanations for a given experiment and target class.
"""
function plot_ce(
    exper::Experiment,
    params::Union{CounterfactualParams,TrainingParams};
    target::Union{Nothing,Int}=nothing,
    nsamples::Int=25,
    kwrgs...,
)
    M = load_results(exper)[3]
    generator = CTExperiments.get_generator(params.generator_params)
    conv = CTExperiments.get_convergence(params)
    data = get_ce_data(exper.data)
    if isnothing(target)
        @info "No target supplied, choosing first label."
        target = data.y_levels[1]
    end

    # Cannot plot counterfactuals for OmnisionGenerator:
    if isa(generator, OmniscientGenerator)
        return Plots.plot(M, data; size=(500, 500), cb=false, target=target, kwrgs...)
    end

    candidates = findall(predict_label(M, data) .!= target)
    idx = rand(candidates, 1)
    x = collect(select_factual(data, idx))[1]
    ce = generate_counterfactual(
        x, target, data, M, generator; convergence=conv, num_counterfactuals=nsamples
    )
    @info "Generator: $(generator)"
    return Plots.plot(
        ce;
        size=(default_axis.width, default_axis.height),
        cb=false,
        target=target,
        kwrgs...,
    )
end

"""
    plot_ce(exper::Experiment; kwargs...)

Generate and plot counterfactual explanations using the counterfactual generator used during training.
"""
function plot_ce(exper::Experiment; kwargs...)
    return plot_ce(exper, exper.training_params; kwargs...)
end

"""
    plot_ce(
        exper::Experiment,
        eval_cfg::Union{Nothing,EvaluationConfig};
        kwrgs...
    )

Generate and plot counterfactual explanations using the counterfactual generator specified in the evaluation configuration. For `eval_cfg=nothing`, the function uses the default parameters from the training configuration.
"""
function plot_ce(exper::Experiment, eval_cfg::Union{Nothing,EvaluationConfig}; kwrgs...)
    if !isnothing(eval_cfg)
        params = eval_cfg.counterfactual_params
        return plot_ce(exper, params; kwrgs...)
    else
        return plot_ce(exper; kwrgs...)
    end
end

"""
    plot_ce(exper_list::Vector{Experiment}; layout=length(exper_list), kwargs...)

Generate and plot counterfactual explanations for a list of experiments and arrange them in a grid layout.
"""
function plot_ce(
    exper_list::Vector{Experiment},
    eval_cfg::Union{Nothing,EvaluationConfig}=nothing;
    layout=length(exper_list),
    titles=nothing,
    kwargs...,
)
    if isnothing(titles)
        titles = []
        for exper in exper_list
            gen = exper.meta_params.generator_type
            obj = exper.training_params.objective
            title = "$(exper.meta_params.experiment_name)\n$(gen) ($obj)"
            push!(titles, title)
        end
    end

    plts = []
    for (i, exper) in enumerate(exper_list)
        plt = plot_ce(
            exper, eval_cfg; title=titles[i], topmargin=5PlotMeasures.mm, kwargs...
        )
        push!(plts, plt)
    end

    if isa(layout, Tuple)
        _rows, _cols = layout
    else
        _rows, _cols = (ceil(sqrt(layout)), floor(sqrt(layout)))
    end

    w, h = (_cols * 2default_axis.width, _rows * 2default_axis.height)

    return Plots.plot(plts...; layout=layout, size=(w, h))
end