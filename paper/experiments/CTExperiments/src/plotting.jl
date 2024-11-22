using AlgebraOfGraphics
using DataFrames
using Makie

const default_axis = (; width=225, height=225)

const default_facet = (; linkyaxes=:minimal, linkxaxes=:minimal)

get_logs(cfg::AbstractEvaluationConfig) = get_logs(ExperimentGrid(cfg.grid_file))

function merge_with_meta(cfg::EvaluationConfig, df::DataFrame)
    # Load data:
    exper_grid = ExperimentGrid(cfg.grid_file)
    df_meta = CTExperiments.expand_grid_to_df(exper_grid)
    if "model" in names(df) && !("id" in names(df))
        rename!(df, :model => :id)
    end
    return innerjoin(df_meta, df; on=:id), df_meta, df
end

function aggregate_data(
    df::DataFrame,
    y::String,
    byvars::Union{Nothing,Vector{String}}=nothing;
    byvars_must_include=Union{Nothing,Vector{String}},
)

    df = filter(row -> all(x -> !(x isa Number && (isnan(x) || isinf(x))), row), df)

    # Aggregate:
    if !isnothing(byvars)
        # Aggregate data by columns specified in `byvars`:
        byvars = union(byvars_must_include, byvars)
        df_agg = groupby(df, byvars)
    else
        # If nothing is specified, aggregate data by epoch:
        df_agg = groupby(df, byvars_must_include)
    end
    df_agg = combine(df_agg, y => (y -> (mean=mean(y), std=std(y))) => AsTable)
    return sort(df_agg)
end

"""
    aggregate_logs(
        cfg::EvaluationConfig;
        y::String="acc_val",
        byvars::Union{Nothing,Vector{String}}=nothing,
    )

Aggregate logs variable `y` from an experiment grid by columns specified in `byvars`.
"""
function aggregate_logs(
    cfg::EvaluationConfig;
    kwrgs...,
)

    # Load data:
    df, df_meta, logs = merge_with_meta(cfg, get_logs(cfg))

    aggregate_logs(df, df_meta, logs; kwrgs...)
end

"""
    aggregate_logs(
        df::DataFrame; y::String, byvars::Union{Nothing,Vector{String}}=nothing
    )

Aggregate data in `df` variable `y` from an experiment grid by columns specified in `byvars`.
"""
function aggregate_logs(
    df::DataFrame,
    df_meta::DataFrame,
    logs::DataFrame;
    y::String,
    byvars::Union{Nothing,Vector{String}}=nothing,
)
    # Assertions:
    valid_y = names(logs)[eltype.(eachcol(logs)) .<: AbstractFloat]
    @assert y in valid_y "Variable `y` must be one of the following: $valid_y."
    @assert byvars isa Nothing || all(col -> col in names(df_meta), byvars) "Columns specified in `byvars` must be one of the following: $(names(df_meta))."

    # Aggregate data:
    return aggregate_data(df, y, byvars; byvars_must_include=["epoch"])
end

"""
    plot_errorbar_logs(
        cfg::EvaluationConfig;
        y::String="acc_val",
        byvars::Union{Nothing,Vector{String}}=nothing,
        colorvar::Union{Nothing,String}=nothing,
        rowvar::Union{Nothing,String}=nothing,
        colvar::Union{Nothing,String}=nothing,
        facet=(; linkyaxes=:minimal, linkxaxes=:minimal),
        axis=(width=225, height=225),
    )

Plot error bars for aggregated logs from an experiment grid.

## Arguments

- `cfg::EvaluationConfig`: Configuration object containing grid file path and other settings.
- `y`: Variable in the logs to aggregate and plot (default: "acc_val").
- `byvars`: Columns to group data by for aggregation (default: nothing).
- `colorvar`, `rowvar`, `colvar`: Variables to use as color, row, and column facets respectively.
- `facet`: Facet options for the plot (default: `(; linkyaxes=:minimal, linkxaxes=:minimal)`.
- `axis`: Axis options for the plot (default: `(width=225, height=225)`).

## Returns

A Makie plot object displaying error bars for aggregated logs.
"""
function plot_errorbar_logs(
    cfg::EvaluationConfig;
    y::String="acc_val",
    byvars::Union{Nothing,Vector{String}}=nothing,
    colorvar::Union{Nothing,String}=nothing,
    rowvar::Union{Nothing,String}=nothing,
    colvar::Union{Nothing,String}="generator_type",
    facet=default_facet,
    axis=default_axis,
)

    byvars = gather_byvars(byvars, colorvar, rowvar, colvar)

    # Aggregate logs:
    df_agg = aggregate_logs(cfg; y=y, byvars=byvars)

    # Plotting:
    plt = data(df_agg) *
        mapping(:epoch => "Epoch", :mean => "Value", :std) *
        visual(Errorbars) 
    if !isnothing(colorvar)
        plt = plt * mapping(; color=colorvar => nonnumeric)
    end
    if !isnothing(rowvar)
        plt = plt * mapping(; row=rowvar => nonnumeric)
    end
    if !isnothing(colvar)
        plt = plt * mapping(; col=colvar => nonnumeric)
    end
    
    return draw(plt; facet=facet, axis=axis)
end

function gather_byvars(
    byvars,
    args...
)
    byvars = isnothing(byvars) ? [byvars] : byvars
    byvars = unique([byvars..., args...])
    return byvars = if length(byvars) == 1 && isnothing(byvars[1])
        nothing
    else
        (x -> string.(x))(byvars[.!isnothing.(byvars)])
    end

    return byvars
end

function aggregate_ce_evaluation(
    cfg::EvaluationConfig;
    kwrgs...
)

    # Load data:
    all_data = merge_with_meta(cfg, CTExperiments.load_ce_evaluation(cfg))

    return aggregate_ce_evaluation(all_data...; kwrgs...)
    
end

function aggregate_ce_evaluation(
    df::DataFrame, 
    df_meta::DataFrame,
    df_eval::DataFrame;
    y::String="plausibility_distance_from_target",
    byvars::Union{Nothing,Vector{String}}=nothing,
)
    # Assertions:
    valid_y = sort(unique(df.variable))
    @assert y in valid_y "Variable `y` must be one of the following: $valid_y."
    @assert byvars isa Nothing || all(col -> col in names(df_meta), byvars) "Columns specified in `byvars` must be one of the following: $(names(df_meta))."

    df = df[df.variable .== y,:]
    rename!(df, :value => y)
    select!(df, Not(:variable))

    # Aggregate:
    return aggregate_data(df, y, byvars; byvars_must_include=["run"])
end

function boxplot_ce(
    df::DataFrame, 
    df_meta::DataFrame,
    df_eval::DataFrame;
    x::String="generator_type",
    y::String="plausibility_distance_from_target",
    byvars::Union{Nothing,Vector{String}}=nothing,
    colorvar::Union{Nothing,String}=nothing,
    rowvar::Union{Nothing,String}=nothing,
    colvar::Union{Nothing,String}=nothing,
    facet=default_facet,
    axis=default_axis,
) 

    byvars = gather_byvars(byvars, colorvar, rowvar, colvar, x)

    # Aggregate:
    df_agg = aggregate_ce_evaluation(df, df_meta, df_eval; y=y, byvars=byvars)
    
    # Plotting:
    plt = data(df_agg) *
        mapping(Symbol(x), :mean => "Value") *
        visual(BoxPlot)
    if !isnothing(colorvar)
        plt = plt * mapping(; color=colorvar => nonnumeric)
    end
    if !isnothing(rowvar)
        plt = plt * mapping(; row=rowvar => nonnumeric)
    end
    if !isnothing(colvar)
        plt = plt * mapping(; col=colvar => nonnumeric)
    end
    
    return draw(plt; facet=facet, axis=axis)
end