using AlgebraOfGraphics
using DataFrames
using Makie

function aggregate_logs(
    exper_grid::ExperimentGrid;
    y::String="acc_val",
    byvars::Union{Nothing,Vector{String}}=nothing,
)
    # Load data:
    df_meta = CTExperiments.expand_grid_to_df(exper_grid)
    logs = get_logs(exper_grid)
    df = innerjoin(df_meta, logs; on=:id)

    # Assertions:
    valid_y = names(logs)[eltype.(eachcol(logs)) .<: AbstractFloat]
    @assert y in valid_y "Variable `y` must be one of the following: $valid_y."
    @assert byvars isa Nothing || all(col -> col in names(df_meta), byvars) "Columns specified in `byvars` must be one of the following: $(names(df_meta))."

    if !isnothing(byvars)
        # Aggregate data by columns specified in `byvars`:
        byvars = union(["epoch"], byvars)
        df_agg = groupby(df, byvars)
    else
        # If nothing is specified, aggregate data by epoch:
        df_agg = groupby(df, "epoch")
    end
    df_agg = combine(df_agg, y => (y -> (mean=mean(y), std=std(y))) => AsTable)

    return df_agg
end

function plot_errorbar_logs(
    exper_grid::ExperimentGrid;
    y::String="acc_val",
    byvars::Union{Nothing,Vector{String}}=nothing,
    colorvar::Union{Nothing,String}=nothing,
    rowvar::Union{Nothing,String}=nothing,
    colvar::Union{Nothing,String}=nothing,
    facet=(; linkyaxes=:minimal, linkxaxes=:minimal),
    axis=(width=225, height=225),
)

    # Aggregate logs:
    byvars = isnothing(byvars) ? [byvars] : byvars
    byvars = [byvars..., colorvar, rowvar, colvar] |> unique
    byvars = if length(byvars) == 1 && isnothing(byvars[1])
        nothing 
    else 
        byvars[.!isnothing.(byvars)] |> x -> string.(x)
    end
    df_agg = aggregate_logs(exper_grid; y=y, byvars=byvars)

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