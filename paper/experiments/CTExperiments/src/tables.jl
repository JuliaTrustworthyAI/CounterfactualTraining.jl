using CategoricalArrays
using ColorSchemes
using DataFrames
using PrettyTables
using StatsBase

function tabulate_results(
    inputs;
    tf=PrettyTables.tf_latex_booktabs,
    wrap_table=true,
    table_type=:longtable,
    longtable_footer="Continuing table below.",
    save_name::Union{String,Nothing}=nothing,
    formatters=PrettyTables.ft_round(2),
    alignment=:c,
    wrap_table_environment="table*",
    sort_by_data::Bool=true,
    hlines::Union{Nothing,Symbol}=nothing,
    kwrgs...,
)
    df = inputs[1]

    # Group averages:
    if "data" in names(df) && hlines != :none
        if "Avg." in df.data
            _ds_order = [ds_order..., "Avg."]
            hlines = [0, 1, 5, length(ds_order)+1, length(ds_order)+2]
        else
            _ds_order = ds_order
            hlines = [0, 1, 5, length(ds_order)+2]
        end
        df.data = categorical(df.data; levels=_ds_order)
        if sort_by_data
            sort!(df, :data)
        end
    end

    if hlines == :none
    hlines = nothing
    end

    other_inputs = inputs[2]
    if isa(other_inputs.backend, Val{:latex})
        tf = PrettyTables.tf_latex_booktabs
        if isnothing(save_name)
            tab = pretty_table(
                df;
                tf=tf,
                formatters=formatters,
                wrap_table=wrap_table,
                wrap_table_environment=wrap_table_environment,
                table_type=table_type,
                longtable_footer=longtable_footer,
                alignment,
                hlines,
                other_inputs...,
                kwrgs...,
            )
            return tab
        else
            open(save_name, "w") do io
                pretty_table(
                    io,
                    df;
                    tf=tf,
                    formatters=formatters,
                    wrap_table=wrap_table,
                    wrap_table_environment=wrap_table_environment,
                    table_type=table_type,
                    longtable_footer=longtable_footer,
                    alignment,
                    hlines,
                    other_inputs...,
                    kwrgs...,
                )
            end
        end
    else
        if isnothing(save_name)
            tab = pretty_table(df; alignment, hlines, other_inputs..., kwrgs...)
            return tab
        else
            open(save_name, "w") do io
                pretty_table(io, df; alignment, hlines, other_inputs..., kwrgs...)
            end
        end
    end
end

function get_table_inputs(
    df::DataFrame,
    value_var::Union{Nothing,String}="mean";
    backend::Val=Val(:text),
    kwrgs...,
)
    df = deepcopy(df)

    if !isnothing(value_var)
        # Highlighters:
        hls = value_highlighter(df, value_var; backend=backend, kwrgs...)
        if "generator_type" in names(df)
            # Filter out "omni":
            df = df[df.generator_type .!= "omni", :]
            # df.generator_type = format_generator.(df.generator_type)
            gen_hl = generator_highlighter(df; backend=backend)
            hls = (hls..., gen_hl)
        end
    else
        hls = ()
    end

    hs = multi_row_header(names(df))
    header =
        [format_header.(h; replacements=LatexHeaderReplacements) for h in hs] |> x -> tuple(x...)
    return df, (; highlighters=hls, backend=backend, header=header)
end

format_generator(s::AbstractString) = get_name(generator_types[s](); pretty=true)

global LatexHeaderReplacements = merge(
    Dict(
        "lambda_energy_exper" => LatexCell("\$\\lambda_{\\text{div}} (\\text{train})\$"),
        "lambda_energy_eval" => LatexCell("\$\\lambda_{\\text{div}} (\\text{eval})\$"),
        "lambda_cost_exper" => LatexCell("\$\\lambda_{\\text{cost}} (\\text{train})\$"),
        "lambda_cost_eval" => LatexCell("\$\\lambda_{\\text{cost}} (\\text{eval})\$"),
    ),
    Dict(k => LatexCell(v) for (k, v) in CTExperiments.LatexReplacements),
)

function value_highlighter(
    df::DataFrame,
    value_var::String="mean";
    scheme::Union{Nothing,ColorSchemes.ColorScheme}=nothing,
    alpha::AbstractFloat=0.5,
    backend::Val=Val(:text),
    byvars::Union{Nothing,String,Vector{String}}=nothing,
)
    @assert value_var in names(df) "Provided variables $(value_var) is not in the dataframe"
    col_idx = findall(value_var .== names(df))

    hls = []

    # Maximum value:
    if !isnothing(byvars)
        byvars = isa(byvars, String) ? [byvars] : byvars
        df.row .= 1:nrow(df)
        max_idx = combine(groupby(df, byvars)) do sdf
            (max_idx=sdf.row[argmax(sdf[:, value_var])],)
        end
        max_idx = max_idx.max_idx
        select!(df, Not(:row))
        hl = bolden_max_hl(max_idx, col_idx, backend, value_var)
        push!(hls, hl)
        hl_bad = bolden_max_hl_bad(max_idx, col_idx, backend, value_var)
        push!(hls, hl_bad)
    end

    # Color scale:
    if !isnothing(scheme)
        lb, ub = (alpha / 2, 1 - alpha / 2)
        lims = (quantile(df[:, value_var], lb), quantile(df[:, value_var], ub))
        hl = color_scale_hl(lims, col_idx, backend, scheme)
        push!(hls, hl)
    end

    return tuple(hls...)
end

function color_scale_hl(
    lims::Tuple, col_idx::Vector{Int}, backend::Val{:text}, scheme::ColorScheme
)
    hl = Highlighter(
        (df, i, j) -> j in col_idx,
        (h, df, i, j) -> begin
            color = get(scheme, df[i, j], lims)
            return Crayon(;
                foreground=(
                    round(Int, color.r * 255),
                    round(Int, color.g * 255),
                    round(Int, color.b * 255),
                ),
            )
        end,
    )
    return hl
end

function bolden_max_hl(max_idx::Vector{Int}, col_idx::Vector{Int}, backend::Val{:text})
    hl = Highlighter((df, i, j) -> i in max_idx, crayon"blue bold")
    return hl
end

function color_scale_hl(
    lims::Tuple, col_idx::Vector{Int}, backend::Val{:latex}, scheme::ColorScheme
)
    hl = LatexHighlighter(
        (df, i, j) -> j in col_idx,
        (df, i, j, s) -> begin
            color = get(scheme, df[i, j], lims)
            return "\\color[rgb]{$(color.r), $(color.g), $(color.b)}{$s}"
        end,
    )
    return hl
end

function bolden_max_hl(
    max_idx::Vector{Int}, col_idx::Vector{Int}, backend::Val{:latex}, value_var
)
    return hl = LatexHighlighter(
        (df, i, j) -> (i in max_idx) && df[i, value_var] > 0, ["color{Green}", "textbf"]
    )
end

function bolden_max_hl_bad(
    max_idx::Vector{Int}, col_idx::Vector{Int}, backend::Val{:latex}, value_var
)
    return hl = LatexHighlighter(
        (df, i, j) -> (i in max_idx) && df[i, value_var] < 0, ["color{Red}", "textbf"]
    )
end

function generator_highlighter(df::DataFrame; backend::Val=Val(:text))
    @assert "generator_type" in names(df) "Dataframe does not include column for generator type"
    col_idx = findall("generator_type" .== names(df))[1]
    return generator_highlighter(col_idx, backend)
end

function generator_highlighter(col_idx::Int, backend::Val{:text})
    hl = Highlighter((df, i, j) -> j == col_idx, crayon"italics")
    return hl
end

function generator_highlighter(col_idx::Int, backend::Val{:latex})
    hl = LatexHighlighter((df, i, j) -> j == col_idx, "textit")
    return hl
end
