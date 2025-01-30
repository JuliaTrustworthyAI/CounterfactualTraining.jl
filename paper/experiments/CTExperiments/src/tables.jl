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
    kwrgs...,
)
    df = inputs[1]
    other_inputs = inputs[2]
    if isa(other_inputs.backend,Val{:latex})
        formatters = PrettyTables.ft_latex_sn(3)
        tf = PrettyTables.tf_latex_booktabs
        tab = pretty_table(
            df;
            tf=tf,
            formatters=formatters,
            wrap_table=wrap_table,
            table_type=table_type,
            longtable_footer=longtable_footer,
            alignment=:c,
            other_inputs...,
            kwrgs...,
        )
    else
        tab = pretty_table(
            df;
            alignment=:c,
            other_inputs...,
            kwrgs...,
        )
    end
    return tab
end

function get_table_inputs(
    df::DataFrame,
    value_var::String="mean";
    backend::Val=Val(:text),
    kwrgs...
)
    df = deepcopy(df)

    # Highlighters:
    hls = value_highlighter(df, value_var; backend=backend, kwrgs...)
    if "generator_type" in names(df)
        df.generator_type = format_generator.(df.generator_type)
        gen_hl = generator_highlighter(df; backend=backend)
        hls = (hls..., gen_hl)
    end

    header = format_header.(names(df); replacements=LatexHeaderReplacements)
    return df, (; highlighters=hls, backend=backend, header=header)
end

format_generator(s::AbstractString) = get_generator_name(generator_types[s](), pretty=true)

global LatexHeaderReplacements = Dict(
    "lambda_energy_exper" =>  latex_cell"$\lambda_{\text{div}} (\text{train})$",
    "lambda_energy_eval" =>  latex_cell"$\lambda_{\text{div}} (\text{eval})$",
    "lambda_cost_exper" =>  latex_cell"$\lambda_{\text{cost}} (\text{train})$",
    "lambda_cost_eval" =>  latex_cell"$\lambda_{\text{cost}} (\text{eval})$",
)

function value_highlighter(
    df::DataFrame,
    value_var::String="mean";
    scheme::Union{Nothing,ColorSchemes.ColorScheme}=nothing,
    alpha::AbstractFloat=0.5,
    backend::Val=Val(:text),
    byvars::Union{Nothing,String,Vector{String}}=nothing,
)
    @assert value_var in names(df) "Provided variables $(value_var) is no in the dataframe"
    col_idx = findall(value_var .== names(df))

    hls = []

    # Maximum value:
    if !isnothing(byvars)
        byvars = isa(byvars, String) ? [byvars] : byvars
        df.row .= 1:nrow(df)
        max_idx = combine(groupby(df, byvars...)) do sdf
            (max_idx=sdf.row[argmax(sdf[:, value_var])],)
        end
        max_idx = max_idx.max_idx
        select!(df, Not(:row))
        hl = bolden_max_hl(max_idx, col_idx, backend)
        push!(hls, hl)
    end

    # Color scale:
    if !isnothing(scheme)
        lb, ub = (alpha/2, 1 - alpha/2)
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

function bolden_max_hl(max_idx::Vector{Int},col_idx::Vector{Int},backend::Val{:text})
    hl = Highlighter(
        (df, i, j) -> i in max_idx,
        crayon"blue bold",
    )
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

function bolden_max_hl(max_idx::Vector{Int},col_idx::Vector{Int},backend::Val{:latex})
    hl = LatexHighlighter((df, i, j) -> i in max_idx, ["color{blue}", "textbf"])
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
    hl = LatexHighlighter((df, i, j) -> j == col_idx,"textit")
    return hl
end