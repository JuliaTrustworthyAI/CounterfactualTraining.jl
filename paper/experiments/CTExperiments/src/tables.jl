using ColorSchemes
using DataFrames
using PrettyTables
using StatsBase

function tabulate_results(
    df::DataFrame,
    value_var::String="mean";
    backend::Val=Val(:text),
    kwrgs...
)
    df = deepcopy(df)

    # Highlighters:
    hls = (value_highlighter(df, value_var; backend=backend, kwrgs...),)
    if "generator_type" in names(df)
        df.generator_type = format_generator.(df.generator_type)
        gen_hl = generator_highlighter(df; backend=backend)
        hls = (hls..., gen_hl)
    end

    header = format_header.(names(df))
    return pretty_table(df; highlighters=hls, backend=backend, header=header)
end

format_generator(s::AbstractString) = get_generator_name(generator_types[s](), pretty=true)

function format_header(s::String)
    s = split(s, "_") |>
        ss -> [s in ["exper", "eval"] ? "($s)" : uppercasefirst(s) for s in ss] |>
        ss -> join(ss, " ")
    return s
end

function value_highlighter(
    df::DataFrame,
    value_var::String="mean";
    scheme::ColorSchemes.ColorScheme=reverse(colorschemes[:rose]),
    alpha::AbstractFloat=0.5,
    backend::Val=Val(:text)
)
    @assert value_var in names(df) "Provided variables $(value_var) is no in the dataframe"
    col_idx = findall(value_var .== names(df))
    lb, ub = (alpha/2, 1 - alpha/2)
    @info "Lower bound $(lb), Upper bound $(ub)"
    lims = (quantile(df[:, value_var], lb), quantile(df[:, value_var], ub))
    @info "Limits: $(lims)"

    return value_highlighter(lims, col_idx, backend, scheme)
end

function value_highlighter(
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

function value_highlighter(
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