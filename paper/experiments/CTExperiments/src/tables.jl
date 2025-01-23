using ColorSchemes
using DataFrames
using PrettyTables

function get_highlighter(
    df::DataFrame,
    value_var::String="mean";
    scheme::ColorSchemes.ColorScheme=colorschemes[:coolwarm],
)
    @assert value_var in names(df) "Provided variables $(value_var) is no in the dataframe"
    col_idx = findall(value_var .== names(df))[1]
    @info "Highlighting column $col_idx"

    hl = Highlighter(
        (df, i, j) -> j == col_idx,
        (h, df, i, j) -> begin
            color = get(scheme, df[i, j], (0, 20))
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

