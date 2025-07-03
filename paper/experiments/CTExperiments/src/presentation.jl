using AlgebraOfGraphics
using CounterfactualExplanations
using CairoMakie
using DataFrames
using Plots: Plots, PlotMeasures
using PrettyTables
using TaijaPlotting

# Global parameters:
global _colorvar = nothing
global _colvar = nothing
global _rowvar = nothing
global _rowvar_ce = nothing
global _colvar_ce = nothing
global _byvars_ce = nothing
global _lnstyvar = nothing
global _dodgevar = nothing
global _sidevar = nothing

# Function to check if a value is effectively empty
function is_empty_value(v::Any)
    if v isa String
        return isempty(v)
    elseif v isa Vector
        return isempty(v) || length(v) == 1
    elseif v isa Dict
        # A dictionary is empty if it's empty itself or if all its filtered values would be empty
        filtered = filter_dict(v)
        return isempty(filtered)
    else
        return false
    end
end

# Function to filter dictionary
function filter_dict(
    dict::Dict;
    drop_fields=[
        "name",
        "data",
        "data_params",
        "experiment_name",
        "dim_reduction",
        "nneighbours",
        "grid_file",
    ],
    filter_empty::Bool=true,
)

    # Take care of nested dicts:
    for (k, v) in dict
        if v isa Dict
            dict[k] = filter_dict(v; drop_fields, filter_empty)
        end
    end

    if filter_empty
        # Filter out empty values and specified fields
        dict = filter(dict) do (k, v)
            !is_empty_value(v) && !(k in drop_fields)
        end
    else
        # Filter out empty values and specified fields
        dict = filter(dict) do (k, v)
            !(k in drop_fields)
        end
    end

    return dict
end

global LatexReplacements = Dict(
    "lambda_energy" => "\$\\lambda_{\\text{egy}}\$",
    "lambda_cost" => "\$\\lambda_{\\text{cst}}\$",
    "lambda_adversarial" => "\$\\lambda_{\\text{adv}}\$",
    "lambda_energy_diff" => "\$\\lambda_{\\text{div}}\$",
    "lambda_energy_reg" => "\$\\lambda_{\\text{reg}}\$",
    "lambda_class_loss" => "\$\\lambda_{\\text{clf}}\$",
    "gmsc" => get_name(GMSC(); pretty=true),
    "lin_sep" => "LS",
    "over" => "OL",
    "cali" => "CH",
    "mnist" => get_name(MNIST(); pretty=true),
    "circles" => "Circ",
    "moons" => "Moon",
    "credit" => "Cred",
)

function split_at_parentheses_precise(s::String)
    # This regex looks for either:
    # 1. Standard parentheses: (...)
    # 2. LaTeX math expressions: $\(...\)$
    m = match(r"(.*)\s+((?:\$\\?\([^)]*\\?\)\$)|(?:\([^)]*\)).*$)", s)

    if m === nothing
        return (s, "")  # No match found
    else
        return m.captures[1], m.captures[2]  # Return the two parts
    end
end

function swap_legy(s::Vector{String})
    if !any(contains.(s, "\\lambda_{\\text{egy}}"))
        return s
    end
    idx = contains.(s, r"\$ \\lambda_{\\text{egy}}=[^$]*\$")
    s[idx] = [
        replace(si, r"\$ \\lambda_{\\text{egy}}=[^$]*\$" => match(r"\d*\.?\d+", si).match)
        for si in s[idx]
    ]
    s[1] = "$(uppercasefirst(s[1])) \\\\ \$ \\lambda_{\\text{egy}} \$"
    return s
end

function multi_row_header(s::Vector{String})
    s = swap_legy(s)

    if !any(contains.(s, r" \\\\ "))
        return [s]
    else
        h1 = String[]
        h2 = String[]
        for si in s
            if contains(si, "\\\\")
                h1i, h2i = split(si, " \\\\ ")
            else
                h1i = si
                h2i = ""
            end
            push!(h1, string(h1i))
            push!(h2, string(h2i))
        end
    end

    if !any(contains.(h1, r"\([^)]*\)"))
        return [h1, h2]
    else
        h3 = h2
        s = h1
        h1 = []
        h2 = []
        for si in s
            if contains(si, r"\([^)]*\)")
                h1i, h2i = split_at_parentheses_precise(si)
            else
                h1i = si
                h2i = ""
            end
            push!(h1, string(h1i))
            push!(h2, string(h2i))
        end
        return [h1, h2, h3]
    end
end

function format_header(s::String; replacements::Dict=LatexReplacements)
    if contains(s, "\$")
        return LatexCell(s)
    end
    s =
        replace(s, r"\bnce\b" => "ncounterfactuals") |>
        s ->
            replace(s, "_exper" => "") |>
            s ->
                replace(s, "_eval" => "") |>
                s ->
                    replace(s, "_type" => "") |>
                    s ->
                        replace(s, "_params" => "_parameters") |>
                        s ->
                            replace(s, "lr" => "learning_rate") |>
                            s ->
                                replace(s, "maxiter" => "maximum_iterations") |>
                                s ->
                                    replace(s, "opt" => "optimizer") |>
                                    s ->
                                        replace(s, "conv" => "convergence") |>
                                        s ->
                                            replace(s, r"\bopt\b" => "optimizer") |>
                                            s ->
                                                replace(s, r"^n" => "no._") |>
                                                s ->
                                                    replace(s, "__" => "_") |>
                                                    s -> if s in keys(replacements)
                                                        replacements[s]
                                                    else
                                                        s |>
                                                        s ->
                                                            split(s, "_") |>
                                                            ss ->
                                                                [
                                                                    uppercasefirst(s)
                                                                    for s in ss
                                                                ] |> ss -> join(ss, " ")
                                                    end
    return s
end

global _drop_fields = [
    "name",
    "concatenate_output",
    "parallelizer",
    "store_ce",
    "threaded",
    "verbose",
    "vertical_splits",
    "grid_file",
    "inherit",
    "save_dir",
    "test_time",
    "ndiv",
]

function to_mkd(
    dict::Dict,
    level::Int=0;
    header::Union{Nothing,String}=nothing,
    drop_fields=_drop_fields,
)
    dict = filter(((k, v),) -> length(v) > 0 && !(k in drop_fields), dict)

    # Create indent string based on level
    indent = repeat("    ", level)

    # Initialize array to store markdown lines
    if isnothing(header)
        lines = String[]
    else
        header = "\n*$header*\n"
        lines = [header]
    end

    # Sort dictionary keys for consistent output
    for key in sort(collect(keys(dict)))
        value = dict[key]
        key = format_header(key; replacements=LatexReplacements)

        if value isa Dict
            # Handle nested dictionary
            push!(lines, "$(indent)- $(key):")
            # Recursively process nested dictionary with increased indentation
            nested_lines = to_mkd(value, level + 1)
            push!(lines, nested_lines)
        elseif value isa Vector
            # Handle vector values by joining with commas
            value_str = join(value, ", ")
            push!(lines, "$(indent)- $(key): `$(value_str)`")
        else
            # Handle single values
            push!(lines, "$(indent)- $(key): `$(value)`")
        end
    end

    # Join all lines with newlines
    return join(lines, "\n")
end

# Function to create final Markdown string
function dict_to_markdown(
    dict::Dict; header::Union{Nothing,String}=nothing, filter_empty::Bool=true
)
    filtered_dict = filter_dict(dict; filter_empty)
    return "md\"\"\"\n$(to_mkd(filtered_dict; header=header))\n\"\"\""
end

# New function specifically for Quarto output
function dict_to_quarto_markdown(
    dict::Dict; header::Union{Nothing,String}=nothing, filter_empty::Bool=true
)
    filtered_dict = filter_dict(dict; filter_empty)
    return "$(to_mkd(filtered_dict; header=header))\n"
end

function adjust_plot_var(x::Union{Nothing,String}, cfg::CTExperiments.EvalConfigOrGrid)
    if isnothing(x)
        return x
    else
        if typeof(cfg) == EvaluationConfig
            x = replace(x, "_exper" => "")
            return x
        else
            return x
        end
    end
end

include("plots.jl")
include("tables.jl")

export tabulate_results, get_table_inputs

function useful_byvars(df_meta::DataFrame)
    return names(df_meta)[findall([length(unique(x)) != 1 for x in eachcol(df_meta)])]
end

function save_dir(params::PlotParams, root::String; prefix::String)
    suffix = (x -> (x -> join(x, "---"))(x[.!isnothing.(x)]))([
        isnothing(v) ? nothing : "$(v)" for (k, v) in pairs(params())
    ])
    return mkpath(joinpath(root, prefix, suffix))
end

"""
    get_logs(cfg::EvalConfigOrGrid)

Get the logs of an experiment grid pertaining to either a single evaluation or a grid of evaluations.
"""
get_logs(cfg::EvalConfigOrGrid) = get_logs(ExperimentGrid(cfg.grid_file))

"""
    merge_with_meta(cfg::EvaluationConfig, df::DataFrame)

Merge the metadata of a CTExperiments experiment grid with the data frame.
"""
function merge_with_meta(cfg::EvaluationConfig, df::DataFrame)
    # Load data:
    exper_grid = ExperimentGrid(cfg.grid_file)
    df_meta = CTExperiments.expand_grid_to_df(exper_grid)

    df_merged = innerjoin(df_meta, df; on=:id)
    select!(df_merged, :id, Not(:id))
    return df_merged, df_meta, df
end

"""
    merge_with_meta(grid::EvaluationGrid, df::DataFrame)

Merge the metadata of a CTExperiments experiment grid with the data frame.
"""
function merge_with_meta(grid::EvaluationGrid, df::DataFrame)
    # Load data:
    exper_grid = ExperimentGrid(grid.grid_file)
    df_meta_exper = CTExperiments.expand_grid_to_df(exper_grid)

    if "evaluation" in names(df)
        df_meta_eval = CTExperiments.expand_grid_to_df(grid; name_prefix="evaluation")
        rename!(df_meta_eval, :id => :evaluation)

        # Merge meta data:
        isduplicate(s) = s in names(df_meta_eval) && s in names(df_meta_exper)
        df_meta = crossjoin(
            df_meta_eval,
            df_meta_exper;
            renamecols=(x -> isduplicate(x) ? "$(x)_eval" : x) =>
                (x -> isduplicate(x) ? "$(x)_exper" : x),
        )
        select!(df_meta, :evaluation, :id, Not(:evaluation, :id))

        df_merged = innerjoin(df_meta, df; on=[:id, :evaluation])
        select!(df_merged, :evaluation, :id, Not(:evaluation, :id))
    else
        df_meta = df_meta_exper
        df_merged = innerjoin(df_meta, df; on=:id)
        select!(df_merged, :id, Not(:id))
    end
    return df_merged, df_meta, df
end

function aggregate_data(
    df::DataFrame,
    y::String,
    byvars::Union{Nothing,String,Vector{String}}=nothing;
    byvars_must_include::Union{Nothing,Vector{String}}=nothing,
    agg_fun::Union{Symbol,Function}=mean,
)

    # Filter:
    df = filter(row -> all(x -> !(x isa Number && (isinf(x))), row), df)
    df = filter(row -> all(x -> !(isnothing(x)), row), df)
    keep_rows = [!any(isnan.(x)) for x in df[:, y]]
    df = df[keep_rows, :]
    if "value" in names(df)
        keep_rows = [!any(isnan.(x)) for x in df.value]
        df = df[keep_rows, :]
    end

    # Aggregate:
    if !isnothing(byvars)
        if isa(byvars, String)
            byvars = [byvars]
        end

        # Aggregate data by columns specified in `byvars`:
        if !isnothing(byvars_must_include)
            byvars = union(byvars_must_include, byvars)
        end
        df_agg = groupby(df, byvars)
    else
        # If nothing is specified, aggregate data by epoch:
        df_agg = groupby(df, byvars_must_include)
    end
    if agg_fun == :identity
        df_agg = combine(df_agg, y => (y -> (mean=y, se=y)) => AsTable)
        return df_agg
    else
        df_agg = combine(df_agg, y => (y -> (mean=agg_fun(y), se=std(y)/sqrt(length(y)))) => AsTable)
        return sort(df_agg)
    end
end

"""
    aggregate_performance(
        eval_grids::Vector{EvalConfigOrGrid}; byvars=["objective"], kwrgs...
    )   

Aggregate performance measures across multiple experiments.
"""
function aggregate_performance(
    eval_grids::Vector{<:EvalConfigOrGrid}; byvars=["objective"], kwrgs...
)
    @assert "objective" in byvars "Need to specify 'objective' as a byvar to aggregate performance measures."

    df = Logging.with_logger(Logging.NullLogger()) do
        map(eval_grids) do cfg
            df = CTExperiments.aggregate_performance(cfg; byvars=byvars, kwrgs...)
            exper_grid = ExperimentGrid(cfg.grid_file)
            df.dataset .= CTExperiments.format_header(exper_grid.data)
            df.objective .= CTExperiments.format_header.(df.objective)
            keyvars = [:dataset, :variable, :objective]
            select!(df, keyvars, Not(keyvars))
            sort!(df, keyvars)
            return df
        end |> dfs -> reduce(vcat, dfs)
    end
    return df
end

"""
    aggregate_performance(cfg::EvalConfigOrGrid; kwrgs...)

Aggregate performance variable `y` from an experiment grid by columns specified in `byvars`.
"""
function aggregate_performance(
    cfg::EvalConfigOrGrid;
    measure=[accuracy, multiclass_f1score],
    adversarial::Bool=false,
    bootstrap::Union{Nothing,Int}=nothing,
    kwrgs...,
)

    # Load data:
    exper_grid = ExperimentGrid(cfg.grid_file)
    df, df_meta, df_perf = merge_with_meta(
        cfg,
        CTExperiments.test_performance(exper_grid; measure, adversarial, bootstrap, return_df=true),
    )

    return aggregate_performance(df, df_meta, df_perf; kwrgs...)
end

function aggregate_performance(
    df::DataFrame,
    df_meta::DataFrame,
    df_perf::DataFrame;
    y::Union{Nothing,String}=nothing,
    byvars::Union{Nothing,String,Vector{String}}=nothing,
    bootstrap::Union{Nothing,Int}=nothing,
)
    # Assertions:
    if isa(byvars, String)
        byvars = [byvars]
    end
    @assert byvars isa Nothing || all(col -> col in names(df_meta), byvars) "Columns specified in `byvars` must be one of the following: $(names(df_meta))."


    # Aggregate data:
    df = aggregate_data(df, "value", byvars; byvars_must_include=["variable"])
    if !isnothing(y)
        valid_y = valid_y_perf(df_perf)
        @assert y in valid_y "Variable `y` must be one of the following: $valid_y."
        df = df[df.variable .== y, :]
        return select!(df, Not(:variable))
    else
        return df
    end
end

function valid_y_perf(df::DataFrame)
    return sort(unique(df.variable))
end

"""
    aggregate_logs(
        cfg::EvaluationConfig;
        y::String="acc_val",
        byvars::Union{Nothing,String,Vector{String}}=nothing,
    )

Aggregate logs variable `y` from an experiment grid by columns specified in `byvars`.
"""
function aggregate_logs(cfg::EvalConfigOrGrid; kwrgs...)

    # Load data:
    df, df_meta, logs = merge_with_meta(cfg, get_logs(cfg))

    return aggregate_logs(df, df_meta, logs; kwrgs...)
end

"""
    aggregate_logs(
        df::DataFrame; y::String, byvars::Union{Nothing,String,Vector{String}}=nothing
    )

Aggregate data in `df` variable `y` from an experiment grid by columns specified in `byvars`.
"""
function aggregate_logs(
    df::DataFrame,
    df_meta::DataFrame,
    logs::DataFrame;
    y::String,
    byvars::Union{Nothing,String,Vector{String}}=nothing,
)
    # Assertions:
    valid_y = valid_y_logs(logs)
    @assert y in valid_y "Variable `y` must be one of the following: $valid_y."
    if isa(byvars, String)
        byvars = [byvars]
    end
    @assert byvars isa Nothing || all(col -> col in names(df_meta), byvars) "Columns specified in `byvars` must be one of the following: $(names(df_meta))."

    # Aggregate data:
    return aggregate_data(df, y, byvars; byvars_must_include=["epoch"])
end

function valid_y_logs(logs::DataFrame)
    return names(logs)[eltype.(eachcol(logs)) .<: Union{Nothing,AbstractFloat}]
end

function valid_y_logs(cfg::EvalConfigOrGrid)
    return valid_y_logs(get_logs(cfg))
end

function gather_byvars(byvars, args...)
    if isa(byvars, String)
        byvars = [byvars]
    end
    byvars = isnothing(byvars) ? [byvars] : byvars
    byvars = unique([byvars..., args...])
    return byvars = if length(byvars) == 1 && isnothing(byvars[1])
        nothing
    else
        (x -> string.(x))(byvars[.!isnothing.(byvars)])
    end

    return byvars
end

"""
    aggregate_ce_evaluation(
        cfg::EvaluationConfig;
        kwrgs...
    )

Aggregate the results from a single counterfactual evaluation.
"""
function aggregate_ce_evaluation(cfg::EvalConfigOrGrid; kwrgs...)

    # Load data:
    all_data = Logging.with_logger(Logging.NullLogger()) do
        merge_with_meta(cfg, CTExperiments.load_ce_evaluation(cfg))
    end

    return aggregate_ce_evaluation(all_data...; kwrgs...)
end

"""
    aggregate_ce_evaluation(
        df::DataFrame, 
        df_meta::DataFrame,
        df_eval::DataFrame;
        y::String="plausibility_distance_from_target",
        byvars::Union{Nothing,String,Vector{String}}=nothing,
    )

Aggregate the results from a single counterfactual evaluation.
"""
function aggregate_ce_evaluation(
    df::DataFrame,
    df_meta::DataFrame,
    df_eval::DataFrame=DataFrame();
    y::String="plausibility_distance_from_target",
    byvars::Union{Nothing,String,Vector{String}}=nothing,
    agg_further_vars::Union{Nothing,Vector{String}}=nothing,
    rebase::Bool=true,
    lambda_eval::Union{Nothing,Vector{<:Real}}=nothing,
    ratio::Bool=false,
    total_uncertainty::Bool=true,
)
    # Assertions:
    valid_y = valid_y_ce(df)
    @assert y in valid_y "Variable `y` must be one of the following: $valid_y."
    if isa(byvars, String)
        byvars = [byvars]
    end
    @assert isnothing(byvars) || all(col -> col in names(df_meta), byvars) "Columns specified in `byvars` must be one of the following: $(names(df_meta))."

    df = df[df.variable .== y, :]
    if y == "mmd"
        # To align with plausibility metric (negative distance), we need to invert the MMD metric. We also first clamp values to 0 (sometimes MMD is slightly negative for numeric reasons).
        df.value .= .-clamp.(df.value, 0, Inf)
    end
    rename!(df, :value => y)
    select!(df, Not(:variable))

    # Filter:
    if !isnothing(lambda_eval)
        @assert "lambda_energy_eval" in names(df) "Variables `lambda_energy_eval` not included."
        df = filter(df -> df.lambda_energy_eval in lambda_eval, df)
    end

    # Aggregate:
    if "run" in names(df)
        byvars_must_include = ["run", "lambda_energy_eval", "objective"]
        byvars_must_include = byvars_must_include[[
            x in names(df) for x in byvars_must_include
        ]]

        df_agg = aggregate_data(df, y, byvars; byvars_must_include=byvars_must_include)

        if !isnothing(agg_further_vars)
            # Compute mean of means and std of means:
            byvars =
                isnothing(byvars) ? byvars_must_include : union(byvars_must_include, byvars)
            filter!(x -> x ∉ agg_further_vars, byvars)

            # Computing across-fold averages and between fold standard errors for ratios:
            if ratio
                @assert sort(unique(df_agg.objective)) == ["full", "vanilla"] "Ratio calculation only works when comparing `full` vs. `vanilla`"
                
                if total_uncertainty
                    # Include uncertainty around lambda_energy_eval in standard error.
                    # This means that standard error also reflects possibly different hyperparameter
                    # choices at test time as a source of uncertainty.
                    select!(df_agg, Not(:se))
                else
                    # Otherwise, aggregate by all variables not including "run" and "objective".
                    # This means that uncertainty around lambda_energy_eval is marginalised out
                    # allowing for a clean model comparison.
                    bootstrap_vars = ["run", "objective"]
                    df_agg = groupby(df_agg, bootstrap_vars) |>
                        df -> combine(df, :mean => (y -> mean(skipmissing(y))) => :mean)
                    if "data" in names(df)
                        df_agg.data .= unique(df.data)
                    end
                end

                # Final aggregation and standard errors:
                df_agg = DataFrames.unstack(df_agg, :objective, :mean)
                df_agg.ratio .= df_agg.full ./ df_agg.vanilla
                byvars = setdiff(byvars, ["objective"])
                df_agg =
                    groupby(df_agg, byvars) |>
                    df -> combine(df, :ratio => (y -> (mean=-(mean(skipmissing(y))-1)*100, 
                        se=100*std(skipmissing(y))/sqrt(length(y)))) => AsTable)
                return df_agg
            end

            df_agg =
                groupby(df_agg, byvars) |>
                df -> combine(df, :mean => (y -> (mean=mean(y), se=std(y))) => AsTable)
        end
    else
        df_agg = aggregate_data(df, y, byvars)
    end

    # Subtract from Vanilla:
    if rebase
        @assert "objective" in names(df_agg) "Cannot rebase with respect to 'vanilla' objective is the 'objective' column is not present."
        objectives = unique(df_agg.objective)
        df_agg = DataFrames.unstack(df_agg[:, Not(:se)], :objective, :mean)
        vanilla_name = objectives[lowercase.(objectives) .== "vanilla"][1]
        other_names = objectives[lowercase.(objectives) .!= "vanilla"]
        for obj in other_names

            # Compute differences:
            df_agg[:, Symbol(obj)] .= (
                df_agg[:, Symbol(obj)] .- df_agg[:, Symbol(vanilla_name)]
            )
            df_agg.is_pct .= false
            df_agg = filter(
                row ->
                    !ismissing(row[Symbol(vanilla_name)]) &&
                        isfinite(row[Symbol(vanilla_name)]),
                df_agg,
            )
            # Further adjustment
            if !any(df_agg[:, Symbol(vanilla_name)] .== 0) .&&
                !(y in ["validity_strict", "validity", "redundancy"])
                # Compute percentage if only non-zero:
                df_agg[:, Symbol(obj)] .=
                    100 .* df_agg[:, Symbol(obj)] ./ abs.(df_agg[:, Symbol(vanilla_name)])
                df_agg.is_pct .= true
            else
                # Otherwise store average level of baseline:
                df_agg.avg_baseline .= mean(df_agg[:, Symbol(vanilla_name)])
                df_agg.baseline .= df_agg[:, Symbol(vanilla_name)]
                # Let outcome be difference from baseline:
                df_agg[:, Symbol(obj)] .+= df_agg[:, Symbol(vanilla_name)]
            end
        end
        df_agg = DataFrames.stack(
            df_agg[:, Not(Symbol(vanilla_name))],
            Symbol.(other_names);
            value_name=:mean,
            variable_name=:objective,
        )
    end

    # Filter out rows with missing, Inf, or NaN values in the mean column 
    filtered_df = filter(row -> !ismissing(row.mean) && isfinite(row.mean), df_agg)

    return filtered_df
end

function valid_y_ce(df::DataFrame)
    return sort(unique(df.variable))
end

function valid_y_ce(cfg::AbstractEvaluationConfig)
    return valid_y_ce(CTExperiments.load_ce_evaluation(cfg))
end

function aggregate_counterfactuals(eval_grid::EvaluationGrid; kwrgs...)
    eval_list = Logging.with_logger(Logging.NullLogger()) do
        load_list(eval_grid)
    end
    return aggregate_counterfactuals.(eval_list)
end

function aggregate_counterfactuals(
    eval_config::EvaluationConfig; overwrite::Bool=false, nce::Int=1, kwrgs...
)
    bmk = generate_factual_target_pairs(eval_config; overwrite, nce)
    df = innerjoin(bmk.evaluation, bmk.counterfactuals; on=:sample)
    rename!(df, :model => :id)
    all_data = CTExperiments.merge_with_meta(eval_config, df)
    return aggregate_counterfactuals(all_data...; kwrgs...)
end

function aggregate_counterfactuals(
    df::DataFrame,
    df_meta::DataFrame,
    df_eval::DataFrame;
    byvars::Union{Nothing,String,Vector{String}}=nothing,
    byvars_must_include=["factual", "target", "generator_type"],
)
    if isa(byvars, String)
        byvars = [byvars]
    end

    # Assertions:
    @assert byvars isa Nothing || all(col -> col in names(df_meta), byvars) "Columns specified in `byvars` must be one of the following: $(names(df_meta))."

    # Drop redundant:
    select!(df, Not(:variable, :value))

    # Adjust ce:
    if eltype(df.ce) == CounterfactualExplanation
        df.ce .= CounterfactualExplanations.counterfactual.(df.ce)
    end

    # Aggregate:
    return aggregate_data(
        df, "ce", byvars; byvars_must_include=byvars_must_include, agg_fun=:identity
    )
end

function get_img_command(data_names, full_paths, fig_labels; fig_caption="", width=100)
    fig_cap = fig_caption == "" ? fig_caption : "$fig_caption "
    return [
        "![$(fig_cap)Data: $(CTExperiments.get_data_name(nm)).](/$pth){#$(lbl) width=$(width)%}"
        for (nm, pth, lbl) in zip(data_names, full_paths, fig_labels)
    ]
end

global LatexMetricReplacements = Dict(
    "mmd" => "\$ \\text{IP}^* \$",
    "plausibility_distance_from_target" => "\$ \\text{IP} \$",
    "distance" => "Cost",
    "sens_outid:1" => "sens_1",
    "sens_outid:2" => "sens_2",
)

function format_metric(m::String)
    return LatexMetricReplacements[m]
end

function aggregate_ce_evaluation(
    res_dir::String; y="mmd", byvars=nothing, rebase=true, ratio=true, kwrgs...
)
    byvars = gather_byvars(byvars, "data")
    eval_grids, _ = final_results(res_dir)
    df = DataFrame()
    for (i, cfg) in enumerate(eval_grids)
        df_i = aggregate_ce_evaluation(cfg; y, byvars, rebase, ratio, kwrgs...)
        df = vcat(df, df_i)
    end
    rename!(df, :data => :dataset)
    df.dataset .= CTExperiments.format_header.(df.dataset)
    if "objective" in names(df)
        df.objective .= CTExperiments.format_header.(df.objective)
    end
    df.variable .= CTExperiments.format_metric.(y)
    if rebase || ratio
        df.variable .= (x -> LatexCell("$(x) \$(-%)\$")).(df.variable)
        if rebase 
            select!(df, Not([:is_pct, :objective]))
        end
    end
    return df
end

global allowed_perf_measures = Dict("acc" => accuracy, "f1" => multiclass_f1score)

function aggregate_performance(
    res_dir::String; measure::Vector=["acc"], adversarial::Bool=false, bootstrap::Union{Nothing,Int}=nothing,
)

    # Get measures:
    if eltype(measure) == String
        measure = [allowed_perf_measures[m] for m in measure]
    end

    eval_grids, _ = final_results(res_dir)
    df = aggregate_performance(eval_grids; measure, adversarial, bootstrap)
    df.objective .= replace.(df.objective, "Full" => "CT")
    df.objective .= replace.(df.objective, "Vanilla" => "BL")
    df.variable .= replace.(df.variable, "Accuracy" => adversarial ? "Acc.\$^*\$" : "Acc.")
    df.variable .= ["$v ($o)" for (v, o) in zip(df.variable, df.objective)]

    select!(df, Not([:objective]))
    return df
end

function final_results(res_dir::String)

    # Get model and data directories:
    model_dirs = joinpath.(res_dir, readdir(res_dir)) |> x -> x[isdir.(x)]
    data_dirs =
        [joinpath.(d, readdir(d)) |> x -> x[isdir.(x)] for d in model_dirs] |> x -> reduce(vcat, x)

    # Filter out directories with missing results:
    data_dirs = filter(
        x -> isfile(joinpath(x, "evaluation/evaluation_grid_config.toml")), data_dirs
    )
    eval_grids =
        EvaluationGrid.(joinpath.(data_dirs, "evaluation/evaluation_grid_config.toml"))
    data_dirs = filter(x -> isfile(joinpath(x, "grid_config.toml")), data_dirs)
    exper_grids = ExperimentGrid.(joinpath.(data_dirs, "grid_config.toml"))

    return eval_grids, exper_grids
end

function final_table(
    res_dir::String;
    tbl_mtbl::Union{Nothing,DataFrame}=nothing,
    ce_var=["plausibility_distance_from_target", "mmd"],
    perf_var=["acc"],
    agg_further_vars=[["run", "lambda_energy_eval"], ["run", "lambda_energy_eval"]],
    longformat::Bool=true,
    bootstrap::Int=100,
    total_uncertainty::Bool=true,
)
    # CE:
    df_ce = DataFrame()
    for (i, y) in enumerate(ce_var)
        df = aggregate_ce_evaluation(
            res_dir; y, agg_further_vars=agg_further_vars[i], 
            rebase=false, 
            ratio=true,
        )
        df_ce = vcat(df_ce, df; cols=:union)
    end

    # Missing:
    df_ce = coalesce.(df_ce, "(agg.)")

    # Performance:
    df_perf = aggregate_performance(res_dir; measure=perf_var, bootstrap)                          # unperturbed
    df_adv_perf = aggregate_performance(res_dir; measure=perf_var, adversarial=true, bootstrap)    # adversarial
    df_perf = vcat(df_perf, df_adv_perf)
    df = vcat(df_ce, df_perf; cols=:union) |> 
        df -> transform!(df, [:mean, :se] => ((m, s) -> [isnan(si) ? "$(round(mi, digits=2))" : "$(round(mi, digits=2))+-$(round(si, digits=2))" for (mi, si) in zip(m, s)]) => :mean) |>
        df -> select!(df, Not(:se)) |>
        df -> DataFrames.unstack(df, :dataset, :mean)

    # Mutability:
    if !isnothing(tbl_mtbl)
        df = vcat(df, tbl_mtbl; cols=:union)
    end


    # Missing:
    df = coalesce.(df, "")
    rename!(df, :variable => :measure)
    select!(df, :measure, Not([:measure]))

    if longformat
        if "lambda_energy_eval" in names(df)
            df.measure = combine_header.(df.measure, string.(df.lambda_energy_eval))
            select!(df, Not(:lambda_energy_eval))
        end
        df =
            DataFrames.stack(df, Not(:measure)) |>
            df -> DataFrames.unstack(df, :measure, :value)
        rename!(df, :variable => :data)
    end
    return df
end

combine_header(m::String, l::String) = m

function combine_header(m::LatexCell, l::String)
    s = m.data
    if l == "(agg.)"
        new_s = "$s \\\\ (agg.)"
        return new_s
    end
    if l == ""
        return m.data
    end
    new_s = "$s \\\\ \$ \\lambda_{\\text{egy}}=$l \$"
    return new_s
end

function final_mutability(
    res_dir::String;
    var=["distance", "sens_outid:1", "sens_outid:2"],
    byvars=["objective", "mutability"],
    agg_cases::Bool=true,
)
    # CE:
    df_ce = DataFrame()
    for (i, y) in enumerate(var)
        df = aggregate_ce_evaluation(
            res_dir; byvars=byvars, y, agg_further_vars=["run"], rebase=true
        )
        df_ce = vcat(df_ce, df; cols=:union)
    end

    # For sensitivity, need to adjust values cause `aggregate_ce_evaluation` returns levels not changes of zeros are present:
    adjusted_sens = []
    if "avg_baseline" in names(df_ce)
        for (i, val) in enumerate(df_ce.mean)
            if ismissing(df_ce.avg_baseline[i]) || val == 0
                push!(adjusted_sens, val)
            else
                val_bl = df_ce.baseline[i]
                val_pct = 100 .* (val - val_bl) ./ abs.(val_bl)
                push!(adjusted_sens, val_pct)
            end
        end
        select!(df_ce, Not([:avg_baseline, :baseline]))
        df_ce.mean = adjusted_sens
    end

    # Multiply by -1 to turn into -Δ%:
    df_ce.mean .*= -1

    # Aggregate across cases of interest:
    if agg_cases
        filter!(df_ce -> !all(df_ce.mutability .== "both"), df_ce)
        df_ce =
            groupby(df_ce, [:dataset, :variable]) |>
            gdf -> combine(gdf, :mean => mean => :mean)
    end

    df = df_ce
    df = DataFrames.unstack(df, :dataset, :mean)

    return df
end

function final_params(res_dir::String)
    _, exper_grids = final_results(res_dir)
    df = DataFrame()
    for cfg in exper_grids
        df_meta = CTExperiments.expand_grid_to_df(cfg)
        df = vcat(df, df_meta)
    end
    df.n_test = Int.(round.((1 .- df.train_test_ratio) .* df.n_train))
    df.n_train = df.n_train .+ df.n_validation
    df = df[:, [!allequal(x) for x in eachcol(df)]]
    df = select(df, Not([:id, :objective, :n_validation])) |> unique
    dataparams = [:data, :n_train, :n_test, :batchsize]
    select!(df, dataparams, Not(dataparams))
    df.data .= CTExperiments.format_header.(df.data)
    sort!(df, :data)
    return df
end
