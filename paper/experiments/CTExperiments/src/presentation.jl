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
        df_agg = combine(df_agg, y => (y -> (mean=y, std=y)) => AsTable)
        return df_agg
    else
        df_agg = combine(df_agg, y => (y -> (mean=agg_fun(y), std=std(y))) => AsTable)
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
        end |>
            dfs -> reduce(vcat, dfs) 
    end
    return df
end

"""
    aggregate_performance(cfg::EvalConfigOrGrid; kwrgs...)

Aggregate performance variable `y` from an experiment grid by columns specified in `byvars`.
"""
function aggregate_performance(cfg::EvalConfigOrGrid; kwrgs...)

    # Load data:
    exper_grid = ExperimentGrid(cfg.grid_file)
    df, df_meta, df_perf = merge_with_meta(cfg, CTExperiments.test_performance(exper_grid; return_df=true))

    return aggregate_performance(df, df_meta, df_perf; kwrgs...)
end

function aggregate_performance(
    df::DataFrame,
    df_meta::DataFrame,
    df_perf::DataFrame;
    y::Union{Nothing,String}=nothing,
    byvars::Union{Nothing,String,Vector{String}}=nothing,
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
        df = df[df.variable .== y,:]
        return select!(df,Not(:variable))
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
    all_data = merge_with_meta(cfg, CTExperiments.load_ce_evaluation(cfg))

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
)
    # Assertions:
    valid_y = valid_y_ce(df)
    @assert y in valid_y "Variable `y` must be one of the following: $valid_y."
    if isa(byvars, String)
        byvars = [byvars]
    end
    @assert isnothing(byvars) || all(col -> col in names(df_meta), byvars) "Columns specified in `byvars` must be one of the following: $(names(df_meta))."

    df = df[df.variable .== y, :]
    rename!(df, :value => y)
    select!(df, Not(:variable))

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
            df_agg =
                groupby(df_agg, byvars) |>
                df -> combine(df, :mean => (y -> (mean=mean(y), std=std(y))) => AsTable)
        end
        df_agg
    else
        df_agg = aggregate_data(df, y, byvars)
    end

    # Subtract from Vanilla:
    if rebase
        @assert "objective" in names(df_agg) "Cannot rebase with respect to 'vanilla' objective is the 'objective' column is not present."
        objectives = unique(df_agg.objective)
        df_agg = DataFrames.unstack(df_agg[:, Not(:std)], :objective, :mean)
        vanilla_name = objectives[lowercase.(objectives) .== "vanilla"][1]
        other_names = objectives[lowercase.(objectives) .!= "vanilla"]
        for obj in other_names

            # Compute differences:
            df_agg[:, Symbol(obj)] .= (
                df_agg[:, Symbol(obj)] .- df_agg[:, Symbol(vanilla_name)]
            )
            df_agg.is_pct .= false
            df_agg = filter(row -> !ismissing(row[Symbol(vanilla_name)]) && isfinite(row[Symbol(vanilla_name)]), df_agg)
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
    eval_list = load_list(eval_grid)
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

function get_img_command(data_names, full_paths, fig_labels; fig_caption="")
    fig_cap = fig_caption == "" ? fig_caption : "$fig_caption "
    return ["![$(fig_cap)Data: $(CTExperiments.get_data_name(nm)).](/$pth){#$(lbl)}" for (nm, pth, lbl) in zip(data_names,full_paths,fig_labels)]
end

function tbl_test_performance(grid::ExperimentGrid; include_adv::Bool=false, kwrgs...)

end

function aggregate_ce_evaluation(res_dir::String; byvars=nothing, kwrgs...)
    byvars = gather_byvars(byvars, "data")
    eval_grids, exper_grids = final_results(res_dir)
    df_agg = DataFrame()
    for (i,cfg) in enumerate(eval_grids)
        df_agg_i = aggregate_ce_evaluation(cfg; byvars=byvars, kwrgs...)
        df_agg = vcat(df_agg, df_agg_i)
    end
    return df_agg
end

function final_results(res_dir::String)

    # Get model and data directories:
    model_dirs = joinpath.(res_dir, readdir(res_dir)) |> x -> x[isdir.(x)]
    data_dirs = [joinpath.(d, readdir(d)) |> x -> x[isdir.(x)] for d in model_dirs] |> x -> reduce(vcat, x)

    # Filter out directories with missing results:
    data_dirs = filter(x -> isfile(joinpath(x,"evaluation/evaluation_grid_config.toml")), data_dirs)
    eval_grids = EvaluationGrid.(joinpath.(data_dirs, "evaluation/evaluation_grid_config.toml"))
    data_dirs = filter(x -> isfile(joinpath(x, "grid_config.toml")), data_dirs)
    exper_grids = ExperimentGrid.(joinpath.(data_dirs, "grid_config.toml"))

    return eval_grids, exper_grids

end