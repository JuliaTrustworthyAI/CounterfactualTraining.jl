using CTExperiments
using CTExperiments.CairoMakie
using CTExperiments.CSV
using CTExperiments.DataFrames
using CTExperiments: adjust_plot_var
using DotEnv

DotEnv.load!()

# Get config and set up grid:
eval_grid = EvaluationGrid(get_config_from_args(; new_save_dir=ENV["OUTPUT_DIR"]))
eval_list = load_list(eval_grid)
exper_grid = ExperimentGrid(eval_grid.grid_file)

local_save_dir = get_work_dir(eval_grid, ENV["EVAL_WORK_DIR"], ENV["OUTPUT_DIR"])
output_dir = results_dir(eval_grid)

# Get variables:
colorvar = get_global_param("colorvar", CTExperiments._colorvar)
rowvar = get_global_param("rowvar", CTExperiments._rowvar)
colvar = get_global_param("colvar", CTExperiments._colvar)
colorvar_logs = get_global_param("colorvar_logs", colorvar)
rowvar_logs = get_global_param("rowvar_logs", rowvar)
colvar_logs = get_global_param("colvar_logs", colvar)
colorvar_ce = get_global_param("colorvar_ce", colorvar)
rowvar_ce = get_global_param("rowvar_ce", CTExperiments._rowvar_ce)
colvar_ce = get_global_param("colvar_ce", CTExperiments._colvar_ce)
lnstyvar = get_global_param("lnstyvar", CTExperiments._lnstyvar)
sidevar = get_global_param("sidevar", CTExperiments._sidevar)
dodgevar = get_global_param("dodgevar", colorvar_ce)

global _save_plots = get_global_param("save_plots", true)

# Visualize logs:
prefix = "logs"
valid_y = CTExperiments.valid_y_logs(eval_grid)
params = PlotParams(;
    colorvar=get_global_param("colorvar_logs", colorvar),
    rowvar=get_global_param("rowvar_logs", rowvar),
    colvar=get_global_param("colvar_logs", colvar),
    lnstyvar=get_global_param("lnstyvar", lnstyvar),
)
@info "Logs Errorbars"
@info params
final_save_dir = save_dir(params, output_dir; prefix)
for y in valid_y
    plt, df_agg = plot_errorbar_logs(eval_grid; y=y, params()...)
    display(plt)
    if _save_plots
        save(joinpath(final_save_dir, "$y.png"), plt; px_per_unit=3)
    end
end
if _save_plots
    @info "Images stored in $final_save_dir/"
end

# Visualize CE:
prefix = "ce"
cfg = eval_grid
all_data = CTExperiments.merge_with_meta(
    cfg, CTExperiments.load_ce_evaluation(cfg)
)
valid_y = CTExperiments.valid_y_ce(all_data[1])
params = PlotParams(;
    colorvar=get_global_param("colorvar_ce", colorvar_ce) |> x -> adjust_plot_var(x,cfg),
    rowvar=get_global_param("rowvar_ce", rowvar_ce) |> x -> adjust_plot_var(x, cfg),
    colvar=get_global_param("colvar_ce", colvar_ce) |> x -> adjust_plot_var(x, cfg),
    sidevar=get_global_param("sidevar", sidevar) |> x -> adjust_plot_var(x, cfg),
    dodgevar=get_global_param("dodgevar", dodgevar) |> x -> adjust_plot_var(x, cfg),
)
@info "CE Boxplots"
@info params
final_save_dir = save_dir(params, output_dir; prefix)
for y in valid_y
    plt, tbl = plot_measure_ce(all_data...; y=y, params()...)
    display(plt)
    if _save_plots
        save(joinpath(final_save_dir, "$y.png"), plt; px_per_unit=3)
        CSV.write(joinpath(final_save_dir, "$y.csv"), tbl)
    end
end

# # Plot images:
# try
#     plot_ce(
#         eval_grid;
#         save_dir=final_save_dir,
#         byvars=get_global_param("byvars_ce", CTExperiments._byvars_ce),
#     )
# catch
#     @info "Skipping CE plots for multi-dim data."
# end

# exper_list = load_list(exper_grid)
# eval_list = load_list(eval_grid)
# plot_ce(exper_list; layout=(4, 3))
# plot_ce(exper_list, eval_list[1]; layout=(4, 3), target=2)

# @info "Images stored in $final_save_dir/"
