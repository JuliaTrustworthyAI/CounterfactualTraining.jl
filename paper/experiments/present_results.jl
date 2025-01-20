using CTExperiments
using CTExperiments.CairoMakie
using CTExperiments.DataFrames
using DotEnv

DotEnv.load!()

# Get config and set up grid:
eval_grid = EvaluationGrid(get_config_from_args(; new_save_dir=ENV["OUTPUT_DIR"]))
exper_grid = ExperimentGrid(eval_grid.grid_file)

local_save_dir = get_work_dir(eval_grid, ENV["EVAL_WORK_DIR"], ENV["OUTPUT_DIR"])
output_dir = results_dir(eval_grid)

# Get variables:
colorvar = get_global_param("colorvar", nothing)
rowvar = get_global_param("rowvar", nothing)
colvar = get_global_param("colvar", nothing)
colorvar_logs = get_global_param("colorvar_logs", colorvar)
rowvar_logs = get_global_param("rowvar_logs", rowvar)
colvar_logs = get_global_param("colvar_logs", colvar)
colorvar_ce = get_global_param("colorvar_ce", colorvar)
rowvar_ce = get_global_param("rowvar_ce", rowvar)
colvar_ce = get_global_param("colvar_ce", colvar)


# Visualize logs:
prefix = "logs"
valid_y = CTExperiments.valid_y_logs(eval_grid)
params = PlotParams(;
    colorvar=get_global_param("colorvar_logs", colorvar),
    rowvar=get_global_param("rowvar_logs", rowvar),
    colvar=get_global_param("colvar_logs", colvar),
)
final_save_dir = save_dir(params, output_dir; prefix)
for y in valid_y
    plt = plot_errorbar_logs(eval_grid; y=y, params()...)
    display(plt)
    save(joinpath(final_save_dir, "$y.png"), plt; px_per_unit=3)
end
@info "Images stored in $final_save_dir/"

# Visualize CE:
prefix = "ce"
all_data = CTExperiments.merge_with_meta(
    eval_grid, CTExperiments.load_ce_evaluation(eval_grid)
)
valid_y = CTExperiments.valid_y_ce(all_data[1])
params = PlotParams(;
    colorvar=get_global_param("colorvar_ce", colorvar),
    rowvar=get_global_param("rowvar_ce", rowvar),
    colvar=get_global_param("colvar_ce", colvar),
)
final_save_dir = save_dir(params, output_dir; prefix)
for y in valid_y
    plt = boxplot_ce(all_data...; y=y, params()...)
    display(plt)
    save(joinpath(final_save_dir, "$y.png"), plt; px_per_unit=3)
end

# # Plot images:
# exper_list = load_list(exper_grid)
# eval_list = load_list(eval_grid)
# plot_ce(exper_list; layout=(4, 3))
# plot_ce(exper_list, eval_list[1]; layout=(4, 3), target=2)
plot_ce(eval_grid; save_dir=final_save_dir, byvars=["mutability"])

@info "Images stored in $final_save_dir/"
