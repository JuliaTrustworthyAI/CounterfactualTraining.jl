using CTExperiments
using CTExperiments.CairoMakie
using CTExperiments.DataFrames
using DotEnv

DotEnv.load!()

# Get config and set up grid:
eval_grid = EvaluationGrid(get_config_from_args())
eval_list = load_list(eval_grid)
exper_grid = ExperimentGrid(eval_grid.grid_file)
exper_list = load_list(exper_grid)

local_save_dir = get_work_dir(eval_grid, ENV["EVAL_WORK_DIR"])
output_dir = results_dir(eval_grid)

# Visualize logs:
prefix = "logs"
valid_y = CTExperiments.valid_y_logs(eval_grid)

params = PlotParams(; colvar="generator_type", rowvar="objective")
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

params = PlotParams(; rowvar="lambda_energy_eval", colvar="objective")
final_save_dir = save_dir(params, output_dir; prefix)
for y in valid_y
    plt = boxplot_ce(all_data...; y=y, params()..., facet=(; linkyaxes=:minimal))
    display(plt)
    save(joinpath(final_save_dir, "$y.png"), plt; px_per_unit=3)
end

# Plot images:
plot_ce(eval_grid; save_dir=final_save_dir, byvars=["objective"])
plot_ce(exper_list)
plot_ce(exper_list, eval_list[1])

@info "Images stored in $final_save_dir/"
