using CTExperiments
using CTExperiments.CairoMakie
using CTExperiments.DataFrames
using DotEnv

DotEnv.load!()

# Get config and set up grid:
eval_grid = EvaluationGrid(get_config_from_args())
exper_grid = ExperimentGrid(eval_grid.grid_file)

local_save_dir = get_work_dir(eval_grid, ENV["EVAL_WORK_DIR"], ENV["OUTPUT_DIR"])
output_dir = results_dir(eval_grid)

# Visualize logs:
prefix = "logs"
valid_y = CTExperiments.valid_y_logs(eval_grid)

params = PlotParams(; colvar="generator_type")
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

params = PlotParams(; colvar="lambda_energy_eval")
final_save_dir = save_dir(params, output_dir; prefix)
for y in valid_y
    plt = boxplot_ce(all_data...; y=y, params()..., facet=(; linkyaxes=:none))
    display(plt)
    save(joinpath(final_save_dir, "$y.png"), plt; px_per_unit=3)
end

# Plot images:
plot_ce(CTExperiments.MNIST(), eval_grid; save_dir=final_save_dir)
@info "Images stored in $final_save_dir/"
