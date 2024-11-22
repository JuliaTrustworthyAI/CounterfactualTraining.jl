using CTExperiments
using CTExperiments.DataFrames
using CTExperiments.Makie
using DotEnv

DotEnv.load!()

# Get config and set up grid:
eval_config = EvaluationConfig(
    joinpath(ENV["EVAL_WORK_DIR"], "grid_run_2024-11-13_ablation", "eval_config.toml")
)
exper_grid = ExperimentGrid(eval_config.grid_file)
df_meta = CTExperiments.expand_grid_to_df(exper_grid)
local_save_dir = get_work_dir(eval_config, ENV["EVAL_WORK_DIR"])
output_dir = results_dir(cfg)

# Visualize logs:
prefix = "logs"
valid_y = CTExperiments.valid_y_logs(eval_config)

params = PlotParams(; rowvar="lambda_energy", colorvar="objective", colvar="generator_type")
final_save_dir = save_dir(params, output_dir; prefix)
for y in valid_y
    plt = plot_errorbar_logs(eval_config; y=y, params()...)
    display(plt)
    save(joinpath(final_save_dir, "$y.png"), plt; px_per_unit=3)
end
@info "Images stored in $final_save_dir/"

# Visualize CE:
prefix = "ce"
all_data = CTExperiments.merge_with_meta(
    eval_config, CTExperiments.load_ce_evaluation(eval_config)
)
valid_y = CTExperiments.valid_y_ce(all_data[1])

params = PlotParams(; colvar="lambda_energy", colorvar="objective")
final_save_dir = save_dir(params, output_dir; prefix)
for y in valid_y
    plt = boxplot_ce(all_data...; y=y, params()...)
    display(plt)
    save(joinpath(final_save_dir, "$y.png"), plt; px_per_unit=3)
end
@info "Images stored in $final_save_dir/"
