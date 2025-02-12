using CTExperiments
using DotEnv

# Setup:
DotEnv.load!()
set_global_seed()

# Get config and set up grid:
grid_file = get_config_from_args(; new_save_dir=ENV["OUTPUT_DIR"])
eval_grid = EvaluationGrid(grid_file)
eval_list = generate_list(eval_grid)

for (i, eval_config) in enumerate(eval_list)
    # Generate factual target pairs for plotting:
    generate_factual_target_pairs(eval_config; overwrite=true)
end
