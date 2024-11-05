using BSON
using CTExperiments
using CTExperiments.CounterfactualExplanations
using DotEnv
using Serialization

DotEnv.load!()

# Get config and set up grid:
config_file = joinpath(ENV["EXPERIMENT_DIR"], "grid_config.toml")
exper_grid = ExperimentGrid(config_file)

# Save grid config to output folder:
grid_save_dir = joinpath(ENV["OUTPUT_DIR"], exper_grid.name)
mkpath(grid_save_dir)
config_name = joinpath(grid_save_dir, "config.toml")
CTExperiments.to_toml(exper_grid, config_name)

# Generate list of experiments and run them:
exper_list = setup_experiments(exper_grid)
@info "Running $(length(exper_list)) experiments ..."
for (i, experiment) in enumerate(exper_list)

    # Saving the config file (for the ith experiment)
    name = experiment.meta_params.experiment_name
    save_dir = joinpath(grid_save_dir, "$(name)")
    mkpath(save_dir)
    config_name = joinpath(save_dir, "config.toml")
    CTExperiments.to_toml(experiment, config_name)

    # Running the experiment
    mname = joinpath(save_dir, "model.jls")
    if isfile(mname)
        @info "Skipping $(name), model already exists."
        continue
    end
    @info "Running experiment: $(name) ($i/$(length(exper_list)))"
    model, logs = run_training(experiment)

    # Saving the results:
    logs_name = joinpath(save_dir, "logs.jls")
    bson_name = joinpath(save_dir, "model.bson")
    M = MLP(model; likelihood=:classification_multi)
    @info "Saving model and logs to $(save_dir):"
    serialize(mname, M)
    serialize(logs_name, logs)
    BSON.@save bson_name model
end
