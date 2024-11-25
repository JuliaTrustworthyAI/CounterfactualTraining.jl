using CTExperiments
using CounterfactualExplanations
using DotEnv

DotEnv.load!()

# Get config and set up grid:
config_file = joinpath(ENV["EXPERIMENT_DIR"], "poc.toml")
exper_grid = ExperimentGrid(config_file)
exper_list = setup_experiments(exper_grid)

@info "Running $(length(exper_list)) experiments ..."

# Divide the experiments among the available ranks
for (i, experiment) in enumerate(exper_list)

    # Setup:
    save_dir = experiment.meta_params.save_dir
    _name = experiment.meta_params.experiment_name

    # Skip if already finished
    if has_results(experiment)
        @info "Skipping $(_name), model already exists."
        continue
    end

    # Running the experiment
    @info "Running experiment: $(_name) ($i/$(length(exper_list)))"
    model, logs = run_training(experiment; checkpoint_dir=save_dir)

    # Saving the results:
    save_results(experiment, model, logs)
end