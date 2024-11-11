using BSON
using CTExperiments
using CTExperiments.CounterfactualExplanations
using DotEnv
using Logging
using MPI
using Serialization

DotEnv.load!()

# Get config and set up grid:
config_file = joinpath(ENV["EXPERIMENT_DIR"], "run_grid_config.toml")
_name = CTExperiments.from_toml(config_file)["name"]
save_dir = joinpath(ENV["OUTPUT_DIR"], _name)
exper_grid = ExperimentGrid(config_file; new_save_dir=save_dir)

# Initialize MPI
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)
if MPI.Comm_rank(MPI.COMM_WORLD) != 0
    global_logger(NullLogger())
else
    # Generate list of experiments and run them:
    exper_list = setup_experiments(exper_grid)
    @info "Running $(length(exper_list)) experiments ..."
end
MPI.Barrier(comm)  # Ensure all processes reach this point before finishing

# Divide the experiments among the available ranks
for (i, experiment) in enumerate(exper_list)
    if mod(i, nprocs) != rank
        continue  # Skip experiments that belong to other ranks
    end

    # Running the experiment
    save_dir = experiment.meta_params.save_dir
    _name = experiment.meta_params.experiment_name
    mname = joinpath(save_dir, "model.jls")
    if isfile(mname)
        @info "Rank $(rank): Skipping $(_name), model already exists."
        continue
    end
    @info "Rank $(rank): Running experiment: $(_name) ($i/$(length(exper_list)))"
    model, logs = run_training(experiment; checkpoint_dir=save_dir)

    # Saving the results:
    save_results(experiment, model, logs)
end

# Finalize MPI
MPI.Barrier(comm)  # Ensure all processes reach this point before finishing
MPI.Finalize()