using BSON
using CTExperiments
using CounterfactualExplanations
using DotEnv
using Logging
using MPI
using Serialization

DotEnv.load!()

# Get config and set up grid:
config_file = get_config_from_args()
root_name = CTExperiments.from_toml(config_file)["name"]
root_save_dir = joinpath(ENV["OUTPUT_DIR"], root_name)
exper_grid = ExperimentGrid(config_file; new_save_dir=root_save_dir)

# Initialize MPI
MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)
if MPI.Comm_rank(MPI.COMM_WORLD) != 0
    global_logger(NullLogger())
    exper_list = nothing
else
    # Generate list of experiments and run them:
    exper_list = setup_experiments(exper_grid)
    @info "Running $(length(exper_list)) experiments ..."
end

# Broadcast exper_list from rank 0 to all ranks
exper_list = MPI.bcast(exper_list, comm; root=0)

MPI.Barrier(comm)  # Ensure all processes reach this point before finishing

@assert length(exper_list) >= nprocs "Ensure there are enough experiments to distribute across all ranks"
chunks = TaijaParallel.split_obs(exper_list, nprocs)     # distribute across processes

for (i, chunk) in enumerate(chunks)

    if i != rank 
        continue    # Skip experiments that belong to other ranks
    end

    # Divide the experiments among the available ranks
    for experiment in chunk

        if rank != 0
            # Shut up logging for other ranks to avoid cluttering output
            CTExperiments.shutup!(experiment.training_params)
        end

        # Setup:
        _save_dir = experiment.meta_params.save_dir
        _name = experiment.meta_params.experiment_name


        # Skip if already finished
        if has_results(experiment)
            @info "Rank $(rank): Skipping $(_name), model already exists."
            continue
        end

        # Running the experiment
        @info "Rank $(rank): Running experiment: $(_name) ($i/$(length(exper_list)))"
        model, logs = run_training(experiment; checkpoint_dir=_save_dir)

        # Saving the results:
        save_results(experiment, model, logs)
    end

end

# Finalize MPI
MPI.Barrier(comm)  # Ensure all processes reach this point before finishing
