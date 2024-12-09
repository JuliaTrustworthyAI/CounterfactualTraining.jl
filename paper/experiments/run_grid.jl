using BSON
using CTExperiments
using CounterfactualExplanations
using DotEnv
using Logging
using MPI
using Serialization
using TaijaParallel

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

# Only process 0 generates the experiment list
if rank == 0
    exper_list = setup_experiments(exper_grid)
    @info "Running $(length(exper_list)) experiments ..."
else
    exper_list = nothing
end

# Broadcast exper_list from rank 0 to all ranks
exper_list = MPI.bcast(exper_list, comm; root=0)

# Custom distribution logic
local_experiments = if rank == 0
    # Distribute experiments across processes
    chunks = TaijaParallel.split_obs(exper_list, nprocs)
    chunks[rank + 1]  # Julia is 1-indexed, MPI is 0-indexed
else
    Experiment[]  # Empty list for other ranks
end

# Scatter the experiment chunks using serialization
if rank == 0
    @info "Total processes: $nprocs"
    @info "Total experiments: $(length(exper_list))"
    chunk_lengths = [length(chunk) for chunk in TaijaParallel.split_obs(exper_list, nprocs)]
    @info "Chunk lengths: $chunk_lengths"
end

# Process experiments for this rank
if !isempty(local_experiments)
    for (i, experiment) in enumerate(local_experiments)
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
        @info "Rank $(rank): Running experiment: $(_name) ($i/$(length(local_experiments)))"
        println("Saving checkpoints in: ", _save_dir)
        model, logs = run_training(experiment; checkpoint_dir=_save_dir)

        # Saving the results:
        save_results(experiment, model, logs)
    end
else
    @info "Rank $(rank): No experiments to process"
end

# Finalize MPI
MPI.Barrier(comm)  # Ensure all processes reach this point before finishing
MPI.Finalize()