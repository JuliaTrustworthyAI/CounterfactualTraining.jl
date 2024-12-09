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

# Logging configuration
if rank != 0
    global_logger(NullLogger())
end

# Generate list of experiments
exper_list = rank == 0 ? setup_experiments(exper_grid) : nothing

# Broadcast exper_list from rank 0 to all ranks
exper_list = MPI.bcast(exper_list, comm; root=0)

# Compute chunks with a more robust distribution method
function distribute_experiments(exper_list, nprocs)
    # Strategy: Cycle through processes, ensuring even distribution
    chunks = [typeof(exper_list)() for _ in 1:nprocs]
    for (i, experiment) in enumerate(exper_list)
        # Use modulo to cycle through processes
        chunks[mod(i - 1, nprocs) + 1] = push!(chunks[mod(i - 1, nprocs) + 1], experiment)
    end
    return chunks
end

# Distribute experiments
chunks = distribute_experiments(exper_list, nprocs)
worker_chunk = chunks[rank + 1]  # +1 because MPI ranks are 0-indexed but Julia arrays are 1-indexed

# Log some diagnostic information
@info "Rank $(rank): Assigned $(length(worker_chunk)) experiments out of $(length(exper_list)) total"

# Main processing loop
for (i, experiment) in enumerate(worker_chunk)
    # Skip logging suppression for rank 0
    if rank != 0
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
    @info "Rank $(rank): Running experiment: $(_name) ($i/$(length(worker_chunk)))"
    println("Saving checkpoints in: ", _save_dir)

    try
        model, logs = run_training(experiment; checkpoint_dir=_save_dir)
        # Saving the results:
        save_results(experiment, model, logs)
    catch e
        @error "Rank $(rank): Error in experiment $(_name)" exception = (
            e, catch_backtrace()
        )
    end
end

# Ensure all processes finish
MPI.Barrier(comm)
MPI.Finalize()