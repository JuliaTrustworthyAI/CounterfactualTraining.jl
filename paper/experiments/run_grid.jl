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

# Configure logging
if rank != 0
    global_logger(NullLogger())
end

# Generate experiment list only on rank 0
exper_list = if rank == 0
    result = setup_experiments(exper_grid)
    @info "Running $(length(result)) experiments ..."
    result
else
    nothing
end

# Broadcast exper_list from rank 0 to all ranks
exper_list = MPI.bcast(exper_list, comm; root=0)

# Warn about process efficiency if needed
if rank == 0 && length(exper_list) < nprocs
    @warn "There are less experiments than processes. Check CPU efficiency of job."
end

# Distribute experiments across processes
# This approach ensures even distribution and handles cases with fewer experiments than processes
function distribute_experiments(exper_list, nprocs)
    # Compute chunks for each process
    chunks = TaijaParallel.split_obs(exper_list, nprocs)

    # Each process gets its chunk
    worker_chunk = MPI.scatter(chunks, comm)

    return worker_chunk
end

# Distribute and process experiments
worker_chunk = distribute_experiments(exper_list, nprocs)

# Process experiments for this process
for (i, experiment) in enumerate(worker_chunk)
    # Shut up logging for other ranks to avoid cluttering output
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
    catch err
        @error "Rank $(rank): Error in experiment $(_name)" exception = (
            err, catch_backtrace()
        )
    end
end

# Finalize MPI
MPI.Barrier(comm)  # Ensure all processes reach this point before finishing