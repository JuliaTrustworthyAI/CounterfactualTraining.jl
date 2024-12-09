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

if rank != 0
    global_logger(NullLogger())
    exper_list = nothing
else
    # Generate list of experiments and run them:
    exper_list = setup_experiments(exper_grid)
    @info "Running $(length(exper_list)) experiments ..."
end

# Broadcast exper_list from rank 0 to all ranks
exper_list = MPI.bcast(exper_list, comm; root=0)

# Number of experiments (outer loop size)
m = length(exper_list)

if m < nprocs
    @warn "There are fewer experiments ($(m)) than processes ($(nprocs)). Check CPU efficiency of job."
end

# Assign each rank to an outer loop task based on `rank % m`
outer_task_id = rank % m

# Split communicator by outer loop task
sub_comm = MPI.Comm_split(comm, outer_task_id, rank)
sub_rank = MPI.Comm_rank(sub_comm)
sub_nprocs = MPI.Comm_size(sub_comm)

# Each sub-communicator processes its corresponding outer-loop task
if outer_task_id < m
    experiment = exper_list[outer_task_id + 1]  # Outer loop task assigned to this communicator

    if sub_rank == 0
        @info "Processing experiment $outer_task_id with $sub_nprocs processes in sub-communicator."
    end

    if sub_rank != 0
        # Shut up logging for non-root ranks of sub-communicator
        CTExperiments.shutup!(experiment.training_params)
    end

    # Setup:
    _save_dir = experiment.meta_params.save_dir
    _name = experiment.meta_params.experiment_name

    # Skip if already finished
    if has_results(experiment)
        if sub_rank == 0
            @info "Experiment $_name is already finished. Skipping."
        end
    else
        # Run the experiment
        if sub_rank == 0
            @info "Running experiment $_name (Outer Task ID $outer_task_id)."
        end
        model, logs = run_training(experiment; checkpoint_dir=_save_dir)

        # Save the results
        if sub_rank == 0
            save_results(experiment, model, logs)
        end
    end
end

# Finalize MPI
MPI.Barrier(comm)