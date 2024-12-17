using BSON
using CTExperiments
using CounterfactualExplanations
using DotEnv
using Logging
using MPI
using Serialization
using TaijaParallel

# Setup:
DotEnv.load!()
set_global_seed()

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

if length(exper_list) < nprocs
    @warn "There are less experiments ($(length(exper_list))) than processes ($(nprocs)). Check CPU efficiency of job."
end
chunks = TaijaParallel.split_obs(exper_list, nprocs)    # split experiments into chunks for each process

# Set up dummies for processes without tasks to avoid deadlock:
max_chunk_size = maximum(length.(chunks))
chunks = Logging.with_logger(Logging.NullLogger()) do
    for (i, chunk) in enumerate(chunks)
        if length(chunk) < max_chunk_size
            n_missing = max_chunk_size - length(chunk)
            for j in 1:n_missing
                exper = deepcopy(exper_list[1])
                make_dummy(exper, i, j)
                push!(chunk, exper)
            end
        end
    end
    return chunks
end

@assert allequal(length.(chunks)) "Need all processes to have the same number of tasks."

worker_chunk = MPI.scatter(chunks, comm)                # distribute across processes

for (i, experiment) in enumerate(worker_chunk)
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
    @info "Rank $(rank): Running experiment: $(_name) ($i/$(length(worker_chunk)))"

    if rank == 0
        @info "Memory usage:"
        meminfo_julia()
    end
    model, logs = run_training(experiment; checkpoint_dir=_save_dir)

    # Saving the results:
    if !isdummy(experiment)  # Avoid overwriting dummy results
        save_results(experiment, model, logs)
    else
        remove_dummy!(experiment)
    end
end

# Finalize MPI
MPI.Barrier(comm)  # Ensure all processes reach this point before finishing
if mpi_should_finalize()
    MPI.Finalize()          # finalize MPI
end
